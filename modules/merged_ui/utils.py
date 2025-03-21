import datetime
import time
import logging
import os
import platform
import shutil
import re
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import torch
import threading
from queue import Queue, Empty
from contextlib import nullcontext

# Import modules from your packages
from spark.cli.SparkTTS import SparkTTS
from rvc_ui.initialization import vc

# Initialize the Spark TTS model (moved outside function to avoid reinitializing)
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0

def initialize_model(model_dir, device):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")

    # Determine appropriate device based on platform and availability
    if platform.system() == "Darwin":
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    model = SparkTTS(model_dir, device)
    return model


def run_tts(
    text,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="TEMP/spark",  
    save_filename=None,
    cuda_stream=None,
):
    """Perform TTS inference using a specific CUDA stream."""
    model = initialize_model(model_dir, device=device)
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Determine the save path
    if save_filename:
        save_path = os.path.join(save_dir, save_filename)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting TTS inference...")

    # Perform inference using the specified CUDA stream
    with torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext():
        with torch.no_grad():
            wav = model.inference(
                text,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,
            )
    
    # Save the audio (CPU operation)
    sf.write(save_path, wav, samplerate=16000)
        
    logging.info(f"TTS audio saved at: {save_path}")
    return save_path


def process_with_rvc(
    spk_item, input_path, vc_transform, f0method,
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect, 
    output_path, cuda_stream=None
):
    """Process audio through RVC with a specific CUDA stream."""
    logging.info(f"Starting RVC inference for {input_path}...")
    
    # Set the CUDA stream if provided
    with torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext():
        # Call RVC processing function
        f0_file = None  # We're not using an F0 curve file
        output_info, output_audio = vc.vc_single(
            spk_item, input_path, vc_transform, f0_file, f0method,
            file_index1, file_index2, index_rate, filter_radius,
            resample_sr, rms_mix_rate, protect
        )
    
    # Save RVC output (CPU operation)
    rvc_saved = False
    try:
        if isinstance(output_audio, str) and os.path.exists(output_audio):
            # Case 1: output_audio is a file path string
            shutil.copy2(output_audio, output_path)
            rvc_saved = True
        elif isinstance(output_audio, tuple) and len(output_audio) >= 2:
            # Case 2: output_audio might be (sample_rate, audio_data)
            sf.write(output_path, output_audio[1], output_audio[0])
            rvc_saved = True
        elif hasattr(output_audio, 'name') and os.path.exists(output_audio.name):
            # Case 3: output_audio might be a file-like object
            shutil.copy2(output_audio.name, output_path)
            rvc_saved = True
    except Exception as e:
        output_info += f"\nError saving RVC output: {str(e)}"
        
    logging.info(f"RVC inference completed for {input_path}")
    return rvc_saved, output_info


def split_into_sentences(text):
    """
    Split text into sentences using regular expressions.
    
    Args:
        text (str): The input text to split
        
    Returns:
        list: A list of sentences
    """
    # Split on period, exclamation mark, or question mark followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
    # Remove any empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def concatenate_audio_files(file_paths, output_path, sample_rate=44100):
    """
    Concatenate multiple audio files into a single file
    
    Args:
        file_paths (list): List of paths to audio files
        output_path (str): Path to save the concatenated audio
        sample_rate (int): Sample rate for the output file
        
    Returns:
        bool: True if concatenation was successful, False otherwise
    """
    try:
        # Use pydub to concatenate audio files
        combined = AudioSegment.empty()
        for file_path in file_paths:
            segment = AudioSegment.from_file(file_path)
            combined += segment
        
        # Export the combined audio
        combined.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error concatenating audio files: {str(e)}")
        
        # Fallback method using soundfile
        try:
            all_audio = []
            for file_path in file_paths:
                data, sr = sf.read(file_path)
                # Convert to mono if stereo
                if len(data.shape) > 1 and data.shape[1] > 1:
                    data = data.mean(axis=1)
                all_audio.append(data)
            
            # Concatenate all audio data
            concatenated = np.concatenate(all_audio)
            sf.write(output_path, concatenated, sample_rate)
            return True
        except Exception as e2:
            print(f"Fallback concatenation failed: {str(e2)}")
            return False


def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""
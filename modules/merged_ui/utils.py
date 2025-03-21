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
from pydub import AudioSegment

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
    save_dir="TEMP/spark",  # Updated default save directory
    save_filename=None,      # New parameter to specify filename
):
    """Perform TTS inference and save the generated audio."""
    model = initialize_model(model_dir, device=device)
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Determine the save path based on save_filename if provided; otherwise, use a timestamp
    if save_filename:
        save_path = os.path.join(save_dir, save_filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")
    return save_path

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

def process_single_sentence(
    sentence_index, sentence, prompt_speech, prompt_text_clean,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect,
    base_fragment_num
):
    """
    Process a single sentence through the TTS and RVC pipeline.
    """
    fragment_num = base_fragment_num + sentence_index

    # Generate TTS audio for this sentence, saving directly to the correct location
    tts_path = run_tts(
        sentence,
        prompt_text=prompt_text_clean,
        prompt_speech=prompt_speech,
        save_dir="./TEMP/spark",
        save_filename=f"fragment_{fragment_num}.wav"
    )

    # Make sure we have a TTS file to process
    if not tts_path or not os.path.exists(tts_path):
        return None, None, False, f"Failed to generate TTS audio for sentence: {sentence}"

    # Use the tts_path as the Spark output (no need to copy)
    spark_output_path = tts_path

    # Call RVC processing function
    f0_file = None  # We're not using an F0 curve file in this pipeline
    output_info, output_audio = vc.vc_single(
        spk_item, tts_path, vc_transform, f0_file, f0method,
        file_index1, file_index2, index_rate, filter_radius,
        resample_sr, rms_mix_rate, protect
    )

    # Save RVC output to TEMP/rvc directory
    rvc_output_path = f"./TEMP/rvc/fragment_{fragment_num}.wav"
    rvc_saved = False

    try:
        if isinstance(output_audio, str) and os.path.exists(output_audio):
            # Case 1: output_audio is a file path string
            shutil.copy2(output_audio, rvc_output_path)
            rvc_saved = True
        elif isinstance(output_audio, tuple) and len(output_audio) >= 2:
            # Case 2: output_audio might be (sample_rate, audio_data)
            try:
                sf.write(rvc_output_path, output_audio[1], output_audio[0])
                rvc_saved = True
            except Exception as inner_e:
                output_info += f"\nFailed to save RVC tuple format: {str(inner_e)}"
        elif hasattr(output_audio, 'name') and os.path.exists(output_audio.name):
            # Case 3: output_audio might be a file-like object
            shutil.copy2(output_audio.name, rvc_output_path)
            rvc_saved = True
    except Exception as e:
        output_info += f"\nError saving RVC output: {str(e)}"

    # Prepare info message
    info_message = f"Sentence {sentence_index+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
    info_message += f"  - Spark output: {spark_output_path}\n"
    if rvc_saved:
        info_message += f"  - RVC output: {rvc_output_path}"
    else:
        info_message += f"  - Could not save RVC output to {rvc_output_path}"

    return spark_output_path, rvc_output_path, rvc_saved, info_message

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

def generate_and_process_with_rvc(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect
):
    """
    Handle combined TTS and RVC processing for multiple sentences and yield outputs as they are processed.
    The output is just the latest processed audio.
    Before yielding a new audio fragment, the function waits for the previous one to finish playing,
    based on its duration.
    """
    # Ensure TEMP directories exist
    os.makedirs("./TEMP/spark", exist_ok=True)
    os.makedirs("./TEMP/rvc", exist_ok=True)
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    if not sentences:
        yield "No valid text to process.", None
        return
    
    # Get next base fragment number
    base_fragment_num = 1
    while any(os.path.exists(f"./TEMP/spark/fragment_{base_fragment_num + i}.wav") or 
              os.path.exists(f"./TEMP/rvc/fragment_{base_fragment_num + i}.wav") 
              for i in range(len(sentences))):
        base_fragment_num += 1
    
    # Process reference speech
    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
    
    info_messages = [f"Processing {len(sentences)} sentences..."]
    
    # Yield initial message with no audio yet
    yield "\n".join(info_messages), None

    # Set up a timer to simulate playback duration
    next_available_time = time.time()

    for i, sentence in enumerate(sentences):
        spark_path, rvc_path, success, info = process_single_sentence(
            i, sentence, prompt_speech, prompt_text_clean,
            spk_item, vc_transform, f0method, 
            file_index1, file_index2, index_rate, filter_radius,
            resample_sr, rms_mix_rate, protect,
            base_fragment_num
        )
        
        info_messages.append(info)
        # Only update output if processing was successful and we have an audio file
        if success and rvc_path:
            try:
                audio_seg = AudioSegment.from_file(rvc_path)
                duration = audio_seg.duration_seconds
            except Exception as e:
                duration = 0

            current_time = time.time()
            if current_time < next_available_time:
                time.sleep(next_available_time - current_time)
            
            yield "\n".join(info_messages), rvc_path
            
            next_available_time = time.time() + duration

    yield "\n".join(info_messages), rvc_path

def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""
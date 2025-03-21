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


def generate_and_process_with_rvc_parallel(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect,
    num_tts_workers=2  # New parameter to control number of parallel TTS instances
):
    """
    Handle combined TTS and RVC processing using multiple TTS instances in parallel.
    Uses a producer-consumer pattern where multiple TTS workers produce audio files for RVC to consume.
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
    
    # Create CUDA streams if CUDA is available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Create multiple TTS streams and one RVC stream
        tts_streams = [torch.cuda.Stream() for _ in range(num_tts_workers)]
        rvc_stream = torch.cuda.Stream()
        logging.info(f"Using {num_tts_workers} CUDA streams for Spark TTS and 1 for RVC")
    else:
        tts_streams = [None] * num_tts_workers
        rvc_stream = None
        logging.info("CUDA not available, parallel processing will be limited")
    
    # Create queues for communication between TTS and RVC
    tts_to_rvc_queue = Queue()
    rvc_results_queue = Queue()
    
    # Flags to signal completion
    tts_complete = [threading.Event() for _ in range(num_tts_workers)]
    processing_complete = threading.Event()
    
    info_messages = [f"Processing {len(sentences)} sentences using {num_tts_workers} parallel TTS instances..."]
    
    # Yield initial message with no audio yet
    yield "\n".join(info_messages), None
    
    # Split sentences into batches for each TTS worker - using contiguous chunks
    sentence_batches = []
    batch_size = len(sentences) // num_tts_workers
    remainder = len(sentences) % num_tts_workers
    
    start_idx = 0
    for i in range(num_tts_workers):
        # Add one extra sentence to the first 'remainder' batches
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        
        batch = sentences[start_idx:end_idx]
        batch_indices = list(range(start_idx, end_idx))
        sentence_batches.append((batch, batch_indices))
        
        start_idx = end_idx
    
    # Modified TTS worker function that initializes the model once
    def tts_worker(worker_id, sentences_batch, global_indices, cuda_stream):
        # Initialize the TTS model once for this worker
        logging.info(f"TTS Worker {worker_id}: Initializing Spark TTS model")
        
        # Determine device based on platform
        if platform.system() == "Darwin":
            model_device = torch.device(f"mps:{device}")
        elif torch.cuda.is_available():
            model_device = torch.device(f"cuda:{device}")
        else:
            model_device = torch.device("cpu")
            
        # Initialize model once for this worker
        tts_model = SparkTTS(model_dir, model_device)
        
        # Process each sentence in this worker's batch
        for local_idx, (sentence, global_idx) in enumerate(zip(sentences_batch, global_indices)):
            fragment_num = base_fragment_num + global_idx
            tts_filename = f"fragment_{fragment_num}.wav"
            save_path = os.path.join("./TEMP/spark", tts_filename)
            
            try:
                logging.info(f"TTS Worker {worker_id}: Processing text: {sentence[:30]}...")
                
                # Use the TTS CUDA stream assigned to this worker
                stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext()
                with stream_ctx:
                    with torch.no_grad():
                        wav = tts_model.inference(
                            sentence,
                            prompt_speech,
                            prompt_text_clean,
                            None,  # gender
                            None,  # pitch
                            None,  # speed
                        )
                
                # Save the audio (CPU operation)
                sf.write(save_path, wav, samplerate=16000)
                logging.info(f"TTS Worker {worker_id}: Audio saved at: {save_path}")
                
                # Put the path and sentence info to the queue for RVC processing
                tts_to_rvc_queue.put((global_idx, fragment_num, sentence, save_path))
            except Exception as e:
                logging.error(f"TTS worker {worker_id} processing error for sentence {global_idx}: {str(e)}")
                tts_to_rvc_queue.put((global_idx, fragment_num, sentence, None, str(e)))
        
        # Signal this TTS worker's completion
        logging.info(f"TTS Worker {worker_id}: Completed processing {len(sentences_batch)} sentences")
        tts_complete[worker_id].set()
        
        # If all TTS workers are done, add the sentinel to the queue
        if all(event.is_set() for event in tts_complete):
            tts_to_rvc_queue.put(None)
    
    # RVC worker function
    def rvc_worker():
        while True:
            # Get item from the queue
            item = tts_to_rvc_queue.get()
            
            # Check for the sentinel value (None) that signals completion
            if item is None:
                break
                
            # Unpack the item
            if len(item) == 5:  # Error case
                i, fragment_num, sentence, _, error = item
                rvc_results_queue.put((i, None, None, False, f"TTS error for sentence {i+1}: {error}"))
                continue
                
            i, fragment_num, sentence, tts_path = item
            
            if not tts_path or not os.path.exists(tts_path):
                rvc_results_queue.put((i, None, None, False, f"No TTS output for sentence {i+1}"))
                continue
            
            # Prepare RVC path
            rvc_path = os.path.join("./TEMP/rvc", f"fragment_{fragment_num}.wav")
            
            try:
                # Process with RVC
                rvc_success, rvc_info = process_with_rvc(
                    spk_item, tts_path, vc_transform, f0method,
                    file_index1, file_index2, index_rate, filter_radius,
                    resample_sr, rms_mix_rate, protect,
                    rvc_path, cuda_stream=rvc_stream
                )
                
                # Create info message
                info_message = f"Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
                info_message += f"  - Spark output: {tts_path}\n"
                if rvc_success:
                    info_message += f"  - RVC output: {rvc_path}"
                else:
                    info_message += f"  - Could not save RVC output to {rvc_path}"
                
                # Put the results to the queue
                rvc_results_queue.put((i, tts_path, rvc_path if rvc_success else None, rvc_success, info_message))
            except Exception as e:
                logging.error(f"RVC processing error for sentence {i}: {str(e)}")
                info_message = f"Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
                info_message += f"  - Spark output: {tts_path}\n"
                info_message += f"  - RVC processing error: {str(e)}"
                rvc_results_queue.put((i, tts_path, None, False, info_message))
                
        # Signal RVC completion
        processing_complete.set()
    
    # Start multiple TTS worker threads and one RVC thread
    tts_threads = []
    for i in range(num_tts_workers):
        if sentence_batches[i][0]:  # Only start if there are sentences in this batch
            thread = threading.Thread(
                target=tts_worker, 
                args=(i, sentence_batches[i][0], sentence_batches[i][1], tts_streams[i])
            )
            tts_threads.append(thread)
            thread.start()
        else:
            # Mark this worker as complete if it has no work
            tts_complete[i].set()
    
    rvc_thread = threading.Thread(target=rvc_worker)
    rvc_thread.start()
    
    # Process results as they become available
    completed_sentences = {}
    next_to_yield = 0
    
    while not processing_complete.is_set() or not rvc_results_queue.empty():
        try:
            # Try to get an item from the results queue with a timeout
            try:
                i, tts_path, rvc_path, success, info = rvc_results_queue.get(timeout=0.1)
                completed_sentences[i] = (tts_path, rvc_path, success, info)
            except Empty:
                # No results available yet, continue the loop
                continue
                
            # Check if we can yield the next sentence
            while next_to_yield in completed_sentences:
                _, rvc_path, _, info = completed_sentences[next_to_yield]
                info_messages.append(info)
                
                # Yield the current state
                yield "\n".join(info_messages), rvc_path
                
                # Move to the next sentence
                next_to_yield += 1
        except Exception as e:
            logging.error(f"Error in main processing loop: {str(e)}")
            info_messages.append(f"Error in processing: {str(e)}")
            yield "\n".join(info_messages), None
            break
    
    # Join the threads
    for thread in tts_threads:
        thread.join()
    rvc_thread.join()
    
    # Yield any remaining sentences in order
    remaining_indices = sorted([i for i in completed_sentences if i >= next_to_yield])
    for i in remaining_indices:
        _, rvc_path, _, info = completed_sentences[i]
        info_messages.append(info)
        yield "\n".join(info_messages), rvc_path


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
import os
import shutil
import platform
import logging
from contextlib import nullcontext
from queue import Empty
import torch
import soundfile as sf
import time

from spark.cli.SparkTTS import SparkTTS
from rvc_ui.initialization import vc

def persistent_tts_worker(worker_id, job_queue, active_event, model_dir, device, model_manager):
    """
    Persistent TTS worker that stays alive and processes jobs from the queue.
    
    Args:
        worker_id (int): Unique worker ID
        job_queue (Queue): Queue for receiving jobs
        active_event (Event): Event to signal when worker is active/idle
        model_dir (str): Path to TTS model directory
        device (str): Device to use for inference
        model_manager (ModelManager): Reference to the model manager
    """
    logging.info(f"Persistent TTS Worker {worker_id}: Initializing Spark TTS model")
    
    # Determine proper device based on platform and availability
    if platform.system() == "Darwin":
        model_device = torch.device(f"mps:{device}")
    elif torch.cuda.is_available():
        model_device = torch.device(f"cuda:{device}")
    else:
        model_device = torch.device("cpu")
    
    # Initialize model - this is the expensive operation we're trying to avoid repeating
    tts_model = SparkTTS(model_dir, model_device)
    logging.info(f"Persistent TTS Worker {worker_id}: Model initialized and ready")
    
    try:
        while True:
            try:
                # Get job from queue, using a timeout to periodically check if we should shut down
                job = job_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if job is None:
                    logging.info(f"Persistent TTS Worker {worker_id}: Received shutdown signal")
                    break
                
                # Process the job
                active_event.set()  # Mark as active
                
                (
                    queue_lock, sentence_queue, sentence_count, processed_count,
                    cuda_stream, base_fragment_num, prompt_speech, prompt_text_clean,
                    tts_to_rvc_queue, tts_complete_events, num_rvc_workers
                ) = job
                
                while True:
                    # Get next sentence from priority queue with thread safety
                    with queue_lock:
                        if sentence_queue.empty():
                            break
                        
                        priority, global_idx, sentence = sentence_queue.get()
                        with processed_count.get_lock():
                            processed_count.value += 1
                            current_count = processed_count.value
                    
                    fragment_num = base_fragment_num + global_idx
                    tts_filename = f"fragment_{fragment_num}.wav"
                    save_path = os.path.join("./TEMP/spark", tts_filename)
                    
                    logging.info(f"Persistent TTS Worker {worker_id}: Processing sentence {global_idx+1}/{sentence_count} (priority {priority}): {sentence[:30]}...")
                    
                    try:
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
                        sf.write(save_path, wav, samplerate=16000)
                        logging.info(f"Persistent TTS Worker {worker_id}: Audio saved at: {save_path}")
                        tts_to_rvc_queue.put((global_idx, fragment_num, sentence, save_path))
                    except Exception as e:
                        logging.error(f"Persistent TTS Worker {worker_id} error for sentence {global_idx}: {str(e)}")
                        tts_to_rvc_queue.put((global_idx, fragment_num, sentence, None, str(e)))
                
                logging.info(f"Persistent TTS Worker {worker_id}: Completed processing sentences")
                tts_complete_events[worker_id].set()
                
                # If all TTS workers are done, add sentinel values for each RVC worker.
                if all(event.is_set() for event in tts_complete_events):
                    for _ in range(num_rvc_workers):
                        tts_to_rvc_queue.put(None)
                
                # Mark as idle after job completes
                model_manager.mark_tts_worker_idle(worker_id)
                
            except Empty:
                # No job, continue waiting
                continue
            except Exception as e:
                logging.error(f"Persistent TTS Worker {worker_id} unexpected error: {str(e)}")
                # Mark as idle after error
                model_manager.mark_tts_worker_idle(worker_id)
    
    finally:
        # Clean up resources
        logging.info(f"Persistent TTS Worker {worker_id}: Shutting down and releasing resources")
        # Explicit cleanup for the model if needed
        del tts_model
        torch.cuda.empty_cache()


def persistent_rvc_worker(worker_id, cuda_stream, job_queue, active_event, model_manager):
    """
    Persistent RVC worker that stays alive and processes jobs from the queue.
    
    Args:
        worker_id (int): Unique worker ID
        cuda_stream: CUDA stream to use
        job_queue (Queue): Queue for receiving jobs
        active_event (Event): Event to signal when worker is active/idle
        model_manager (ModelManager): Reference to the model manager
    """
    logging.info(f"Persistent RVC Worker {worker_id}: Starting and waiting for jobs")
    
    try:
        while True:
            try:
                # Get job from queue, using a timeout to periodically check if we should shut down
                job = job_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if job is None:
                    logging.info(f"Persistent RVC Worker {worker_id}: Received shutdown signal")
                    break
                
                # Process the job
                active_event.set()  # Mark as active
                
                (
                    tts_to_rvc_queue, rvc_results_queue, rvc_complete_events, 
                    tts_complete_events, spk_item, vc_transform, f0method,
                    file_index1, file_index2, index_rate, filter_radius,
                    resample_sr, rms_mix_rate, protect, processing_complete
                ) = job
                
                while True:
                    try:
                        item = tts_to_rvc_queue.get(timeout=0.5)
                    except Empty:
                        if all(event.is_set() for event in tts_complete_events):
                            break
                        continue
                    
                    if item is None:
                        break
                    
                    # Check if the item indicates an error from a TTS worker (length==5)
                    if len(item) == 5:
                        i, fragment_num, sentence, _, error = item
                        rvc_results_queue.put((i, None, None, False, f"TTS error for sentence {i+1}: {error}"))
                        continue
                    
                    i, fragment_num, sentence, tts_path = item
                    if not tts_path or not os.path.exists(tts_path):
                        rvc_results_queue.put((i, None, None, False, f"No TTS output for sentence {i+1}"))
                        continue
                    
                    # Determine output file path
                    rvc_path = os.path.join("./TEMP/rvc", f"fragment_{fragment_num}.wav")
                    
                    try:
                        logging.info(f"Persistent RVC Worker {worker_id}: Processing fragment {fragment_num}")
                        
                        # Merged process_with_rvc logic
                        with (torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext()):
                            # f0_file is not used here
                            f0_file = None
                            output_info, output_audio = vc.vc_single(
                                spk_item, tts_path, vc_transform, f0_file, f0method,
                                file_index1, file_index2, index_rate, filter_radius,
                                resample_sr, rms_mix_rate, protect
                            )
                        
                        # Save RVC output (CPU operation)
                        rvc_saved = False
                        try:
                            if isinstance(output_audio, str) and os.path.exists(output_audio):
                                # Case 1: output_audio is a file path string
                                shutil.copy2(output_audio, rvc_path)
                                rvc_saved = True
                            elif isinstance(output_audio, tuple) and len(output_audio) >= 2:
                                # Case 2: output_audio is a (sample_rate, audio_data) tuple
                                sf.write(rvc_path, output_audio[1], output_audio[0])
                                rvc_saved = True
                            elif hasattr(output_audio, 'name') and os.path.exists(output_audio.name):
                                # Case 3: output_audio is a file-like object
                                shutil.copy2(output_audio.name, rvc_path)
                                rvc_saved = True
                        except Exception as e:
                            output_info += f"\nError saving RVC output: {str(e)}"
                        
                        logging.info(f"RVC inference completed for {tts_path}")
                        
                        info_message = f"Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
                        info_message += f"  - Spark output: {tts_path}\n"
                        if rvc_saved:
                            info_message += f"  - RVC output (Worker {worker_id}): {rvc_path}"
                        else:
                            info_message += f"  - Could not save RVC output to {rvc_path}"
                        
                        rvc_results_queue.put((i, tts_path, rvc_path if rvc_saved else None, rvc_saved, info_message))
                    except Exception as e:
                        logging.error(f"Persistent RVC Worker {worker_id} error for sentence {i}: {str(e)}")
                        info_message = f"Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
                        info_message += f"  - Spark output: {tts_path}\n"
                        info_message += f"  - RVC processing error (Worker {worker_id}): {str(e)}"
                        rvc_results_queue.put((i, tts_path, None, False, info_message))
                
                logging.info(f"Persistent RVC Worker {worker_id}: Completed current job")
                rvc_complete_events[worker_id].set()
                
                if all(event.is_set() for event in rvc_complete_events):
                    processing_complete.set()
                
                # Mark as idle after job completes
                model_manager.mark_rvc_worker_idle(worker_id)
                
            except Empty:
                # No job, continue waiting
                continue
            except Exception as e:
                logging.error(f"Persistent RVC Worker {worker_id} unexpected error: {str(e)}")
                # Mark as idle after error
                model_manager.mark_rvc_worker_idle(worker_id)
    
    finally:
        # Clean up resources
        logging.info(f"Persistent RVC Worker {worker_id}: Shutting down and releasing resources")
        # RVC might need explicit cleanup
        torch.cuda.empty_cache()
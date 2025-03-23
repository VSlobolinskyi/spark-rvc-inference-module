import multiprocessing
import time
import logging
import os
import threading
from merged_ui.buffer_queue import OrderedAudioBufferQueue
from merged_ui.utils import create_queues_and_events, create_sentence_priority_queue, get_base_fragment_num, initialize_cuda_streams, initialize_temp_dirs, prepare_audio_buffer, prepare_prompt, split_text_and_validate
from merged_ui.worker_manager import get_worker_manager


def process_results(sentences, rvc_results_queue: multiprocessing.Queue, buffer: OrderedAudioBufferQueue, processing_complete):
    """
    Process results from the RVC queue and feed them to the ordered buffer queue.
    Files will be added to the buffer as they complete (in any order) but will
    be output in the correct sequence.
    """
    # This function remains the same as the original
    completed_sentences = {}
    info_messages = []
    
    # Continue while processing is active or results are still pending
    while (not processing_complete.is_set() or 
           not rvc_results_queue.empty() or 
           not buffer.is_empty() or 
           (completed_sentences and len(completed_sentences) < len(sentences))):
        try:
            # Gather all available results
            while not rvc_results_queue.empty():
                i, tts_path, rvc_path, success, info = rvc_results_queue.get_nowait()
                completed_sentences[i] = True
                info_messages.append(info)
                
                if success and rvc_path and os.path.exists(rvc_path):
                    # Add directly to the ordered buffer with its position
                    buffer.add_with_position(rvc_path, i)
                    logging.info(f"Sentence {i+1} added to ordered buffer")
                else:
                    logging.warning(f"Sentence {i+1} had no valid output file")
                
            # Get next file from the buffer (this respects timing and order)
            next_file = buffer.get_next()
            if next_file:
                yield "\n".join(info_messages), next_file
            else:
                time.sleep(0.1)
                
        except Exception as e:
            logging.error(f"Error in main processing loop: {str(e)}")
            info_messages.append(f"Error in processing: {str(e)}")
            yield "\n".join(info_messages), None
            break


def generate_and_process_with_rvc_parallel(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect,
    num_tts_workers=2, num_rvc_workers=1,
    model_dir="spark/pretrained_models/Spark-TTS-0.5B", device="0",
    **kwargs
):
    """
    Modified orchestration function that supports both original and persistent worker modes.
    Compatible with existing UI integration.
    
    The model_unload_delay can be specified in three ways (in order of precedence):
    1. As a keyword argument: model_unload_delay=60
    2. Through the MODEL_UNLOAD_DELAY environment variable
    3. Default value (30 seconds)
    """
    # Get model_unload_delay from kwargs if provided
    model_unload_delay = kwargs.get('model_unload_delay', None)
    
    # Step 1: Initialize environment and inputs (same as original)
    initialize_temp_dirs()
    buffer = prepare_audio_buffer()
    sentences = split_text_and_validate(text)
    base_fragment_num = get_base_fragment_num(sentences)
    prompt_speech, prompt_text_clean = prepare_prompt(prompt_wav_upload, prompt_wav_record, prompt_text)
    
    # Step 2: Set up CUDA streams (same as original)
    tts_streams, rvc_streams = initialize_cuda_streams(num_tts_workers, num_rvc_workers)
    
    # Step 3: Set up inter-thread communication (same as original)
    (tts_to_rvc_queue, rvc_results_queue, 
     tts_complete_events, rvc_complete_events, 
     processing_complete) = create_queues_and_events(num_tts_workers, num_rvc_workers)
    
    # Create shared sentence priority queue (same as original)
    sentence_queue, sentence_count = create_sentence_priority_queue(sentences)
    queue_lock = threading.Lock()
    processed_count = multiprocessing.Value('i', 0)  # Shared counter for processed sentences
    
    info_messages = [f"Processing {len(sentences)} sentences using {num_tts_workers} TTS workers and {num_rvc_workers} RVC workers..."]
    yield "\n".join(info_messages), None  # initial status message
    
    model_manager = get_worker_manager(model_unload_delay)

    # Start or reuse TTS workers through the model manager
    for i in range(num_tts_workers):
        tts_job_queue = model_manager.get_tts_worker(i, model_dir, device)
        
        # Create the job payload
        tts_job = (
            queue_lock, sentence_queue, sentence_count, processed_count,
            tts_streams[i], base_fragment_num, prompt_speech, prompt_text_clean,
            tts_to_rvc_queue, tts_complete_events, num_rvc_workers
        )
        
        # Submit the job to the worker
        tts_job_queue.put(tts_job)

    # Start or reuse RVC workers through the model manager
    for i in range(num_rvc_workers):
        rvc_job_queue = model_manager.get_rvc_worker(i, rvc_streams[i])
        
        # Create the job payload
        rvc_job = (
            tts_to_rvc_queue, rvc_results_queue, rvc_complete_events, 
            tts_complete_events, spk_item, vc_transform, f0method,
            file_index1, file_index2, index_rate, filter_radius,
            resample_sr, rms_mix_rate, protect, processing_complete
        )
        
        # Submit the job to the worker
        rvc_job_queue.put(rvc_job)

    # Step 5: Process results and yield output as they become available (same as original)
    for output in process_results(sentences, rvc_results_queue, buffer, processing_complete):
        yield output
        
    logging.info(f"Processing complete. Processed {len(sentences)} sentences.")
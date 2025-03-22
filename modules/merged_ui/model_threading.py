import time
import logging
import os
import platform
import soundfile as sf
import torch
import threading
from queue import Queue, Empty
from contextlib import nullcontext
from merged_ui.utils import process_with_rvc, split_into_sentences
from spark.cli.SparkTTS import SparkTTS

model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0

class AudioBufferQueue:
    """
    A buffer queue for audio file outputs that paces the release of files based on their duration.
    This ensures each audio file has time to finish playing before the next one is released.
    """
    def __init__(self):
        self.queue = []  # Queue to store (file_path, duration) tuples
        self.current_file = None  # Currently playing file path
        self.current_duration = 0  # Duration of currently playing file in seconds
        self.playback_start_time = None  # When the current file started playing
        
    def add(self, file_path):
        """
        Add a file to the buffer queue.
        
        Args:
            file_path (str): Path to the audio file
        """
        if file_path and os.path.exists(file_path):
            try:
                # Get audio duration using soundfile
                with sf.SoundFile(file_path) as sound_file:
                    duration = len(sound_file) / sound_file.samplerate
                
                # Add file to queue with its duration
                self.queue.append((file_path, duration))
                logging.info(f"Added file to buffer queue: {file_path} (duration: {duration:.2f}s)")
            except Exception as e:
                logging.error(f"Error getting audio duration: {str(e)}")
                # If we can't get duration, use a default value
                self.queue.append((file_path, 1.0))
        else:
            # If file doesn't exist, add it with zero duration to pass through immediately
            self.queue.append((file_path, 0))
    
    def get_next(self):
        """
        Get the next file from the queue if enough time has passed for the current file.
        
        Returns:
            str or None: The file path if ready, None otherwise
        """
        current_time = time.time()
        
        # If we're currently playing a file, check if it's finished based on real elapsed time
        if self.current_file is not None and self.playback_start_time is not None:
            elapsed_time = current_time - self.playback_start_time
            time_remaining = self.current_duration - elapsed_time
            
            # Debugging information
            logging.debug(f"Current file: {self.current_file}, Elapsed: {elapsed_time:.2f}s, " +
                         f"Duration: {self.current_duration:.2f}s, Remaining: {time_remaining:.2f}s")
            
            # If not enough time has passed (with a small buffer for processing delays)
            if elapsed_time < (self.current_duration - 0.1):
                return None
            
            # Log that the file has finished playing
            logging.info(f"Finished playing {self.current_file} (duration: {self.current_duration:.2f}s)")
            self.current_file = None
        
        # If we don't have a current file and there are files in the queue, get the next one
        if self.current_file is None and self.queue:
            file_path, duration = self.queue.pop(0)
            self.current_file = file_path
            self.current_duration = duration
            self.playback_start_time = current_time
            
            logging.info(f"Started playing {file_path} (duration: {duration:.2f}s)")
            return file_path
        
        return None
    
    def is_empty(self):
        """
        Check if the queue is empty and there's no current file.
        
        Returns:
            bool: True if empty, False otherwise
        """
        return len(self.queue) == 0 and self.current_file is None

def generate_and_process_with_rvc_parallel(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect,
    num_tts_workers=2
):
    """
    Handle combined TTS and RVC processing using multiple TTS instances in parallel.
    Uses a producer-consumer pattern where multiple TTS workers produce audio files for RVC to consume.
    Includes buffer queue for pacing output based on audio durations.
    """
    # Ensure TEMP directories exist
    os.makedirs("./TEMP/spark", exist_ok=True)
    os.makedirs("./TEMP/rvc", exist_ok=True)
    
    # Create a buffer queue for outputting files
    buffer = AudioBufferQueue()
    
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
    next_to_buffer = 0
    last_yielded_idx = -1
    
    # Loop continues as long as processing is happening, buffer has items, or there are results to process
    while (not processing_complete.is_set() or not rvc_results_queue.empty() or not buffer.is_empty() 
           or (completed_sentences and next_to_buffer <= max(completed_sentences.keys()))):
        try:
            # Try to get an item from the results queue with a timeout
            try:
                i, tts_path, rvc_path, success, info = rvc_results_queue.get(timeout=0.1)
                completed_sentences[i] = (tts_path, rvc_path, success, info)
            except Empty:
                # No results available yet
                pass
                
            # Add completed items to the buffer in order
            while next_to_buffer in completed_sentences:
                _, rvc_path, _, info = completed_sentences[next_to_buffer]
                
                # Add to buffer if file exists
                if rvc_path and os.path.exists(rvc_path):
                    buffer.add(rvc_path)
                
                # Add info message
                info_messages.append(info)
                
                # Move to the next sentence
                next_to_buffer += 1
            
            # Try to get a file from the buffer if it's ready
            next_file = buffer.get_next()
            if next_file:
                # Yield the current state with the next file
                yield "\n".join(info_messages), next_file
                last_yielded_idx = list(completed_sentences.keys()).index(next_to_buffer - 1)
            else:
                # No file ready yet, sleep briefly
                time.sleep(0.1)
                
        except Exception as e:
            logging.error(f"Error in main processing loop: {str(e)}")
            info_messages.append(f"Error in processing: {str(e)}")
            yield "\n".join(info_messages), None
            break
    
    # Join the threads
    for thread in tts_threads:
        thread.join()
    rvc_thread.join()
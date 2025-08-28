import os
import re
import shutil
import threading
import logging
from queue import PriorityQueue, Queue
import torch

from merged_ui.buffer_queue import OrderedAudioBufferQueue
from rvc_ui.initialization import vc

# Initialize the Spark TTS model (moved outside function to avoid reinitializing)
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0

def initialize_temp_dirs():
    temp_dirs = ["./TEMP/spark", "./TEMP/rvc"]
    for dir_path in temp_dirs:
        os.makedirs(dir_path, exist_ok=True)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    logging.info(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logging.info(f"Removed directory: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")

def prepare_audio_buffer(buffer_time=1.5):
    """
    Create and return an OrderedAudioBufferQueue for managing audio output order.
    """
    return OrderedAudioBufferQueue(buffer_time)

def split_text_and_validate(text):
    sentences = split_into_sentences(text)
    if not sentences:
        raise ValueError("No valid text to process.")
    return sentences

def get_base_fragment_num(sentences):
    base_fragment_num = 1
    while any(
        os.path.exists(f"./TEMP/spark/fragment_{base_fragment_num + i}.wav") or 
        os.path.exists(f"./TEMP/rvc/fragment_{base_fragment_num + i}.wav")
        for i in range(len(sentences))
    ):
        base_fragment_num += 1
    return base_fragment_num

def prepare_prompt(prompt_wav_upload, prompt_wav_record, prompt_text):
    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
    return prompt_speech, prompt_text_clean

def initialize_cuda_streams(num_tts_workers, num_rvc_workers):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        tts_streams = [torch.cuda.Stream() for _ in range(num_tts_workers)]
        rvc_streams = [torch.cuda.Stream() for _ in range(num_rvc_workers)]
        logging.info(f"Using {num_tts_workers} CUDA streams for Spark TTS and {num_rvc_workers} for RVC")
    else:
        tts_streams = [None] * num_tts_workers
        rvc_streams = [None] * num_rvc_workers
        logging.info("CUDA not available, parallel processing will be limited")
    return tts_streams, rvc_streams

def create_queues_and_events(num_tts_workers, num_rvc_workers):
    tts_to_rvc_queue = Queue()
    rvc_results_queue = Queue()
    tts_complete_events = [threading.Event() for _ in range(num_tts_workers)]
    rvc_complete_events = [threading.Event() for _ in range(num_rvc_workers)]
    processing_complete = threading.Event()
    return tts_to_rvc_queue, rvc_results_queue, tts_complete_events, rvc_complete_events, processing_complete

def create_sentence_priority_queue(sentences):
    """
    Creates a priority queue of sentences, prioritized by their original order.
    
    Args:
        sentences: List of sentences to process
        
    Returns:
        A priority queue containing tuples of (priority, index, sentence)
    """
    sentence_queue = PriorityQueue()
    for idx, sentence in enumerate(sentences):
        # Use index as priority to maintain original order
        sentence_queue.put((idx, idx, sentence))
    
    return sentence_queue, len(sentences)

def split_into_sentences(text):
    """
    Split text into sentences using regular expressions.
    
    Args:
        text (str): The input text to split
        
    Returns:
        list: A list of sentences
    """
    sentences = re.split(r'(?<=[.!?,:;—–-])\s+|(?<=[.!?,:;—–-])$', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Combine sentences with < 3 words with the next sentence
    combined = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        
        # If current sentence has < 3 words, combine with next (if exists)
        while len(current.split()) < 3 and i + 1 < len(sentences):
            i += 1
            current += " " + sentences[i]
        
        combined.append(current)
        i += 1

    sentences = combined
    return sentences


def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""
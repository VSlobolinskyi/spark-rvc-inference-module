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

def prepare_audio_buffer(buffer_time=1.0):
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

def split_into_sentences(text, max_chunk_size=40):
    """
    Split text into balanced chunks for TTS processing.
    
    The function first splits the text into sentences, then tries to create chunks
    of approximately equal size without breaking words. Long sentences are split at
    natural phrase boundaries (commas, semicolons, colons) when possible.
    
    Args:
        text (str): The input text to split
        max_chunk_size (int): Target maximum size of each chunk in characters
        
    Returns:
        list: A list of text chunks balanced for TTS processing
    """
    def split_long_text(text, max_size):
        """Split a long text into chunks at natural phrase boundaries."""
        result = []
        
        # Try to split on natural phrase boundaries
        phrases = re.split(r'(?<=[,;:])\s+', text)
        current_chunk = ""
        
        for phrase in phrases:
            # If this phrase fits in the current chunk
            if len(current_chunk) + len(phrase) + (1 if current_chunk else 0) <= max_size:
                if current_chunk:
                    current_chunk += " " + phrase
                else:
                    current_chunk = phrase
            else:
                # Add the current chunk if it exists
                if current_chunk:
                    result.append(current_chunk)
                
                # If the phrase itself is too long, split by words
                if len(phrase) > max_size:
                    words = phrase.split()
                    word_chunk = ""
                    
                    for word in words:
                        if len(word_chunk) + len(word) + (1 if word_chunk else 0) <= max_size:
                            if word_chunk:
                                word_chunk += " " + word
                            else:
                                word_chunk = word
                        else:
                            result.append(word_chunk)
                            word_chunk = word
                    
                    current_chunk = word_chunk
                else:
                    current_chunk = phrase
        
        # Add the final chunk if it exists
        if current_chunk:
            result.append(current_chunk)
        
        return result
    
    # First split into sentences
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If the sentence fits in the current chunk, add it
        if len(current_chunk) + len(sentence) + (1 if current_chunk else 0) <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # This sentence won't fit in the current chunk
            
            # Add the current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # If the sentence itself is shorter than max_chunk_size, use it as the new current chunk
            if len(sentence) <= max_chunk_size:
                current_chunk = sentence
            else:
                # The sentence is too long, we need to split it
                sentence_chunks = split_long_text(sentence, max_chunk_size)
                
                # Add all but the last chunk
                if sentence_chunks:
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = ""
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""
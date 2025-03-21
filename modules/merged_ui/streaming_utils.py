import os
import time

# Import and reuse existing functions
from merged_ui.utils import (
  split_into_sentences, 
  process_single_sentence,
  concatenate_audio_files,
  split_into_sentences,
  process_single_sentence,
  initialize_model
)

# Initialize the Spark TTS model (moved outside function to avoid reinitializing)
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0
spark_model = initialize_model(model_dir, device=device)

def generate_and_process_with_rvc_streaming(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect
):
    """
    Stream process TTS and RVC, yielding audio updates as sentences are processed
    This is a generator function that yields (status_text, audio_path) tuples
    """
    # Ensure TEMP directories exist
    os.makedirs("./TEMP/spark", exist_ok=True)
    os.makedirs("./TEMP/rvc", exist_ok=True)
    os.makedirs("./TEMP/stream", exist_ok=True)
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    if not sentences:
        yield "No valid text to process.", None
        return
    
    # Get timestamp to create unique session ID for this run
    session_id = str(int(time.time()))
    
    # Process reference speech
    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
    
    # Initialize status
    status_messages = [f"Starting to process {len(sentences)} sentences..."]
    # Create a list to track processed fragments
    processed_fragments = []
    
    # Create a temporary directory for streaming
    stream_dir = f"./TEMP/stream/{session_id}"
    os.makedirs(stream_dir, exist_ok=True)
    
    # Yield initial status
    yield "\n".join(status_messages), None
    
    # Process each sentence and update the stream
    for i, sentence in enumerate(sentences):
        # Add current sentence to status
        current_msg = f"Processing sentence {i+1}/{len(sentences)}: {sentence[:30]}..."
        status_messages.append(current_msg)
        yield "\n".join(status_messages), None
        
        # Process this sentence
        spark_path, rvc_path, success, info = process_single_sentence(
            i, sentence, prompt_speech, prompt_text_clean,
            spk_item, vc_transform, f0method, 
            file_index1, file_index2, index_rate, filter_radius,
            resample_sr, rms_mix_rate, protect,
            int(session_id)  # Use session ID as base fragment number
        )
        
        # Update status with processing result
        status_messages[-1] = info
        
        if success and rvc_path and os.path.exists(rvc_path):
            processed_fragments.append(rvc_path)
            
            # Create a streaming update file by concatenating all fragments processed so far
            stream_path = os.path.join(stream_dir, f"stream_update_{i+1}.wav")
            
            # Concatenate all fragments processed so far
            concatenate_success = concatenate_audio_files(processed_fragments, stream_path)
            
            if concatenate_success:
                # Yield the updated status and the current stream path
                yield "\n".join(status_messages), stream_path
            else:
                # If concatenation failed, just yield the most recent fragment
                yield "\n".join(status_messages), rvc_path
        else:
            # If processing failed, update status but don't update audio
            yield "\n".join(status_messages), None if not processed_fragments else processed_fragments[-1]
    
    # Final streaming update with completion message
    if processed_fragments:
        # Create final output file
        final_output_path = f"./TEMP/stream/{session_id}/final_output.wav"
        concatenate_success = concatenate_audio_files(processed_fragments, final_output_path)
        
        if concatenate_success:
            status_messages.append(f"\nAll {len(sentences)} sentences processed successfully!")
            yield "\n".join(status_messages), final_output_path
        else:
            status_messages.append("\nWarning: Failed to create final concatenated file.")
            yield "\n".join(status_messages), processed_fragments[-1]
    else:
        status_messages.append("\nNo sentences were successfully processed.")
        yield "\n".join(status_messages), None
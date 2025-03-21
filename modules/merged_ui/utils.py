import os
import shutil
import re
import numpy as np
from time import sleep
import soundfile as sf
from pydub import AudioSegment

# Import modules from your packages
from rvc_ui.initialization import vc
from spark_ui.main import initialize_model, run_tts
from spark.sparktts.utils.token_parser import LEVELS_MAP_UI

# Initialize the Spark TTS model (moved outside function to avoid reinitializing)
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0
spark_model = initialize_model(model_dir, device=device)

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
    Process a single sentence through the TTS and RVC pipeline
    
    Args:
        sentence_index (int): Index of the sentence in the original text
        sentence (str): The sentence text to process
        ... (other parameters are the same as generate_and_process_with_rvc)
        
    Returns:
        tuple: (spark_output_path, rvc_output_path, success, info_message)
    """
    fragment_num = base_fragment_num + sentence_index
    
    # Generate TTS audio for this sentence
    tts_path = run_tts(
        sentence,
        spark_model,
        prompt_text=prompt_text_clean,
        prompt_speech=prompt_speech
    )
    
    # Make sure we have a TTS file to process
    if not tts_path or not os.path.exists(tts_path):
        return None, None, False, f"Failed to generate TTS audio for sentence: {sentence}"
    
    # Save Spark output to TEMP/spark
    spark_output_path = f"./TEMP/spark/fragment_{fragment_num}.wav"
    shutil.copy2(tts_path, spark_output_path)
    
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
    
    # Try different ways to save the RVC output based on common formats
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
    Handle combined TTS and RVC processing for multiple sentences and save outputs to TEMP directories
    """
    # Ensure TEMP directories exist
    os.makedirs("./TEMP/spark", exist_ok=True)
    os.makedirs("./TEMP/rvc", exist_ok=True)
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    if not sentences:
        return "No valid text to process.", None
    
    # Get next base fragment number
    base_fragment_num = 1
    while any(os.path.exists(f"./TEMP/spark/fragment_{base_fragment_num + i}.wav") or 
              os.path.exists(f"./TEMP/rvc/fragment_{base_fragment_num + i}.wav") 
              for i in range(len(sentences))):
        base_fragment_num += 1
    
    # Process reference speech
    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
    
    # Process each sentence
    results = []
    info_messages = [f"Processing {len(sentences)} sentences..."]
    
    for i, sentence in enumerate(sentences):
        spark_path, rvc_path, success, info = process_single_sentence(
            i, sentence, prompt_speech, prompt_text_clean,
            spk_item, vc_transform, f0method, 
            file_index1, file_index2, index_rate, filter_radius,
            resample_sr, rms_mix_rate, protect,
            base_fragment_num
        )
        
        info_messages.append(info)
        if success and rvc_path:
            results.append(rvc_path)
    
    # If no sentences were successfully processed
    if not results:
        return "\n".join(info_messages) + "\n\nNo sentences were successfully processed.", None
    
    # Concatenate all successful RVC fragments
    final_output_path = f"./TEMP/final_output_{base_fragment_num}.wav"
    concatenation_success = concatenate_audio_files(results, final_output_path)
    
    if concatenation_success:
        info_messages.append(f"\nAll fragments concatenated successfully to: {final_output_path}")
        return "\n".join(info_messages), final_output_path
    else:
        # If concatenation failed but we have at least one successful fragment, return the first one
        info_messages.append(f"\nFailed to concatenate fragments. Returning first successful fragment.")
        return "\n".join(info_messages), results[0]

def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""

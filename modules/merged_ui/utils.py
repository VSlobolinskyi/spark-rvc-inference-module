import os
import shutil

# Import modules from your packages
from rvc_ui.initialization import vc
from spark_ui.main import initialize_model, run_tts
from spark.sparktts.utils.token_parser import LEVELS_MAP_UI

# Initialize the Spark TTS model (moved outside function to avoid reinitializing)
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0
spark_model = initialize_model(model_dir, device=device)

def generate_and_process_with_rvc(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect
):
    """
    Handle combined TTS and RVC processing and save outputs to TEMP directories
    """
    # Ensure TEMP directories exist
    os.makedirs("./TEMP/spark", exist_ok=True)
    os.makedirs("./TEMP/rvc", exist_ok=True)
    
    # Get next fragment number
    fragment_num = 1
    while (os.path.exists(f"./TEMP/spark/fragment_{fragment_num}.wav") or 
           os.path.exists(f"./TEMP/rvc/fragment_{fragment_num}.wav")):
        fragment_num += 1
    
    # First generate TTS audio
    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
    
    tts_path = run_tts(
        text,
        spark_model,
        prompt_text=prompt_text_clean,
        prompt_speech=prompt_speech
    )
    
    # Make sure we have a TTS file to process
    if not tts_path or not os.path.exists(tts_path):
        return "Failed to generate TTS audio", None
    
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
                import soundfile as sf
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
    
    # Add file paths to output info
    output_info += f"\nSpark output saved to: {spark_output_path}"
    if rvc_saved:
        output_info += f"\nRVC output saved to: {rvc_output_path}"
    else:
        output_info += f"\nCould not automatically save RVC output to {rvc_output_path}"
    
    return output_info, output_audio

def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""
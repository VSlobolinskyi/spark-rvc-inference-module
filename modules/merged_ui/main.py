import os
import gradio as gr
import torch
from time import sleep

# Import modules from your packages
from rvc_ui.initialization import now_dir, config, vc
from rvc_ui.main import build_rvc_ui, names, index_paths
from spark_ui.main import build_spark_ui, initialize_model, run_tts
from spark.sparktts.utils.token_parser import LEVELS_MAP_UI

def build_merged_ui():
     # Initialize the Spark TTS model
    model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
    device = 0
    spark_model = initialize_model(model_dir, device=device)
    
    # Create the UI
    with gr.Blocks(title="Unified TTS-RVC Pipeline") as merged_ui:
        gr.Markdown("## Voice Generation and Conversion Pipeline")
        gr.Markdown("Generate speech with Spark TTS and process it through RVC for voice conversion")
        
        with gr.Tabs():
            with gr.TabItem("TTS-to-RVC Pipeline"):
                gr.Markdown("### Step 1: Generate speech with Spark TTS")
                
                # TTS Generation Section
                with gr.Tabs():
                    # Voice Clone option
                    with gr.TabItem("Voice Clone"):
                        with gr.Row():
                            prompt_wav_upload = gr.Audio(
                                sources="upload",
                                type="filepath",
                                label="Reference voice (upload)",
                            )
                            prompt_wav_record = gr.Audio(
                                sources="microphone",
                                type="filepath",
                                label="Reference voice (record)",
                            )

                        with gr.Row():
                            tts_text_input = gr.Textbox(
                                label="Text to synthesize", 
                                lines=3, 
                                placeholder="Enter text for TTS"
                            )
                            prompt_text_input = gr.Textbox(
                                label="Text of prompt speech (Optional)",
                                lines=3,
                                placeholder="Enter text of the reference audio",
                            )
                    
                    # Voice Creation option
                    with gr.TabItem("Voice Creation"):
                        with gr.Row():
                            with gr.Column():
                                gender = gr.Radio(
                                    choices=["male", "female"], value="male", label="Gender"
                                )
                                pitch = gr.Slider(
                                    minimum=1, maximum=5, step=1, value=3, label="Pitch"
                                )
                                speed = gr.Slider(
                                    minimum=1, maximum=5, step=1, value=3, label="Speed"
                                )
                            with gr.Column():
                                tts_text_input_creation = gr.Textbox(
                                    label="Text to synthesize",
                                    lines=3,
                                    placeholder="Enter text for TTS",
                                    value="Generate speech with this text and then convert the voice with RVC.",
                                )
                
                tts_audio_output = gr.Audio(label="Generated TTS Audio")
                
                with gr.Row():
                    generate_clone_button = gr.Button("Generate with Voice Clone", variant="primary")
                    generate_create_button = gr.Button("Generate with Voice Creation", variant="primary")
                
                # Hidden text field to store the TTS audio path
                tts_audio_path = gr.Textbox(visible=False)
                
                gr.Markdown("### Step 2: Convert with RVC")
                
                # RVC Settings
                with gr.Row():
                    with gr.Column():
                        sid0 = gr.Dropdown(
                            label="Target voice model:", choices=sorted(names)
                        )
                        vc_transform0 = gr.Number(
                            label="Transpose (semitones):",
                            value=0,
                        )
                        f0method0 = gr.Radio(
                            label="Pitch extraction algorithm:",
                            choices=(
                                ["pm", "harvest", "crepe", "rmvpe"]
                                if config.dml == False
                                else ["pm", "harvest", "rmvpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        file_index1 = gr.Textbox(
                            label="Path to feature index file (leave blank for auto):",
                            placeholder="Leave blank to use dropdown selection",
                            interactive=True,
                        )
                        file_index2 = gr.Dropdown(
                            label="Select feature index:",
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        
                    with gr.Column():
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Feature search ratio (accent strength):",
                            value=0.75,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Median filter radius (3+ reduces breathiness):",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume envelope scaling (0=original, 1=constant):",
                            value=0.25,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Consonant protection (0=max, 0.5=disable):",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Output sample rate (0=no resampling):",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                
                # Speaker ID (hidden)
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Speaker ID:",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                
                # Process button and outputs
                process_button = gr.Button("Process with RVC", variant="primary")
                
                with gr.Row():
                    vc_output1 = gr.Textbox(label="Output information")
                    vc_output2 = gr.Audio(label="Final converted audio")
                
                # Function to handle voice clone TTS generation
                def voice_clone_tts(text, prompt_text, prompt_wav_upload, prompt_wav_record):
                    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
                    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
                    
                    audio_output_path = run_tts(
                        text,
                        spark_model,
                        prompt_text=prompt_text_clean,
                        prompt_speech=prompt_speech
                    )
                    
                    return audio_output_path, audio_output_path
                
                # Function to handle voice creation TTS generation
                def voice_creation_tts(text, gender, pitch, speed):
                    pitch_val = LEVELS_MAP_UI[int(pitch)]
                    speed_val = LEVELS_MAP_UI[int(speed)]
                    
                    audio_output_path = run_tts(
                        text,
                        spark_model,
                        gender=gender,
                        pitch=pitch_val,
                        speed=speed_val
                    )
                    
                    return audio_output_path, audio_output_path
                
                # Function to process audio with RVC
                def process_with_rvc(
                    tts_path, spk_item, vc_transform, f0method, 
                    file_index1, file_index2, index_rate, filter_radius,
                    resample_sr, rms_mix_rate, protect
                ):
                    # Make sure we have a TTS file to process
                    if not tts_path or not os.path.exists(tts_path):
                        return "No TTS audio generated yet", None
                    
                    # Call RVC processing function
                    f0_file = None  # We're not using an F0 curve file in this pipeline
                    output_info, output_audio = vc.vc_single(
                        spk_item, tts_path, vc_transform, f0_file, f0method,
                        file_index1, file_index2, index_rate, filter_radius,
                        resample_sr, rms_mix_rate, protect
                    )
                    
                    return output_info, output_audio
                
                # Connect functions to buttons
                generate_clone_button.click(
                    voice_clone_tts,
                    inputs=[
                        tts_text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                    ],
                    outputs=[tts_audio_output, tts_audio_path],
                )
                
                generate_create_button.click(
                    voice_creation_tts,
                    inputs=[
                        tts_text_input_creation,
                        gender,
                        pitch,
                        speed,
                    ],
                    outputs=[tts_audio_output, tts_audio_path],
                )
                
                process_button.click(
                    process_with_rvc,
                    inputs=[
                        tts_audio_path,
                        spk_item,
                        vc_transform0,
                        f0method0,
                        file_index1,
                        file_index2,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    outputs=[vc_output1, vc_output2],
                )
                
                def modified_get_vc(sid0_value, protect0_value):
                    protect1_value = protect0_value
                    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
                    
                    if isinstance(outputs, tuple) and len(outputs) >= 3:
                        return outputs[0], outputs[1], outputs[3]
                    
                    return 0, protect0_value, file_index2.choices[0] if file_index2.choices else ""
                
                sid0.change(
                    fn=modified_get_vc,
                    inputs=[sid0, protect0],
                    outputs=[spk_item, protect0, file_index2],
                )
                
    return merged_ui

if __name__ == "__main__":
    build_merged_ui()
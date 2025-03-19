# rvc_ui/main.py
import os
import gradio as gr
import shutil
from time import sleep

# Import modules from your package
from rvc_ui.initialization import now_dir, config, vc
from rvc_ui.utils import (
    update_audio_path,
    lookup_indices,
    change_choices,
    clean,
    ToolButton,
)

# Setup weight and index paths from environment variables
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

# Prepare model names and index paths
names = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
index_paths = []
lookup_indices(index_root, index_paths)
lookup_indices(outside_index_root, index_paths)
uvr5_names = [
    name.replace(".pth", "")
    for name in os.listdir(weight_uvr5_root)
    if name.endswith(".pth") or "onnx" in name
]

# Define additional dictionaries and UI functions if needed
sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
F0GPUVisible = config.dml == False


# Build Gradio UI
def build_rvc_ui():
    with gr.Blocks(title="RVC WebUI") as rvc_ui:
        gr.Markdown("## RVC WebUI")
        gr.Markdown(
            value="This software is open source under the MIT license. The author does not have any control over the software. Users who use the software and distribute the sounds exported by the software are solely responsible. <br>If you do not agree with this clause, you cannot use or reference any codes and files within the software package. See the root directory <b>Agreement-LICENSE.txt</b> for details."
        )
        with gr.Tabs():
            with gr.TabItem("Model Inference"):
                with gr.Row():
                    sid0 = gr.Dropdown(
                        label="Inferencing voice:", choices=sorted(names)
                    )
                    with gr.Column():
                        refresh_button = gr.Button(
                            "Refresh voice list and index path", variant="primary"
                        )
                        clean_button = gr.Button(
                            "Unload voice to save GPU memory:", variant="primary"
                        )
                    spk_item = gr.Slider(
                        minimum=0,
                        maximum=2333,
                        step=1,
                        label="Select Speaker/Singer ID:",
                        value=0,
                        visible=False,
                        interactive=True,
                    )
                    clean_button.click(
                        fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                    )
                with gr.TabItem("Single Inference"):
                    with gr.Group():
                        with gr.Row():
                            with gr.Column():
                                vc_transform0 = gr.Number(
                                    label="Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):",
                                    value=0,
                                )
                                # Add a file uploader for drag & drop.
                                audio_upload = gr.File(
                                    label="拖拽或选择音频文件",
                                    file_types=[".wav"],
                                    file_count="single",
                                    interactive=True,
                                )
                                # Existing textbox for the audio file path.
                                input_audio0 = gr.Textbox(
                                    label="Enter the path of the audio file to be processed (default is the correct format example):",
                                    placeholder="C:\\Users\\Desktop\\model_example.wav",
                                    interactive=True,
                                )
                                # When a file is uploaded, update the textbox.
                                audio_upload.change(
                                    fn=update_audio_path,
                                    inputs=audio_upload,
                                    outputs=input_audio0,
                                )
                                file_index1 = gr.Textbox(
                                    label="Path to the feature index file. Leave blank to use the selected result from the dropdown:",
                                    placeholder="C:\\Users\\Desktop\\model_example.index",
                                    interactive=True,
                                )
                                file_index2 = gr.Dropdown(
                                    label="Auto-detect index path and select from the dropdown:",
                                    choices=sorted(index_paths),
                                    interactive=True,
                                )
                                f0method0 = gr.Radio(
                                    label="Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement",
                                    choices=(
                                        ["pm", "harvest", "crepe", "rmvpe"]
                                        if config.dml == False
                                        else ["pm", "harvest", "rmvpe"]
                                    ),
                                    value="rmvpe",
                                    interactive=True,
                                )

                            with gr.Column():
                                resample_sr0 = gr.Slider(
                                    minimum=0,
                                    maximum=48000,
                                    label="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:",
                                    value=0,
                                    step=1,
                                    interactive=True,
                                )
                                rms_mix_rate0 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label="Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume:",
                                    value=0.25,
                                    interactive=True,
                                )
                                protect0 = gr.Slider(
                                    minimum=0,
                                    maximum=0.5,
                                    label="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:",
                                    value=0.33,
                                    step=0.01,
                                    interactive=True,
                                )
                                filter_radius0 = gr.Slider(
                                    minimum=0,
                                    maximum=7,
                                    label="If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.",
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )
                                index_rate1 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label="Search feature ratio (controls accent strength, too high has artifacting):",
                                    value=0.75,
                                    interactive=True,
                                )
                                f0_file = gr.File(
                                    label="F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:",
                                    visible=False,
                                )

                                refresh_button.click(
                                    fn=change_choices,
                                    inputs=[],
                                    outputs=[sid0, file_index2],
                                    api_name="infer_refresh",
                                )
                    with gr.Group():
                        with gr.Column():
                            but0 = gr.Button("Convert", variant="primary")
                            with gr.Row():
                                vc_output1 = gr.Textbox(label="Output information")
                                vc_output2 = gr.Audio(
                                    label="Export audio (click on the three dots in the lower right corner to download)"
                                )
                            but0.click(
                                vc.vc_single,
                                [
                                    spk_item,
                                    input_audio0,
                                    vc_transform0,
                                    f0_file,
                                    f0method0,
                                    file_index1,
                                    file_index2,
                                    index_rate1,
                                    filter_radius0,
                                    resample_sr0,
                                    rms_mix_rate0,
                                    protect0,
                                ],
                                [vc_output1, vc_output2],
                                api_name="infer_convert",
                            )
                with gr.TabItem("Batch Inference"):
                    gr.Markdown(
                        value="Batch conversion. Enter the folder containing the audio files to be converted or upload multiple audio files. The converted audio will be output in the specified folder (default: 'opt')."
                    )
                    with gr.Row():
                        with gr.Column():
                            vc_transform1 = gr.Number(
                                label="Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):",
                                value=0,
                            )
                            opt_input = gr.Textbox(
                                label="Specify output folder:", value="opt"
                            )
                            file_index3 = gr.Textbox(
                                label="Path to the feature index file. Leave blank to use the selected result from the dropdown:",
                                value="",
                                interactive=True,
                            )
                            file_index4 = gr.Dropdown(
                                label="Auto-detect index path and select from the dropdown:",
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            f0method1 = gr.Radio(
                                label="Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement",
                                choices=(
                                    ["pm", "harvest", "crepe", "rmvpe"]
                                    if config.dml == False
                                    else ["pm", "harvest", "rmvpe"]
                                ),
                                value="rmvpe",
                                interactive=True,
                            )
                            format1 = gr.Radio(
                                label="Export file format",
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="wav",
                                interactive=True,
                            )

                            refresh_button.click(
                                fn=lambda: change_choices()[1],
                                inputs=[],
                                outputs=file_index4,
                                api_name="infer_refresh_batch",
                            )

                        with gr.Column():
                            resample_sr1 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:",
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume:",
                                value=1,
                                interactive=True,
                            )
                            protect1 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:",
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            filter_radius1 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label="If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.",
                                value=3,
                                step=1,
                                interactive=True,
                            )
                            index_rate2 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Search feature ratio (controls accent strength, too high has artifacting):",
                                value=1,
                                interactive=True,
                            )
                    with gr.Row():
                        dir_input = gr.Textbox(
                            label="Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):",
                            placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                        )
                        inputs = gr.File(
                            file_count="multiple",
                            label="Multiple audio files can also be imported. If a folder path exists, this input is ignored.",
                        )

                    with gr.Row():
                        but1 = gr.Button("Convert", variant="primary")
                        vc_output3 = gr.Textbox(label="Output information")

                        but1.click(
                            vc.vc_multi,
                            [
                                spk_item,
                                dir_input,
                                opt_input,
                                inputs,
                                vc_transform1,
                                f0method1,
                                file_index3,
                                file_index4,
                                index_rate2,
                                filter_radius1,
                                resample_sr1,
                                rms_mix_rate1,
                                protect1,
                                format1,
                            ],
                            [vc_output3],
                            api_name="infer_convert_batch",
                        )
                    sid0.change(
                        fn=vc.get_vc,
                        inputs=[sid0, protect0, protect1],
                        outputs=[
                            spk_item,
                            protect0,
                            protect1,
                            file_index2,
                            file_index4,
                        ],
                        api_name="infer_change_voice",
                    )
        return rvc_ui


if __name__ == "__main__":
    build_rvc_ui()

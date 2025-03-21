#!/usr/bin/env python3
import gradio as gr
import traceback
from merged_ui.main import build_merged_ui
from rvc_ui.initialization import config
from rvc_ui.main import build_rvc_ui
from spark_ui.main import build_spark_ui

def build_standalone_ui():
    with gr.Blocks(title="Unified Inference UI") as app:
        gr.Markdown("## Unified Inference UI: RVC WebUI and Spark TTS")
        with gr.Tabs():
            with gr.TabItem("Spark TTS"):
                build_spark_ui()
            with gr.TabItem("RVC WebUI"):
                build_rvc_ui()
    return app


if __name__ == "__main__":
    app = build_merged_ui()
    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="localhost",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
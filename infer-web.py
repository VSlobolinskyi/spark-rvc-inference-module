#!/usr/bin/env python3
import gradio as gr
import traceback
from rvc_ui.initialization import now_dir, config, vc
from rvc_ui.main import build_rvc_ui
from spark_ui.main import build_spark_ui

def build_unified_ui():
    rvc_ui = build_rvc_ui()  # Returns a gr.Blocks instance for RVC WebUI

    with gr.Blocks(title="Unified Inference UI") as app:
        gr.Markdown("## Unified Inference UI: RVC WebUI and Spark TTS")
        with gr.Tabs():
            with gr.TabItem("RVC WebUI"):
                rvc_ui.render()
            with gr.TabItem("Spark TTS"):
                # Instead of calling render() on the Spark UI object,
                # we'll directly build it in this context
                try:
                    # Create the Spark UI directly in this tab's context
                    build_spark_ui()
                except Exception as e:
                    gr.Markdown(f"Error building Spark TTS: {str(e)}")
                    gr.Markdown(traceback.format_exc())
    return app


if __name__ == "__main__":
    app = build_unified_ui()
    # Needed for RVC
    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="localhost",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
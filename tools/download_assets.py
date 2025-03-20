#!/usr/bin/env python3
import os
import subprocess
import sys
import requests
from pathlib import Path

# Part 1: Download Spark assets
def run_command(command, error_message):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(error_message)
        sys.exit(1)

def clone_spark_tts():
    spark_pretrained_dir = os.path.join("spark", "pretrained_models")
    os.makedirs(spark_pretrained_dir, exist_ok=True)

    print("Running 'git lfs install'...")
    run_command(
        ["git", "lfs", "install"],
        "Error: Failed to run 'git lfs install'. Make sure git-lfs is installed (https://git-lfs.com).",
    )

    clone_dir = os.path.join(spark_pretrained_dir, "Spark-TTS-0.5B")
    if not os.path.exists(clone_dir):
        print(f"Cloning Spark TTS repository into {clone_dir}...")
        run_command(
            [
                "git",
                "clone",
                "https://huggingface.co/SparkAudio/Spark-TTS-0.5B",
                clone_dir,
            ],
            "Error: Failed to clone the Spark TTS repository.",
        )
    else:
        print(f"Directory '{clone_dir}' already exists. Skipping clone.")

# Part 2: Download RVC Assets
def dl_model(link, model_name, dir_name):
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dir_name / model_name), exist_ok=True)
        with open(dir_name / model_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_rvc_models():
    RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
    BASE_DIR = Path(__file__).resolve().parent.parent

    def check_and_dl(link, model_name, dest_dir):
        dest_file = dest_dir / model_name
        if dest_file.exists():
            print(f"{model_name} already exists at {dest_file}. Skipping download.")
        else:
            print(f"Downloading {model_name}...")
            dl_model(link, model_name, dest_dir)

    print("Downloading hubert_base.pt...")
    check_and_dl(RVC_DOWNLOAD_LINK, "hubert_base.pt", BASE_DIR / "assets" / "hubert")

    print("Downloading rmvpe.pt...")
    check_and_dl(RVC_DOWNLOAD_LINK, "rmvpe.pt", BASE_DIR / "assets" / "rmvpe")

    print("Downloading rmvpe.onnx...")
    check_and_dl(RVC_DOWNLOAD_LINK, "rmvpe.onnx", BASE_DIR / "assets" / "rmvpe")

    print("Downloading vocals.onnx...")
    vocals_dir = BASE_DIR / "assets" / "uvr5_weights" / "onnx_dereverb_By_FoxJoy"
    check_and_dl(RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/", "vocals.onnx", vocals_dir)

    print("Downloading ffprobe.exe...")
    check_and_dl(RVC_DOWNLOAD_LINK, "ffprobe.exe", BASE_DIR / ".")

    print("Downloading ffmpeg.exe...")
    check_and_dl(RVC_DOWNLOAD_LINK, "ffmpeg.exe", BASE_DIR / ".")

    rvc_models_dir = BASE_DIR / "assets" / "pretrained"
    print("Downloading pretrained models:")
    model_names = [
        "D32k.pth", "D40k.pth", "D48k.pth",
        "G32k.pth", "G40k.pth", "G48k.pth",
        "f0D32k.pth", "f0D40k.pth", "f0D48k.pth",
        "f0G32k.pth", "f0G40k.pth", "f0G48k.pth",
    ]
    for model in model_names:
        check_and_dl(RVC_DOWNLOAD_LINK + "pretrained/", model, rvc_models_dir)

    rvc_models_dir = BASE_DIR / "assets" / "pretrained_v2"
    print("Downloading pretrained models v2:")
    for model in model_names:
        check_and_dl(RVC_DOWNLOAD_LINK + "pretrained_v2/", model, rvc_models_dir)

    print("Downloading uvr5_weights:")
    rvc_models_dir = BASE_DIR / "assets" / "uvr5_weights"
    model_names = [
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]
    for model in model_names:
        check_and_dl(RVC_DOWNLOAD_LINK + "uvr5_weights/", model, rvc_models_dir)

    print("All models downloaded!")

def main():
    clone_spark_tts()
    download_rvc_models()

if __name__ == "__main__":
    main()

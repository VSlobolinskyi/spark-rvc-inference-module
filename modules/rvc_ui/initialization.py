# rvc_ui/initialization.py
import os
import sys
from dotenv import load_dotenv
import torch
import fairseq
import warnings
import shutil
import logging

# Set current directory and load environment variables
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

# Configure logging and warnings
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Cleanup and create necessary directories
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(f"{now_dir}/runtime/Lib/site-packages/infer_pack", ignore_errors=True)
shutil.rmtree(f"{now_dir}/runtime/Lib/site-packages/uvr5_pack", ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

# Import your configuration and voice conversion modules
from configs.config import Config
from infer.modules.vc.modules import VC

# Instantiate configuration and VC
config = Config()
vc = VC(config)

# Optionally override fairseq grad multiply if dml is enabled
if config.dml:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        return x.clone().detach()

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

# GPU detection and info collection
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "4060",
                "L",
                "6000",
            ]
        ):
            if_gpu_ok = True
            gpu_infos.append(f"{i}\t{gpu_name}")
            mem.append(
                int(torch.cuda.get_device_properties(i).total_memory / (1024**3) + 0.4)
            )

if if_gpu_ok and gpu_infos:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = (
        "Unfortunately, there is no compatible GPU available to support your training."
    )
    default_batch_size = 1

gpus = "-".join([i[0] for i in gpu_infos])

# Expose useful variables for other modules
__all__ = [
    "now_dir",
    "config",
    "vc",
    "gpu_info",
    "default_batch_size",
    "gpus",
    "logger",
]

import os
import subprocess

from huggingface_hub import hf_hub_download


def download_dino():
    url = (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    )
    output_dir = "./weights"
    command = ["wget", url, "-P", output_dir]

    try:
        subprocess.run(command, check=True)
        print("File downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading the file: {e}")


def download_clip():
    repo = "laion/CLIP-ViT-g-14-laion2B-s34B-b88K"
    file = "open_clip_pytorch_model.bin"
    hf_hub_download(repo, file)


if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    download_dino()
    download_clip()

import torch


def get_dino(
    repo_name: str = "facebookresearch/dinov2", model_name: str = "dinov2_vits14_reg"
):
    dino = torch.hub.load(repo_name, model_name)
    return dino

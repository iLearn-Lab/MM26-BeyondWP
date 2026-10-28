import numpy as np
import torch


def to_numpy(x):
    return x.detach().cpu().numpy()


def from_numpy(x):
    return torch.from_numpy(x).float()


ACTION_STATS = {"min": [-1.15, -1.06], "max": [1.13, 1.15]}

ACTION_STATS = {k: np.asarray(v) for k, v in ACTION_STATS.items()}


def get_action(diffusion_output, action_stats=ACTION_STATS):
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def normalize_data(data, stats):
    device = data.device
    data = to_numpy(data)
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    ndata = ndata * 2 - 1
    ndata = from_numpy(ndata).to(device)
    return ndata


def get_delta(actions):
    device = actions.device
    first_action = torch.zeros((actions.shape[0], 1, actions.shape[-1])).to(device)
    ex_actions = torch.cat([first_action, actions], axis=1)
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def get_action_torch(diffusion_output, action_stats=ACTION_STATS):
    device = diffusion_output.device
    dtype = diffusion_output.dtype
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    min_v = torch.as_tensor(action_stats["min"], device=device, dtype=dtype)
    max_v = torch.as_tensor(action_stats["max"], device=device, dtype=dtype)
    ndata = (ndeltas + 1) / 2
    data = ndata * (max_v - min_v) + min_v
    actions = torch.cumsum(data, dim=1)
    return actions

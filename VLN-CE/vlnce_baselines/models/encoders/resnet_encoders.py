from acm_paths import weight
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
from habitat import logger
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
import torchvision
from pathlib import Path


class VlnResnetDepthEncoder(nn.Module):

    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )
        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)
        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)
            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue
                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v
            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)
        self.spatial_output = spatial_output
        if not self.spatial_output:
            self.output_shape = (output_size,)
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1] * self.visual_encoder.output_shape[2], 64
            )
            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)
        if self.spatial_output:
            b, c, h, w = x.size()
            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0, self.spatial_embeddings.num_embeddings, device=x.device, dtype=torch.long
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )
            return torch.cat([x, spatial_features], dim=1)
        else:
            return x


class CLIPEncoderB16(nn.Module):

    def __init__(self, device, model_name: str = None, local_files_only: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor

        self.device = device
        model_name = model_name or weight("siglip")
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(
            device
        )
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.expected_patches = self._get_expected_patches(self.model)
        self.is_blind = False

    @staticmethod
    def _get_expected_patches(model):
        vision_cfg = getattr(model.config, "vision_config", None)
        if vision_cfg is None:
            return None
        image_size = getattr(vision_cfg, "image_size", None)
        patch_size = getattr(vision_cfg, "patch_size", None)
        if image_size is None or patch_size is None:
            return None
        if isinstance(image_size, (list, tuple)):
            h, w = (image_size[0], image_size[1])
        else:
            h = w = image_size
        return h // patch_size * (w // patch_size)

    @staticmethod
    def _to_hwc_uint8(rgb):
        if torch.is_tensor(rgb):
            rgb_np = rgb.detach().cpu().numpy()
        else:
            rgb_np = rgb
        if rgb_np.ndim == 4 and rgb_np.shape[-1] == 3:
            images = rgb_np
        elif rgb_np.ndim == 4 and rgb_np.shape[1] == 3:
            images = rgb_np.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Unexpected rgb shape: {rgb_np.shape}")
        if images.dtype != np.uint8:
            max_val = images.max() if images.size else 1.0
            if max_val <= 1.5:
                images = (images * 255.0).clip(0, 255).astype(np.uint8)
            else:
                images = images.clip(0, 255).astype(np.uint8)
        return list(images)

    def forward(self, observations):
        images = self._to_hwc_uint8(observations["rgb"])
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                rgb_vector = self.model.get_image_features(**inputs)
            else:
                rgb_vector = None
            if hasattr(self.model, "vision_model"):
                outputs = self.model.vision_model(**inputs)
            else:
                outputs = self.model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            full_tokens = outputs.last_hidden_state
        else:
            full_tokens = outputs[0]
        if rgb_vector is None:
            if getattr(outputs, "pooler_output", None) is not None:
                rgb_vector = outputs.pooler_output
            else:
                rgb_vector = full_tokens[:, 0, :]
        grid_tokens = full_tokens
        if self.expected_patches is not None:
            if grid_tokens.shape[1] == self.expected_patches + 1:
                grid_tokens = grid_tokens[:, 1:, :]
        return (rgb_vector.float(), grid_tokens.float())


class RGBEncoder(nn.Module):

    def __init__(
        self,
        device,
        vit_pretrain: bool = True,
        trainable: bool = False,
        checkpoint_path: str = None,
        repo_path: str = None,
    ):
        super(RGBEncoder, self).__init__()
        self.device = device
        from dinov3.hub.backbones import dinov3_vits16plus

        self.rgb_net = dinov3_vits16plus(
            pretrained=vit_pretrain, weights=checkpoint_path or weight("dinov3.pth")
        )
        self.rgb_net.to(device)
        for param in self.rgb_net.parameters():
            param.requires_grad_(trainable)
        self.rgb_net.eval()
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, rgb_imgs):
        rgb = rgb_imgs["rgb"]
        if rgb.ndim != 4:
            raise ValueError(f"Unexpected rgb shape: {rgb.shape}")
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div(255.0)
        else:
            rgb = rgb.float()
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        rgb = (rgb - self.rgb_mean.to(rgb.device)) / self.rgb_std.to(rgb.device)
        if hasattr(self.rgb_net, "forward_features"):
            feats = self.rgb_net.forward_features(rgb)
            if isinstance(feats, dict):
                rgb_feats = feats.get("x_norm_clstoken", None)
                if rgb_feats is None:
                    rgb_feats = feats.get("x_prenorm", None)
                if rgb_feats is None:
                    raise ValueError("DINOv3 forward_features output missing CLS token.")
            else:
                rgb_feats = feats
        else:
            rgb_feats = self.rgb_net(rgb)
        if rgb_feats.dim() > 2:
            rgb_feats = rgb_feats[:, 0]
        return rgb_feats


class RGBResEncoder(nn.Module):

    def __init__(self, resnet_pretrain=True, trainable=False):
        super(RGBResEncoder, self).__init__()
        if resnet_pretrain:
            pass
        rgb_resnet = torchvision.models.resnet50(pretrained=False)
        rgb_modules = list(rgb_resnet.children())[:-2]
        rgb_net = torch.nn.Sequential(*rgb_modules)
        self.rgb_net = rgb_net
        self.rgb_net.load_state_dict(
            torch.load(weight("resnet50.pth"), map_location="cpu"), strict=True
        )
        for param in self.rgb_net.parameters():
            param.requires_grad_(trainable)
        from torchvision import transforms

        self.rgb_transform = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def forward(self, rgb_imgs):
        rgb_observations = rgb_imgs["rgb"].permute(0, 3, 1, 2)
        rgb_observations = self.rgb_transform(rgb_observations)
        rgb_feats = self.rgb_net(rgb_observations)
        return rgb_feats.squeeze()

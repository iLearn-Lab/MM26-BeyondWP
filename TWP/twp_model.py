import torch
from transformers import PretrainedConfig
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.TRM.TRM_net import BinaryDistPredictor_TRM
from vint_train.models.nomad.nomad_via_cls import NoMaD, Vis_classifier, Fusion

MODEL_CONFIG = {
    "pred_head_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "finetuning_task": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "image_feat_size": 512,
    "depth_feat_size": 128,
    "image_prob_size": 1000,
    "angle_feat_size": 4,
    "obj_feat_size": 0,
    "obj_prob_size": 0,
    "img_feature_type": "imagenet",
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "num_l_layers": 9,
    "num_x_layers": 4,
    "num_pano_layers": 12,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "max_action_steps": 150,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_labels": 2,
    "output_attentions": False,
    "output_hidden_states": False,
    "pruned_heads": {},
    "torchscript": False,
    "type_vocab_size": 2,
    "update_lang_bert": False,
    "vocab_size": 30522,
    "use_lang2visn_attn": True,
    "graph_sprels": True,
    "glocal_fuse": True,
    "lang_bert_name": "bert-base-uncased",
    "num_imgs": 12,
    "obs_neighbors": 1,
    "embed_dim": 64,
    "dis_embed_dim": 16,
    "use_sincos": False,
    "num_distance": 12,
    "num_angle": 120,
    "HEATMAP_OFFSET": 5,
    "depth_output_size": 256,
    "depth_backbone": "resnet50",
    "ddppo_checkpoint": "data/ddppo-models/gibson-2plus-resnet50.pth",
    "spatial_output": False,
    "fusion_mlp_mult": 4,
}


def build_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PretrainedConfig(**MODEL_CONFIG)
    return NoMaD(
        vision_encoder=BinaryDistPredictor_TRM(device),
        noise_pred_net=ConditionalUnet1D(
            input_dim=2, global_cond_dim=768, down_dims=[256, 512, 1024], cond_predict_scale=False
        ),
        direction_classifier=Vis_classifier(config),
        fusion_layer=Fusion(config),
    ).to(device)


def load_model(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state)
    model = build_model(device)
    model.load_state_dict(state, strict=True)
    return model


def scheduler():
    return DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

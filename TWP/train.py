import argparse
from pathlib import Path
import torch
from diffusers.training_utils import EMAModel
from data import loader, on_device
from twp_model import build_model, load_model, scheduler
from trajectory import get_delta, normalize_data, ACTION_STATS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=["prior", "diffusion"], default="diffusion")
    p.add_argument("--init-weights")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.stage == "diffusion" and not a.init_weights:
        p.error("The diffusion stage requires --init-weights from prior training.")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.init_weights, device) if a.init_weights else build_model(device)
    if a.stage == "diffusion":
        model.vision_encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=1e-4)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    ema = EMAModel(model=model, power=0.75)
    noise = scheduler()
    batches = loader(a.data_root, a.annotations, a.batch_size, True, a.limit)
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        if a.stage == "diffusion":
            model.vision_encoder.eval()
        for step, batch in enumerate(batches):
            batch = on_device(batch, device)
            cond, logits = model("vision_encoder", batch=batch)
            if a.stage == "prior":
                target = batch["angle_onehot"].to(logits.dtype)
                loss = (logits - target).square().sum() / target.shape[0] / target.shape[1]
            else:
                cond = model(
                    "fusion_layer",
                    global_cond=cond,
                    direction_label=batch["direction_label"],
                    angle_label=batch["angle_label"],
                    distance_label=batch["distance_label"],
                )
                actions = batch["action_labels"].reshape(-1, 12, 2)
                clean = normalize_data(get_delta(actions), ACTION_STATS)
                eps = torch.randn_like(clean)
                t = torch.randint(0, noise.config.num_train_timesteps, (len(clean),), device=device)
                pred = model(
                    "noise_pred_net",
                    sample=noise.add_noise(clean, eps, t),
                    timestep=t,
                    global_cond=cond.reshape(len(clean), -1),
                )
                masks = batch["action_masks"].reshape(-1, 12, 1).expand_as(pred)
                per_traj = ((pred - eps).square() * masks).sum((1, 2)) / masks.sum(
                    (1, 2)
                ).clamp_min(1e-12)
                valid = batch["mask"].reshape(-1).float()
                loss = (per_traj * valid).mean() / (valid.mean() + 0.01)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step(model)
            print(f"epoch={epoch} batch={step} loss={loss.item():.6f}", flush=True)
            if a.max_batches and step + 1 >= a.max_batches:
                break
        lr.step()
        torch.save(model.state_dict(), output / "latest.pth")
        torch.save(ema.averaged_model.state_dict(), output / "ema_latest.pth")


if __name__ == "__main__":
    main()

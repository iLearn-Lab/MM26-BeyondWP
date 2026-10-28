import argparse
from pathlib import Path
import torch
from data import loader, on_device
from twp_model import load_model, scheduler
from trajectory import get_action


@torch.no_grad()
def predict(model, batch):
    condition, heatmap = model("vision_encoder", batch=batch)
    scores = heatmap.flatten(1)
    indices = scores.topk(5, dim=1).indices
    angles = indices // 12
    distances = indices % 12
    directions = ((angles + 5) // 10) % 12
    condition = model(
        "fusion_layer",
        global_cond=condition,
        direction_label=directions,
        angle_label=angles,
        distance_label=distances,
    )
    size, number, dim = condition.shape
    condition = condition.reshape(-1, dim)
    actions = torch.randn(size * number, 12, 2, device=condition.device)
    noise = scheduler()
    for t in noise.timesteps:
        prediction = model(
            "noise_pred_net",
            sample=actions,
            timestep=t.expand(len(actions)).to(actions.device),
            global_cond=condition,
        )
        actions = noise.step(prediction, t, actions).prev_sample
    return {
        "trajectories": get_action(actions).reshape(size, number, 12, 2) * 0.25,
        "angles": angles,
        "distances": distances,
        "heatmap": heatmap,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.weights, device).eval()
    results = []
    for batch in loader(a.data_root, a.annotations, 1, False, a.limit):
        output = predict(model, on_device(batch, device))
        if not torch.isfinite(output["trajectories"]).all():
            raise RuntimeError("Non-finite prediction")
        results.append({k: v.cpu() for k, v in output.items()})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, a.output)
    print(f"Predicted {len(results)} samples", flush=True)


if __name__ == "__main__":
    main()

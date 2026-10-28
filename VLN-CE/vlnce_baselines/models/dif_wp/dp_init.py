from acm_paths import weight
from twp_model import load_model, scheduler


def get_dif_wp_models(config=None):
    model = load_model(weight("twp.pth"))
    model.requires_grad_(False)
    model.eval()
    return (
        model,
        None,
        scheduler(),
        {"len_traj_pred": 12, "action_dim": 2, "metric_waypoint_spacing": 0.25},
    )

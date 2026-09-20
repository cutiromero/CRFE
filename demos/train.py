"""Two-stage synthetic training demonstration. Run from the repository root."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from core.config import load_config
from core.prior import resize_prior
from demos.common import DemoSegmenter, synthetic_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="runs/demo/model.pt")
    args = parser.parse_args()
    c = load_config(args.config)
    torch.set_num_threads(2)
    torch.manual_seed(c["seed"])
    rgb, labels, coarse_rgb, coarse_labels = synthetic_scene(c, c["seed"])
    coarse, fine = DemoSegmenter(c, False), DemoSegmenter(c, True)
    optimizer = torch.optim.Adam(coarse.parameters(), lr=c["learning_rate"])
    coarse.train()
    for _ in range(c["steps"]):
        optimizer.zero_grad()
        loss = F.cross_entropy(coarse(coarse_rgb), coarse_labels)
        loss.backward()
        optimizer.step()
    coarse.eval()
    with torch.no_grad():
        prior = resize_prior(coarse(coarse_rgb), rgb.shape[-2:])
    optimizer = torch.optim.Adam(fine.parameters(), lr=c["learning_rate"])
    fine.train()
    for _ in range(c["steps"]):
        optimizer.zero_grad()
        loss = F.cross_entropy(fine(rgb, prior), labels)
        loss.backward()
        optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite synthetic training loss")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": c, "coarse": coarse.state_dict(), "fine": fine.state_dict()}, path)
    print(f"Synthetic training completed; final toy loss={loss.item():.4f}; saved {path}")


if __name__ == "__main__":
    main()

"""Inference on new synthetic inputs using the demonstration checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
from core.prior import resize_prior
from demos.common import DemoSegmenter, synthetic_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/demo/model.pt")
    parser.add_argument("--output", default="runs/demo/prediction.npy")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    c = checkpoint["config"]
    coarse, fine = DemoSegmenter(c, False), DemoSegmenter(c, True)
    coarse.load_state_dict(checkpoint["coarse"])
    fine.load_state_dict(checkpoint["fine"])
    coarse.eval()
    fine.eval()
    rgb, _, coarse_rgb, _ = synthetic_scene(c, c["seed"] + 1)
    with torch.no_grad():
        prior = resize_prior(coarse(coarse_rgb), rgb.shape[-2:])
        logits = fine(rgb, prior)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite inference output")
        prediction = logits.argmax(dim=1).numpy().astype(np.uint8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prediction)
    print(f"Synthetic inference completed; output shape={prediction.shape}; saved {path}")


if __name__ == "__main__":
    main()

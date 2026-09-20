"""Configuration for the synthetic demonstration only."""
import json
from pathlib import Path


def load_config(path: str) -> dict:
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"image_size", "classes", "embed_dim", "heads", "window", "steps", "learning_rate", "seed"}
    if set(c) - required - {"description"} or required - set(c):
        raise ValueError("Unexpected or missing demo configuration fields")
    for key in required - {"learning_rate"}:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["classes"] != 2:
        raise ValueError("The synthetic demo uses two classes")
    if c["image_size"] % (2 * c["window"]) or c["embed_dim"] % c["heads"]:
        raise ValueError("Image size must be divisible by twice the window; embedding dimension by heads")
    if not 0 < c["learning_rate"] < 1:
        raise ValueError("Expected a learning rate between 0 and 1")
    return c

# CRFE: Cascade Receptive Field Expansion Framework for Large-scale Mapping in VHR Remote Sensing Imagery

Components of Cascade Receptive Field Expansion (CRFE) for tile-based remote sensing segmentation.

## Key modules

- **Scale-aware prior attention** (`core/sapam.py`): scale embeddings, feature modulation, window-based prior-guided cross-attention, and residual feature fusion.
- **Prior transfer** (`core/prior.py`): nearest-neighbor transfer of coarse class predictions to a finer grid covering the same extent.
- **Tile and input processing** (`core/inputs.py`): tile extraction, edge padding, RGB normalization, and class-prior concatenation.
- **Training and inference demonstrations** (`demos/`): a small two-stage example showing how a coarse prediction supplies the prior for fine-stage SA-PAM.

## Quick start

From the repository root, with Python 3.9+ and PyTorch installed:

```console
python -m pip install -r requirements.txt
python -m demos.train
python -m demos.infer
```

The demonstrations use synthetic RGB arrays, two label classes, and a small purpose-built segmentation head. `config.json` contains demonstration settings selected independently of the manuscript experiments. Training writes `runs/demo/model.pt`; inference uses new synthetic inputs and writes `runs/demo/prediction.npy`. 

The commands have been checked on CPU with Python 3.9, PyTorch 2.7.1 and NumPy 1.26.4.

## Interfaces

`SA_PAM` receives RGB `(B, 3, H, W)`, a normalized single-channel class prior `(B, 1, H, W)`, and a zero-based scale index. It returns `(B, embed_dim, H, W)` features for a downstream segmentation model. Spatial dimensions must be divisible by the attention window size.

`resize_prior` takes coarse logits and returns an upsampled hard-label prior normalized by `classes - 1`. The coarse and fine grids must already cover the same extent. `inputs` instead accepts raw class-label arrays and applies normalization itself; do not normalize its prior twice.

Configuration keys `heads` and `window` map to SA-PAM constructor arguments `num_heads` and `window_size`; the demonstration passes them explicitly.

## Public datasets

| Dataset | Official or author-maintained resource |
| --- | --- |
| GID / GID-Large | [GID project page](https://x-ytong.github.io/project/GID.html), large-scale classification set |
| ISPRS Potsdam | [ISPRS semantic labeling benchmark](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx) |
| ISPRS Vaihingen | [ISPRS semantic labeling benchmark](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx) |
| LandCover.ai | [Project website](https://landcover.ai.linuxpolska.com/), [author-maintained Kaggle page](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai) |
| WHU-Building | [WHU dataset page](https://gpcv.whu.edu.cn/data/building_dataset.html) |
| GLH-Water | [GLH-Water project page](https://jack-bo1220.github.io/project/GLH-water.html) |

Follow each provider's access conditions and cite the corresponding dataset publication. 

## Availability

This review release provides the modules and demonstrations described above. A complete release is planned following further development and the relevant intellectual-property procedures.

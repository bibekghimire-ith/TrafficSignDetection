# Dependencies

Source: `requirements.txt`, `Dockerfile`. No lock file. The Docker image pins only the base image and `notebook`.

| dependency | declared | used for | why it matters |
|---|---|---|---|
| Python | 3.7+ (README), `python:3.11-slim` (Dockerfile) | everything | The Phase 2 experiments ran on 3.11. |
| numpy | `>=1.19` | the whole network, data arrays | **NumPy ≥ 2 promotes float32 × NumPy-float64-scalar to float64** (NEP 50). `update_parameters` uses Python floats to keep float32 (`src/ffnn.py:588`). `np.load` of the npz cache uses the default `allow_pickle=False`. |
| Pillow | `>=8.0` | decode `.ppm`/jpg/png, ROI crop, `ImageOps.equalize`, resize | `preprocess_image` (`src/dataset.py:181`) defines the input distribution. A change in Pillow's resampling changes the cache. |
| scipy | `>=1.5` | `ndimage` rotate/shift/blur/zoom in augmentation | Only imported by `src/data_augmentation.py`. Train and evaluate run without it unless `--use-augmented` is set. |
| matplotlib | `>=3.3` | curves, confusion matrix, sample grids | Imported at module top by `src/dataset.py` and `src/model_utils.py`, so it is required even for headless runs. Use `MPLBACKEND=Agg`. |
| tkinter | system package | `gui.py` only | Not in the Docker image. |
| notebook | `>=7,<8` (Dockerfile only) | `Project_Walkthrough.ipynb` | The container's default command runs Jupyter with the token disabled. |

No ML framework is used, by design.

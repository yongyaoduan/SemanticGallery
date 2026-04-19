# Training

SemanticGallery uses a two-stage retrieval setup.

- `Stage 1` is the published checkpoint bundled with the desktop release.
- `Stage 2` is the optional private adaptation step you can start from `Settings`.

## Default Path: Private Adaptation In Settings

This is the normal path for the desktop app. You choose a folder, let it finish indexing, then press `Start Private Adaptation` in `Settings`.

### Inputs

| Input | Role |
| --- | --- |
| Published Stage 1 checkpoint | Starting point for the image and text encoders |
| Stage 2 public anchor set | Keeps public text-image alignment active during Stage 2 |
| Local adaptation set | `100` images sampled from the selected folder |

### What Stage 2 Optimizes

The text tower stays frozen. The image side keeps three losses active:

`L = 1.0 * L_public_txtimg + 0.3 * L_private_instance + 0.15 * L_distill`

| Loss | Purpose |
| --- | --- |
| `L_public_txtimg` | Keeps the retrieval space aligned to public text-image supervision |
| `L_private_instance` | Pulls two augmented views of the same local image together |
| `L_distill` | Keeps the adapted image encoder close to the published Stage 1 teacher |

### Training Schedule

- The app keeps the same `100` private images across the run.
- It samples `100` public images without replacement per epoch from the bundled `1000`-image public anchor pool.
- One adaptation session runs `10` local epochs.
- When training finishes, the app switches future embeddings to the adapted encoder and rebuilds every stored embedding in the library database.

## Development Note

The original project notes were written around shell scripts and a browser workflow. The current desktop release runs natively in Swift and MLX, but it keeps the same retrieval idea: published public alignment first, then optional gallery-specific adaptation on top.

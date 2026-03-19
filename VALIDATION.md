# Validation

This repository has been checked as an engineering utility for vjepa2.1 dense-feature PCA visualization.

## What Is Verified

- The CLI package entrypoints work:
  - `vjepa21-pca`
  - `python -m vjepa2_1_pca`
- PCA is deterministic across repeated runs on the same features.
- PCA outputs are normalized to `[0, 1]`.
- The token-grid geometry matches the expected `T x H x W` layout.
- The saved panels pair the original frame with the PCA-rendered dense token view.
- Image inputs and video inputs both run through the released vjepa2.1 encoder path.
- The bundled sample video and released checkpoints run successfully.

## What Is Not Claimed

- This repository does not claim pixel-perfect reproduction of any paper teaser figure.
- PCA colors are not semantic labels.
- Different PCA implementations can produce sign or channel-order differences even when the qualitative structure is correct.

## Practical Conclusion

As a visualization tool, the PCA pipeline is correct and stable enough for:

- qualitative dense-feature inspection
- checkpoint debugging
- comparing `last` and `hierarchical` representations
- communicating vjepa2.1 dense-feature behavior to the open-source community

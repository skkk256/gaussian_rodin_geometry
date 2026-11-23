# Third-party runtime sources

- `sam3d_objects/`: the SAM3D stage1 Python runtime used by the validated Gaussian pipeline; upstream module copyright headers are retained.
- `dinov2_source/`: the matching local DINOv2 implementation, including its upstream license files.

These are local source copies, not imports from the old deployment directory. SAM3D's DINO checkpoint default points to this project's `ckpt/`; automatic GPU-0 attention selection at import time has been removed. The owning-device attention selection happens explicitly in `src/coarse.py`. Model weights remain outside these source trees and are referenced only through `ckpt/` symlinks.

Browser dependencies are not included in Git. The optional result viewer uses locally provided `viewer/three/` and `viewer/spark/`; the optional alignment viewer uses locally provided `three/` and `spark/`. These directories are ignored by Git and are not needed for geometry generation. Retain the upstream licenses when providing them locally.

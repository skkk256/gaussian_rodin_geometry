# Third-party runtime sources

- `sam3d_objects/`: the SAM3D stage1 Python runtime used by the validated Gaussian pipeline; upstream module copyright headers are retained.
- `dinov2_source/`: the matching local DINOv2 implementation, including its upstream license files.
- `viewer/`: the Three.js and Spark browser runtime previously used by deploy v3. Only the renderer, GLTF loader, orbit controls and their direct dependencies are copied, together with both license files; no old deployment code is imported.

These are local source copies, not imports from the old deployment directory. SAM3D's DINO checkpoint default points to this project's `ckpt/`; automatic GPU-0 attention selection at import time has been removed. The owning-device attention selection happens explicitly in `src/coarse.py`. Model weights remain outside these source trees and are referenced only through `ckpt/` symlinks.

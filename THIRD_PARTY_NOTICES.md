# Source origins and retained notices

This repository combines the existing local 4DGaussians research modifications
with its dependencies as ordinary tracked files. Directory names are preserved
for compatibility, but no Git submodule checkout is required.

| Component | Source / retained license |
| --- | --- |
| 4DGaussians | https://github.com/hustvl/4DGaussians ; root `LICENSE.md` and per-file notices |
| Modified depth rasterizer | https://github.com/fengqiyu0317/depth-diff-gaussian-rasterization ; `submodules/depth-diff-gaussian-rasterization/LICENSE.md` and per-file notices |
| simple-knn | https://gitlab.inria.fr/bkerbl/simple-knn.git ; original notices in the vendored source files (the local checkout has no separate LICENSE file) |
| GLM headers | https://github.com/g-truc/glm ; `submodules/depth-diff-gaussian-rasterization/third_party/glm/copying.txt` |
| gsplat-derived mixed raster code | https://github.com/nerfstudio-project/gsplat ; `gsplat_mixed_ext/LICENSE.gsplat` |

The root license does not replace the component-specific licenses or notices.
The Python gsplat dependency remains external and is pinned to 1.5.3.
PyTorch, torchvision, CUDA and dataset/model files are not vendored.

The original local checkout before conversion used the following base commits,
with additional uncommitted modifications included in this source release:

- 4DGaussians: `60018d41f4f57b693b0e0dc4052f85b0c9d31a99`.
- Modified rasterizer: `464096373b869fff713265bcdc34b1822277980d`.
- simple-knn: `44f764299fa305faf6ec5ebd99939e0508331503`.

The `e495066...` rasterizer identifier embedded in ABI/profile contracts is a
historical compatibility identifier, not a claim that this release has that
Git revision. Those contracts and CUDA implementations are preserved.

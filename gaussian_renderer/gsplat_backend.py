"""Inference adapter for gsplat, preserving the legacy RGB/radii/depth API.

Validated against gsplat 1.5.3. Importing the default renderer does not import
gsplat; it is an optional dependency. Training is deliberately rejected because
the legacy dummy means2D gradient used by densification is not gsplat's API.
"""

import torch


class GsplatRasterizer:
    def __init__(self, raster_settings, packed=False):
        if torch.is_grad_enabled():
            raise RuntimeError("the gsplat backend currently supports inference only; use torch.no_grad()")
        try:
            from gsplat import rasterization
        except ImportError as error:
            raise ImportError("gsplat backend requested; install gsplat==1.5.3 in the inference environment") from error
        self._rasterization = rasterization
        self.raster_settings = raster_settings
        self.packed = packed
        settings = raster_settings
        width, height = settings.image_width, settings.image_height
        # Legacy matrices are transposed for column-major CUDA access. gsplat
        # expects a conventional world-to-camera matrix. Its pixel centers are
        # (x+.5,y+.5), versus integer centers in the legacy rasterizer, so W/2
        # and H/2 give the same rays as legacy ndc2Pix's (W-1)/2 offset.
        self.viewmats = settings.viewmatrix.transpose(0, 1).contiguous().unsqueeze(0)
        self.Ks = torch.tensor(
            [[width / (2 * settings.tanfovx), 0, width / 2],
             [0, height / (2 * settings.tanfovy), height / 2],
             [0, 0, 1]],
            dtype=settings.viewmatrix.dtype, device=settings.viewmatrix.device,
        ).unsqueeze(0)

    def record_stream(self, stream):
        self.viewmats.record_stream(stream)
        self.Ks.record_stream(stream)

    def __call__(self, means3D, means2D, shs, colors_precomp, opacities,
                 scales, rotations, cov3D_precomp=None):
        del means2D
        if torch.is_grad_enabled():
            raise RuntimeError("the gsplat backend currently supports inference only")
        settings = self.raster_settings
        covars = None
        if cov3D_precomp is not None:
            # Legacy covariance stores xx,xy,xz,yy,yz,zz; scale_modifier is
            # already included by GaussianModel.get_covariance().
            c = cov3D_precomp
            covars = torch.stack((c[:, 0], c[:, 1], c[:, 2],
                                 c[:, 1], c[:, 3], c[:, 4],
                                 c[:, 2], c[:, 4], c[:, 5]), dim=-1).reshape(-1, 3, 3)
        elif settings.scale_modifier != 1.0:
            scales = scales * settings.scale_modifier
        image, _alpha, meta = self._rasterization(
            means=means3D, quats=rotations, scales=scales,
            opacities=opacities.squeeze(-1),
            colors=colors_precomp if colors_precomp is not None else shs,
            sh_degree=None if colors_precomp is not None else settings.sh_degree,
            covars=covars, viewmats=self.viewmats, Ks=self.Ks,
            width=settings.image_width, height=settings.image_height,
            near_plane=0.2, far_plane=1e10, eps2d=0.3,
            radius_clip=0.0, rasterize_mode="classic", packed=self.packed,
            # gsplat 1.5.3's packed wrapper checks a background shape without
            # the camera dimension, while rasterization() requires that dim.
            # Render against zero and composite the RGB background below.
            backgrounds=None if self.packed else settings.bg.unsqueeze(0),
            render_mode="RGB+D",
        )
        radii = meta["radii"]
        # Support both scalar and newer per-axis radii without exposing
        # packed Gaussian indexing through the legacy RenderResult interface.
        if self.packed:
            if radii.ndim == 2:
                radii = radii.amax(dim=-1)
            full_radii = torch.zeros(means3D.shape[0], dtype=radii.dtype, device=radii.device)
            full_radii[meta["gaussian_ids"].long()] = radii
            radii = full_radii
        else:
            radii = radii[0]
            if radii.ndim == 2:
                radii = radii.amax(dim=-1)
        rgb = image[0, ..., :3]
        if self.packed:
            rgb = rgb + (1 - _alpha[0]) * settings.bg
        return (rgb.permute(2, 0, 1).contiguous(), radii,
                image[0, ..., 3].unsqueeze(0).contiguous())

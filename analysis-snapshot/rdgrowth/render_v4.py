"""render_v4.py (VERSION render_v4 2026-10-31): upstream triangle_renderer.render with two cost-only changes.

Same outputs as upstream `render` (same tensors, same values) except:
  1. When the model is not supersampled (pc.scaling == 1) the five `F.interpolate(mode="area")` calls are skipped.
     An area/adaptive-average pool whose output size equals its input size averages exactly one element per output,
     so skipping it is bit-identical in forward and backward; it only removes 5 copies + 5 backward kernels per iteration.
  2. `expected_depth` (used only by the monocular-depth loss) is downsampled only when the camera has an
     `invdepthmap`; otherwise the key holds None.  Mip-360 / T&T cameras have none.
Everything else is the upstream code, verbatim, with its helpers imported from triangle_renderer.
"""
import math
import torch
import torch.nn.functional as F
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from scene.triangle_model import TriangleModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from triangle_renderer import transform_point_4x3, compute_image_2d_pytorch_exact

VERSION = 'render_v4 2026-10-31'


def _down(x, H_init, W_init, upsample):
    """area-downsample [C,H,W] -> [C,H_init,W_init]; identity when not supersampled (bit-identical, see module doc)."""
    if upsample == 1:
        return x
    return F.interpolate(x.unsqueeze(0), size=(H_init, W_init), mode="area").squeeze(0)


def render(viewpoint_camera, pc: TriangleModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    triangles_indices = pc.get_triangle_indices
    vertices = pc.get_vertices
    vertex_weights = pc.get_vertex_weight
    scaling = torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()

    vertex_index = pc._triangle_indices.shape[0]

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H_init = int(viewpoint_camera.image_height)
    W_init = int(viewpoint_camera.image_width)

    upsample = pc.scaling

    H = upsample * H_init
    W = upsample * W_init

    raster_settings = TriangleRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    sigma = pc.get_sigma

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, scaling, allmap, max_blending, was_rendered = rasterizer(
        vertices=vertices,
        triangles_indices=triangles_indices,
        vertex_weights=vertex_weights.squeeze(),
        sigma=sigma,
        shs=shs,
        colors_precomp=colors_precomp,
        scaling=scaling,
    )

    radii = radii[:vertex_index]
    scaling = scaling[:vertex_index]
    max_blending = max_blending[:vertex_index]

    rendered_image_small = _down(rendered_image, H_init, W_init, upsample)

    V = vertices.shape[0]
    idx = triangles_indices[was_rendered > 0].reshape(-1).long()
    vertex_rendered = torch.zeros(V, device=triangles_indices.device, dtype=was_rendered.dtype)
    vertex_rendered.index_fill_(0, idx, 1)

    vertices_cam = transform_point_4x3(vertices, viewpoint_camera.world_view_transform,)
    vertex_depths_pytorch = vertices_cam[:, 2]
    image_2D_pytorch = compute_image_2d_pytorch_exact(vertices, viewpoint_camera.full_proj_transform, W_init, H_init)

    rets = {"render": rendered_image_small,
            "visibility_filter": radii > 0,
            "radii": radii,
            "scaling": scaling,
            "max_blending": max_blending,
            "vertex_depth_out": vertex_depths_pytorch,
            "image_2D": image_2D_pytorch,
            "vertex_rendered": vertex_rendered,
            "full_image": rendered_image,
            "render_normal_full": allmap[2:5],
            "triangle_was_rendered": was_rendered
            }

    render_alpha = _down(allmap[1:2], H_init, W_init, upsample)

    render_normal = _down(allmap[2:5], H_init, W_init, upsample)
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    render_depth_median = _down(allmap[5:6], H_init, W_init, upsample)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    if getattr(viewpoint_camera, "invdepthmap", None) is not None:
        render_expected_depth = _down(allmap[0:1], H_init, W_init, upsample)
        render_expected_depth = torch.nan_to_num(render_expected_depth, 0, 0)
    else:
        render_expected_depth = None

    render_id = allmap[6:7]

    surf_depth = render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_ids': render_id,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            "expected_depth": render_expected_depth,
            "depth_full": allmap[5:6]
    })

    return rets

#!/usr/bin/env python3
"""
Export SLAM3R models to ONNX.

First tries to export the Multiview3D backbone (encoder-only). If that fails
(e.g. custom RoPE/cuROPE), falls back to exporting the two submodels:
  - Image2PointsModel (I2P): multiple views -> 3D pointmaps
  - Local2WorldModel (L2W): views + 3D pointmaps -> refined/world pointmaps

Outputs are written to andy/onnx/ by default. Uses the same directory layout
as andy/onnx_quantize.py.

Requires: slam3r package (project root on PYTHONPATH or run from project root).
Checkpoints: use --i2p_weights and --l2w_weights. Either pass a local path
  (e.g. checkpoints/slam3r_i2p.pth) or a HuggingFace repo id; the script uses
  Image2PointsModel.from_pretrained('siyan824/slam3r_i2p') and
  Local2WorldModel.from_pretrained('siyan824/slam3r_l2w') when the value
  looks like a repo id (contains '/' and is not an existing file).
"""

import argparse
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import onnx

# Suppress TracerWarnings from slam3r during ONNX trace (e.g. len(tensor), int(), .all())
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
# Suppress FutureWarning from slam3r (torch.cuda.amp.autocast -> torch.amp.autocast)
warnings.filterwarnings("ignore", category=FutureWarning, module="slam3r")

# Project root = parent of andy/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default model configs (match recon.py)
I2P_KWARGS = dict(
    pos_embed="RoPE100",
    img_size=(224, 224),
    head_type="linear",
    output_mode="pts3d",
    depth_mode=("exp", float("-inf"), float("inf")),
    conf_mode=("exp", 1, float("inf")),
    enc_embed_dim=1024,
    enc_depth=24,
    enc_num_heads=16,
    dec_embed_dim=768,
    dec_depth=12,
    dec_num_heads=12,
    mv_dec1="MultiviewDecoderBlock_max",
    mv_dec2="MultiviewDecoderBlock_max",
    enc_minibatch=11,
)

L2W_KWARGS = dict(
    pos_embed="RoPE100",
    img_size=(224, 224),
    head_type="linear",
    output_mode="pts3d",
    depth_mode=("exp", float("-inf"), float("inf")),
    conf_mode=("exp", 1, float("inf")),
    enc_embed_dim=1024,
    enc_depth=24,
    enc_num_heads=16,
    dec_embed_dim=768,
    dec_depth=12,
    dec_num_heads=12,
    mv_dec1="MultiviewDecoderBlock_max",
    mv_dec2="MultiviewDecoderBlock_max",
    enc_minibatch=11,
    need_encoder=False,
)


def _load_ckpt(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("model", ckpt)


def _export_onnx(
    model: nn.Module,
    args: Tuple[Any, ...],
    out_path: str,
    *,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset: int = 17,
) -> None:
    """Export model to ONNX. Tries classic export first; on tuple/graph errors, falls back to dynamo_export."""
    # #region agent log
    _log = lambda h, msg, **d: open("/home/campus.ncl.ac.uk/c4071391/Projects/SLAM3R/.cursor/debug.log", "a").write(
        __import__("json").dumps({"hypothesisId": h, "message": msg, "data": d, "timestamp": __import__("time").time()}) + "\n"
    ) or None
    _log("H1", "export_onnx entry", out_path=out_path)
    # #endregion
    dynamic_axes = dynamic_axes or {}
    try:
        _log("H2", "attempting classic torch.onnx.export")
        torch.onnx.export(
            model,
            args,
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except RuntimeError as e:
        err_msg = str(e)
        # #region agent log
        _log("H2", "classic export failed", err_msg=err_msg[:200])
        # #endregion
        # Try dynamo when classic fails: tuple/graph bug, or unsupported op (e.g. aten::cartesian_prod)
        if (
            "INTERNAL ASSERT" in err_msg
            or "outerNode" in err_msg
            or "dead_code_elimination" in err_msg
            or "is not supported" in err_msg
        ):
            # Trace-based export hits tuple/graph bugs; try dynamo exporter (PyTorch 2.1+)
            if not hasattr(torch.onnx, "dynamo_export"):
                raise RuntimeError(
                    f"Classic ONNX export failed ({e}). Install PyTorch 2.1+ for dynamo_export fallback."
                ) from e
            print(f"Classic export failed ({type(e).__name__}), trying dynamo_export...")
            try:
                # #region agent log
                _log("H3", "attempting dynamo_export")
                # #endregion
                export_options = getattr(torch.onnx, "ExportOptions", None)
                opts = export_options(dynamic_shapes=True) if export_options else None
                kwargs = {"export_options": opts} if opts else {}
                program = torch.onnx.dynamo_export(model, *args, **kwargs)
                program.save(out_path)
                # #region agent log
                _log("H3", "dynamo_export succeeded")
                # #endregion
            except Exception as e2:
                # #region agent log
                cause = e2.__cause__ or e2
                cause_msg = str(cause) if cause else ""
                _log("H1", "dynamo_export failed", e2_type=type(e2).__name__, e2_msg=str(e2)[:200], cause_type=type(cause).__name__ if cause else None, cause_msg=cause_msg[:400])
                # #endregion
                raise RuntimeError(
                    f"Classic ONNX export failed: {e}. Dynamo export also failed: {e2}"
                ) from e2
        else:
            raise


def _is_pretrained_repo(weights: str) -> bool:
    """True if weights looks like a HuggingFace repo id (e.g. siyan824/slam3r_i2p)."""
    return "/" in weights and not os.path.isfile(weights)


def _load_i2p_model(weights: str, device: torch.device) -> nn.Module:
    """Load Image2PointsModel from a local checkpoint path or HuggingFace repo id."""
    from slam3r.models import Image2PointsModel

    if _is_pretrained_repo(weights):
        print(f"Loading Image2PointsModel from HuggingFace: {weights}")
        model = Image2PointsModel.from_pretrained(weights)
    else:
        model = Image2PointsModel(**I2P_KWARGS)
        state = _load_ckpt(weights)
        model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def _load_l2w_model(weights: str, device: torch.device) -> nn.Module:
    """Load Local2WorldModel from a local checkpoint path or HuggingFace repo id."""
    from slam3r.models import Local2WorldModel

    if _is_pretrained_repo(weights):
        print(f"Loading Local2WorldModel from HuggingFace: {weights}")
        model = Local2WorldModel.from_pretrained(weights)
    else:
        model = Local2WorldModel(**L2W_KWARGS)
        state = _load_ckpt(weights)
        model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# 1. Multiview3D (encoder-only) wrapper for ONNX
# ---------------------------------------------------------------------------
class Multiview3DEncoderONNX(nn.Module):
    """Wrapper to export Multiview3D encoder only: one image -> encoder features."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, image: torch.Tensor, true_shape: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, H, W), true_shape: (B, 2)
        enc, pos, _ = self.backbone._encode_image(image, true_shape, normalize=True)
        return enc


# ---------------------------------------------------------------------------
# 2. Image2PointsModel wrapper: fixed 2-view (ref + 1 source), tensor in/out
# ---------------------------------------------------------------------------
class Image2PointsONNX(nn.Module):
    """Wrapper for ONNX: ref_img, src_img, true_shape_ref, true_shape_src -> pts3d and conf."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        ref_img: torch.Tensor,
        src_img: torch.Tensor,
        true_shape_ref: torch.Tensor,
        true_shape_src: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ref_img, src_img: (B, 3, H, W); true_shape_*: (B, 2)
        views = [
            {"img": ref_img, "true_shape": true_shape_ref},
            {"img": src_img, "true_shape": true_shape_src},
        ]
        results = self.model(views, ref_id=0, return_corr_score=False)
        # results[0] = ref: pts3d, conf; results[1] = src: pts3d_in_other_view, conf
        pts3d_ref = results[0]["pts3d"]
        conf_ref = results[0]["conf"]
        pts3d_src = results[1]["pts3d_in_other_view"]
        conf_src = results[1]["conf"]
        return pts3d_ref, conf_ref, pts3d_src, conf_src


# ---------------------------------------------------------------------------
# 3. Local2WorldModel wrapper: 2-view with pts3d, tensor in/out
# L2W uses need_encoder=False, so inputs are encoder tokens + poses + pts3d.
# ---------------------------------------------------------------------------
class Local2WorldONNX(nn.Module):
    """Wrapper for ONNX: ref/src img_tokens, poses, pts3d_world, pts3d_cam -> refined pts3d."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        ref_img_tokens: torch.Tensor,
        src_img_tokens: torch.Tensor,
        ref_poses: torch.Tensor,
        src_poses: torch.Tensor,
        pts3d_world: torch.Tensor,
        pts3d_cam: torch.Tensor,
        ref_true_shape: torch.Tensor,
        src_true_shape: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ref_img_tokens, src_img_tokens: (B, S, D); poses: (B, S, 2)
        # pts3d_world, pts3d_cam: (B, H, W, 3); true_shapes: (B, 2)
        views = [
            {
                "img_tokens": ref_img_tokens,
                "true_shape": ref_true_shape,
                "img_pos": ref_poses,
                "pts3d_world": pts3d_world,
            },
            {
                "img_tokens": src_img_tokens,
                "true_shape": src_true_shape,
                "img_pos": src_poses,
                "pts3d_cam": pts3d_cam,
            },
        ]
        results = self.model(views, ref_ids=0)
        pts3d_ref = results[0]["pts3d"]
        conf_ref = results[0]["conf"]
        pts3d_src = results[1]["pts3d_in_other_view"]
        conf_src = results[1]["conf"]
        return pts3d_ref, conf_ref, pts3d_src, conf_src


def try_export_multiview3d_encoder(
    output_dir: str,
    device: torch.device,
    opset: int,
    i2p_weights: Optional[str],
) -> bool:
    """Try to export Multiview3D (encoder-only). Returns True if successful."""
    if not i2p_weights:
        print("Skipping Multiview3D encoder export (no I2P weights).")
        return False
    if not _is_pretrained_repo(i2p_weights) and not os.path.isfile(i2p_weights):
        print(f"Skipping Multiview3D encoder export (not a file or repo id: {i2p_weights}).")
        return False

    print("Attempting Multiview3D (encoder-only) ONNX export...")
    try:
        model = _load_i2p_model(i2p_weights, device)

        wrapper = Multiview3DEncoderONNX(model)
        wrapper.eval().to(device)

        B, C, H, W = 1, 3, 224, 224
        image = torch.randn(B, C, H, W, device=device)
        true_shape = torch.tensor([[H, W]], dtype=torch.long, device=device)

        out_path = os.path.join(output_dir, "multiview3d_encoder.onnx")
        _export_onnx(
            wrapper,
            (image, true_shape),
            out_path,
            input_names=["image", "true_shape"],
            output_names=["enc_tokens"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "true_shape": {0: "batch"},
                "enc_tokens": {0: "batch"},
            },
            opset=opset,
        )
        onnx.checker.check_model(onnx.load(out_path))
        print(f"Saved: {out_path}")
        return True
    except Exception as e:
        print(f"Multiview3D encoder export failed: {e}")
        return False


def export_image2points(
    output_dir: str,
    device: torch.device,
    opset: int,
    i2p_weights: str,
) -> None:
    """Export Image2PointsModel to ONNX (2-view)."""
    print("Loading Image2PointsModel...")
    model = _load_i2p_model(i2p_weights, device)

    wrapper = Image2PointsONNX(model)
    wrapper.eval().to(device)

    B, C, H, W = 1, 3, 224, 224
    ref_img = torch.randn(B, C, H, W, device=device)
    src_img = torch.randn(B, C, H, W, device=device)
    true_shape_ref = torch.tensor([[H, W]], dtype=torch.long, device=device)
    true_shape_src = torch.tensor([[H, W]], dtype=torch.long, device=device)

    out_path = os.path.join(output_dir, "image2points.onnx")
    _export_onnx(
        wrapper,
        (ref_img, src_img, true_shape_ref, true_shape_src),
        out_path,
        input_names=["ref_img", "src_img", "true_shape_ref", "true_shape_src"],
        output_names=["pts3d_ref", "conf_ref", "pts3d_src", "conf_src"],
        dynamic_axes={
            "ref_img": {0: "batch", 2: "height", 3: "width"},
            "src_img": {0: "batch", 2: "height", 3: "width"},
            "true_shape_ref": {0: "batch"},
            "true_shape_src": {0: "batch"},
            "pts3d_ref": {0: "batch"},
            "conf_ref": {0: "batch"},
            "pts3d_src": {0: "batch"},
            "conf_src": {0: "batch"},
        },
        opset=opset,
    )
    onnx.checker.check_model(onnx.load(out_path))
    print(f"Saved: {out_path}")


def export_local2world(
    output_dir: str,
    device: torch.device,
    opset: int,
    l2w_weights: str,
) -> None:
    """Export Local2WorldModel to ONNX (2-view, need_encoder=False)."""
    print("Loading Local2WorldModel...")
    model = _load_l2w_model(l2w_weights, device)

    wrapper = Local2WorldONNX(model)
    wrapper.eval().to(device)

    # 224/16 = 14 -> 14*14 = 196 tokens; enc_embed_dim = 1024
    B, S, D = 1, 196, 1024
    H, W = 224, 224
    ref_img_tokens = torch.randn(B, S, D, device=device)
    src_img_tokens = torch.randn(B, S, D, device=device)
    ref_poses = torch.zeros(B, S, 2, dtype=torch.long, device=device)
    src_poses = torch.zeros(B, S, 2, dtype=torch.long, device=device)
    pts3d_world = torch.randn(B, H, W, 3, device=device)
    pts3d_cam = torch.randn(B, H, W, 3, device=device)
    ref_true_shape = torch.tensor([[H, W]], dtype=torch.long, device=device)
    src_true_shape = torch.tensor([[H, W]], dtype=torch.long, device=device)

    out_path = os.path.join(output_dir, "local2world.onnx")
    l2w_args = (
        ref_img_tokens,
        src_img_tokens,
        ref_poses,
        src_poses,
        pts3d_world,
        pts3d_cam,
        ref_true_shape,
        src_true_shape,
    )
    _export_onnx(
        wrapper,
        l2w_args,
        out_path,
        input_names=[
            "ref_img_tokens",
            "src_img_tokens",
            "ref_poses",
            "src_poses",
            "pts3d_world",
            "pts3d_cam",
            "ref_true_shape",
            "src_true_shape",
        ],
        output_names=["pts3d_ref", "conf_ref", "pts3d_src", "conf_src"],
        dynamic_axes={
            "ref_img_tokens": {0: "batch"},
            "src_img_tokens": {0: "batch"},
            "ref_poses": {0: "batch"},
            "src_poses": {0: "batch"},
            "pts3d_world": {0: "batch"},
            "pts3d_cam": {0: "batch"},
            "ref_true_shape": {0: "batch"},
            "src_true_shape": {0: "batch"},
            "pts3d_ref": {0: "batch"},
            "conf_ref": {0: "batch"},
            "pts3d_src": {0: "batch"},
            "conf_src": {0: "batch"},
        },
        opset=opset,
    )
    onnx.checker.check_model(onnx.load(out_path))
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SLAM3R Multiview3D / Image2Points / Local2World to ONNX."
    )
    parser.add_argument(
        "--i2p_weights",
        type=str,
        default=None,
        help="Image2Points: local path (e.g. checkpoints/i2p.pth) or HuggingFace repo id (e.g. siyan824/slam3r_i2p).",
    )
    parser.add_argument(
        "--l2w_weights",
        type=str,
        default=None,
        help="Local2World: local path or HuggingFace repo id (e.g. siyan824/slam3r_l2w).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for ONNX files (default: andy/onnx).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--skip_multiview3d",
        action="store_true",
        help="Skip attempting Multiview3D encoder export and only export I2P and L2W.",
    )
    args = parser.parse_args()

    # Disable NVTX profiling during export so dynamo can trace (MyNvtxRange becomes a no-op)
    os.environ["SLAM3R_ONNX_EXPORT"] = "1"

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "onnx")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Try Multiview3D (encoder-only) first
    if not args.skip_multiview3d:
        success = try_export_multiview3d_encoder(
            output_dir, device, args.opset, args.i2p_weights
        )
        if not success:
            print("Falling back to Image2PointsModel and Local2WorldModel exports.")
    else:
        print("Skipping Multiview3D (--skip_multiview3d).")

    # 2. Export Image2PointsModel
    if args.i2p_weights and (_is_pretrained_repo(args.i2p_weights) or os.path.isfile(args.i2p_weights)):
        export_image2points(output_dir, device, args.opset, args.i2p_weights)
    else:
        print("No --i2p_weights or file/repo not found; skipping Image2PointsModel export.")

    # 3. Export Local2WorldModel
    if args.l2w_weights and (_is_pretrained_repo(args.l2w_weights) or os.path.isfile(args.l2w_weights)):
        export_local2world(output_dir, device, args.opset, args.l2w_weights)
    else:
        print("No --l2w_weights or file/repo not found; skipping Local2WorldModel export.")

    print("Done. ONNX files in:", output_dir)


if __name__ == "__main__":
    main()

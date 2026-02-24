# Andy scripts

## ONNX conversion (SLAM3R)

Export SLAM3R models (Multiview3D encoder, Image2Points, Local2World) to ONNX. Run from **project root**.

```bash
# Use HuggingFace pretrained weights (same as from_pretrained in code)
python andy/onnx_conversion.py --i2p_weights siyan824/slam3r_i2p --l2w_weights siyan824/slam3r_l2w

# Or local checkpoint paths
python andy/onnx_conversion.py --i2p_weights /path/to/i2p.pth --l2w_weights /path/to/l2w.pth

# Custom output directory
python andy/onnx_conversion.py --i2p_weights siyan824/slam3r_i2p --l2w_weights siyan824/slam3r_l2w --output_dir andy/onnx

# Skip Multiview3D encoder attempt, only export I2P and L2W
python andy/onnx_conversion.py --i2p_weights /path/to/i2p.pth --l2w_weights /path/to/l2w.pth --skip_multiview3d

# Options: --device cpu|cuda, --opset 17
python andy/onnx_conversion.py --i2p_weights /path/to/i2p.pth --l2w_weights /path/to/l2w.pth --device cpu --opset 17
```

Outputs (in `andy/onnx/` by default): `multiview3d_encoder.onnx` (if export succeeds), `image2points.onnx`, `local2world.onnx`.

---

## ONNX quantization (DROID-SLAM)

Quantize DROID-SLAM ONNX models (fnet, cnet, update_core) from `andy/onnx/`. Run from **project root**.

```bash
# Dynamic INT8 (default), output andy/onnx_dynamic_int8/
python andy/onnx_quantize.py

# Static INT8 (TensorRT-friendly), output andy/onnx_static_int8/
python andy/onnx_quantize.py --method static_int8

# FP16 (requires: pip install onnxconverter-common)
python andy/onnx_quantize.py --method fp16

# Custom input/output directories
python andy/onnx_quantize.py --input_dir andy/onnx --suffix my_quant
```

Use the output directory as `--onnx_dir` when running the demo with `--use_onnx`.

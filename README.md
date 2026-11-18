# MirrorCut model artifacts

Runtime-downloadable model packs for the MirrorCut / LibreCuts Android video editor.
The app fetches these files at first use and verifies the SHA-256 before installing.

## EdgeTAM fp32 ONNX (`models-v1`)

Image-predictor export of [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)
(Apache-2.0), produced by `tools/segmentation/export_edgetam_onnx.py` in the MirrorCut
repository with `--edgetam-repo` pointing at that project and
`--checkpoint checkpoints/edgetam.pt`. Export settings: fp32, ONNX opset 17,
`torch.onnx.export(..., dynamo=False)`, point count left dynamic, `image_pe` baked into the
decoder graph.

| Asset | Bytes | SHA-256 |
| --- | --- | --- |
| `edgetam-encoder-fp32.onnx` | 19,772,801 | `97104d7eb4df5055d90d983dbabf9b963b9ae2ccc7a20ed929282c744fdd9933` |
| `edgetam-decoder-fp32.onnx` | 21,253,421 | `a2e156784d0a6a2ae0281424712db20ae32f994160f66c95a8e9c9dc4c260e00` |

Graph interfaces:

- encoder: `image` float32 `[1,3,1024,1024]` (ImageNet-normalized RGB) →
  `image_embed` `[1,256,64,64]`, `high_res_feats_0` `[1,32,256,256]`,
  `high_res_feats_1` `[1,64,128,128]`, `vision_pos_enc` `[1,256,64,64]`.
- decoder: `image_embed`, both high-res features, `point_coords` float32 `[1,N,2]`,
  `point_labels` int32 `[1,N]` (1 include, 0 exclude, -1 pad), `mask_input`
  float32 `[1,1,256,256]`, `has_mask_input` float32 `[1]` →
  `low_res_masks` float32 `[1,3,256,256]` (logits), `iou_predictions` float32 `[1,3]`.

The export was parity-checked against PyTorch on the host: encoder ≤ 2.1e-5,
decoder ≤ 2.5e-5, and the dynamic point-count paths (N = 1, 5, 9) ≤ 5e-6 max absolute
difference.

## Usage

These files are release assets, not source files. Download the assets attached to a
release; do not download the repository archive. Replace artifacts only by publishing a
new tag (for example `models-v2`) so installed clients keep resolving a pinned version.

## License

Apache License 2.0, see [LICENSE](LICENSE). EdgeTAM is © Meta Platforms, Inc. and
affiliates; the checkpoint and derivative export retain the upstream license and
attribution.

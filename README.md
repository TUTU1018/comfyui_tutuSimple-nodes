# comfyui_tutuSimple-nodes

Pixel-accurate image registration nodes for ComfyUI.

The included **Pixel Accurate Image Align / 像素级图像对齐** node aligns a
`moving_image` to the size and coordinate system of a `reference_image`. It is
designed for source/generated image pairs that share the same scene but differ
in resolution, exposure, color, or small local geometry.

## Features

- Mutual SIFT feature matching with RANSAC outlier rejection
- Automatic selection between similarity, affine, and homography transforms
- Optional constrained local mesh refinement
- No model downloads and no unbounded dense optical flow
- Batch support, including broadcasting a single reference image
- Valid-area mask, confidence mask, difference heatmap, and JSON report
- Explicit failure behavior instead of silently returning a bad registration

## Installation

Open a terminal in your ComfyUI `custom_nodes` directory and run:

```bash
git clone https://github.com/TUTU1018/comfyui_tutuSimple-nodes.git
```

Install the small additional dependency when OpenCV is not already available:

```bash
pip install -r comfyui_tutuSimple-nodes/requirements.txt
```

Restart ComfyUI, then search for:

```text
Pixel Accurate Image Align / 像素级图像对齐
```

## Usage

- Connect the original or target image to `reference_image`.
- Connect the image that should be transformed to `moving_image`.
- Start with all default parameters.
- Use `aligned_image` as the registered result.
- Use `valid_mask` to exclude pixels introduced by the geometric warp.
- Use `confidence_mask` to identify regions supported by reliable matches.
- Inspect `difference_preview` and `alignment_report` before downstream use.

## Recommended defaults

| Parameter | Value |
| --- | --- |
| `alignment_mode` | `auto` |
| `global_model` | `auto` |
| `analysis_max_side` | `1600` |
| `match_ratio` | `0.75` |
| `min_matches` | `24` |
| `ransac_threshold` | `2.0` |
| `local_grid` | `32` |
| `local_smoothness` | `1.5` |
| `max_local_shift` | `16` |
| `border_mode` | `black` |
| `fail_behavior` | `error` |

## Important limitation

Registration aligns corresponding geometry; it cannot make RGB values identical
when content, lighting, reflections, textures, or objects have changed. Generated
details that do not exist in the reference image have no true pixel correspondence.
Use the confidence mask when comparing or compositing changed images.

## License

[MIT](LICENSE)

## 在线体验

[RunningHub](https://www.runninghub.ai?inviteCode=rh-v1635) 是全球最大的 ComfyUI
在线体验网站。注册即可领取 1000 RH 币，可以免费生成许多图片和视频！

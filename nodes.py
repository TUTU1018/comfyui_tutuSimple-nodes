import json
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def _as_rgb_u8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected one HWC image, got shape {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 3:
        array = array[:, :, :3]
    else:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {array.shape[2]}")
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def _analysis_image(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    new_width = max(32, int(round(width * scale)))
    new_height = max(32, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray, new_width / width, new_height / height


def _ratio_matches(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float) -> List[cv2.DMatch]:
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    return [first for first, second in pairs if first.distance < ratio * second.distance]


def _mutual_sift_matches(
    moving: np.ndarray,
    reference: np.ndarray,
    analysis_max_side: int,
    match_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    moving_gray, moving_sx, moving_sy = _analysis_image(moving, analysis_max_side)
    reference_gray, reference_sx, reference_sy = _analysis_image(reference, analysis_max_side)

    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.015, edgeThreshold=15)
    moving_keys, moving_desc = sift.detectAndCompute(moving_gray, None)
    reference_keys, reference_desc = sift.detectAndCompute(reference_gray, None)
    if moving_desc is None or reference_desc is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), len(moving_keys), len(reference_keys)

    forward = _ratio_matches(moving_desc, reference_desc, match_ratio)
    reverse = _ratio_matches(reference_desc, moving_desc, match_ratio)
    reverse_lookup = {match.queryIdx: match.trainIdx for match in reverse}
    mutual = [
        match for match in forward
        if reverse_lookup.get(match.trainIdx) == match.queryIdx
    ]

    source = np.float32([
        (moving_keys[match.queryIdx].pt[0] / moving_sx, moving_keys[match.queryIdx].pt[1] / moving_sy)
        for match in mutual
    ])
    target = np.float32([
        (reference_keys[match.trainIdx].pt[0] / reference_sx, reference_keys[match.trainIdx].pt[1] / reference_sy)
        for match in mutual
    ])
    return source, target, len(moving_keys), len(reference_keys)


def _to_homography(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(np.float64)
    result = np.eye(3, dtype=np.float64)
    result[:2, :] = matrix
    return result


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points[:, None, :].astype(np.float32), matrix.astype(np.float64))[:, 0, :]


def _grid_coverage(points: np.ndarray, width: int, height: int, cells: int = 4) -> float:
    if len(points) == 0:
        return 0.0
    xs = np.clip((points[:, 0] / max(width, 1) * cells).astype(np.int32), 0, cells - 1)
    ys = np.clip((points[:, 1] / max(height, 1) * cells).astype(np.int32), 0, cells - 1)
    occupied = len(set(zip(xs.tolist(), ys.tolist())))
    return occupied / float(cells * cells)


def _candidate_metrics(
    name: str,
    matrix: Optional[np.ndarray],
    inlier_mask: Optional[np.ndarray],
    source: np.ndarray,
    target: np.ndarray,
    reference_width: int,
    reference_height: int,
) -> Optional[Dict]:
    if matrix is None or inlier_mask is None:
        return None
    homography = _to_homography(matrix)
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    if not inliers.any():
        return None
    predicted = _project(source, homography)
    errors = np.linalg.norm(predicted - target, axis=1)
    valid_errors = errors[inliers]
    return {
        "name": name,
        "matrix": homography,
        "inlier_mask": inliers,
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "median_error": float(np.median(valid_errors)),
        "p95_error": float(np.percentile(valid_errors, 95)),
        "coverage": _grid_coverage(target[inliers], reference_width, reference_height),
    }


def _estimate_global_transform(
    source: np.ndarray,
    target: np.ndarray,
    model: str,
    ransac_threshold: float,
    min_matches: int,
    reference_width: int,
    reference_height: int,
) -> Dict:
    if len(source) < min_matches:
        raise ValueError(f"Only {len(source)} mutual SIFT matches; at least {min_matches} are required")

    candidates: List[Dict] = []
    if model in ("auto", "similarity"):
        matrix, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=10000, confidence=0.999, refineIters=50,
        )
        item = _candidate_metrics("similarity", matrix, mask, source, target, reference_width, reference_height)
        if item:
            candidates.append(item)

    if model in ("auto", "affine"):
        matrix, mask = cv2.estimateAffine2D(
            source, target, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=10000, confidence=0.999, refineIters=50,
        )
        item = _candidate_metrics("affine", matrix, mask, source, target, reference_width, reference_height)
        if item:
            candidates.append(item)

    if model in ("auto", "homography"):
        matrix, mask = cv2.findHomography(
            source, target, cv2.RANSAC, ransac_threshold,
            maxIters=10000, confidence=0.999,
        )
        item = _candidate_metrics("homography", matrix, mask, source, target, reference_width, reference_height)
        if item:
            candidates.append(item)

    if not candidates:
        raise ValueError(f"Could not estimate a valid {model} transform")

    required_inliers = max(8, min_matches // 2)
    candidates = [item for item in candidates if item["inliers"] >= required_inliers]
    if not candidates:
        raise ValueError(f"No transform had at least {required_inliers} RANSAC inliers")

    if model != "auto":
        return candidates[0]

    complexity_order = {"similarity": 0, "affine": 1, "homography": 2}
    best = max(candidates, key=lambda item: (item["inliers"], -item["median_error"]))
    for item in sorted(candidates, key=lambda value: complexity_order[value["name"]]):
        enough_inliers = item["inliers"] >= max(required_inliers, math.floor(best["inliers"] * 0.82))
        acceptable_error = item["p95_error"] <= best["p95_error"] * 1.35 + ransac_threshold * 0.2
        acceptable_coverage = item["coverage"] >= max(0.125, best["coverage"] - 0.20)
        if enough_inliers and acceptable_error and acceptable_coverage:
            return item
    return best


def _warp_global(
    image: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
    interpolation: int,
    border_mode: int,
) -> np.ndarray:
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=interpolation,
        borderMode=border_mode, borderValue=0,
    )


def _splat_local_grid(
    positions: np.ndarray,
    residuals: np.ndarray,
    width: int,
    height: int,
    grid_long_side: int,
    smoothness: float,
    max_shift: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if width >= height:
        grid_width = grid_long_side + 1
        grid_height = max(5, int(round(grid_long_side * height / width)) + 1)
    else:
        grid_height = grid_long_side + 1
        grid_width = max(5, int(round(grid_long_side * width / height)) + 1)

    numerator_x = np.zeros((grid_height, grid_width), np.float32)
    numerator_y = np.zeros((grid_height, grid_width), np.float32)
    weights = np.zeros((grid_height, grid_width), np.float32)

    grid_x = positions[:, 0] * (grid_width - 1) / max(width - 1, 1)
    grid_y = positions[:, 1] * (grid_height - 1) / max(height - 1, 1)
    for x, y, residual in zip(grid_x, grid_y, residuals):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        for yy in (y0, min(y0 + 1, grid_height - 1)):
            for xx in (x0, min(x0 + 1, grid_width - 1)):
                weight = max(0.0, 1.0 - abs(x - xx)) * max(0.0, 1.0 - abs(y - yy))
                numerator_x[yy, xx] += residual[0] * weight
                numerator_y[yy, xx] += residual[1] * weight
                weights[yy, xx] += weight

    sigma = max(0.25, float(smoothness))
    blur_options = {
        "ksize": (0, 0),
        "sigmaX": sigma,
        "sigmaY": sigma,
        "borderType": cv2.BORDER_REPLICATE,
    }
    smooth_weights = cv2.GaussianBlur(weights, **blur_options)
    smooth_x = cv2.GaussianBlur(numerator_x, **blur_options)
    smooth_y = cv2.GaussianBlur(numerator_y, **blur_options)
    zero_prior = 0.035
    displacement_x = smooth_x / (smooth_weights + zero_prior)
    displacement_y = smooth_y / (smooth_weights + zero_prior)
    magnitude = np.sqrt(displacement_x ** 2 + displacement_y ** 2)
    limiter = np.minimum(1.0, max_shift / np.maximum(magnitude, 1e-6))
    displacement_x *= limiter
    displacement_y *= limiter
    confidence = np.clip(1.0 - np.exp(-smooth_weights * 3.0), 0.0, 1.0).astype(np.float32)
    return displacement_x.astype(np.float32), displacement_y.astype(np.float32), confidence


def _grid_stripe(grid: np.ndarray, width: int, height: int, y0: int, y1: int) -> np.ndarray:
    grid_height, grid_width = grid.shape
    xs = ((np.arange(width, dtype=np.float32) + 0.5) * grid_width / width - 0.5)[None, :]
    ys = ((np.arange(y0, y1, dtype=np.float32) + 0.5) * grid_height / height - 0.5)[:, None]
    map_x = np.broadcast_to(xs, (y1 - y0, width)).copy()
    map_y = np.broadcast_to(ys, (y1 - y0, width)).copy()
    return cv2.remap(grid, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _local_warp_striped(
    image: np.ndarray,
    valid: np.ndarray,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    confidence_grid: np.ndarray,
    border_mode: int,
    stripe_height: int = 384,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    aligned = np.empty_like(image)
    aligned_valid = np.empty((height, width), np.float32)
    confidence = np.empty((height, width), np.float32)
    x_coordinates = np.arange(width, dtype=np.float32)[None, :]

    for y0 in range(0, height, stripe_height):
        y1 = min(height, y0 + stripe_height)
        dx = _grid_stripe(displacement_x, width, height, y0, y1)
        dy = _grid_stripe(displacement_y, width, height, y0, y1)
        map_x = np.broadcast_to(x_coordinates, (y1 - y0, width)).copy() - dx
        map_y = np.broadcast_to(np.arange(y0, y1, dtype=np.float32)[:, None], (y1 - y0, width)).copy() - dy
        aligned[y0:y1] = cv2.remap(
            image, map_x, map_y, cv2.INTER_LANCZOS4,
            borderMode=border_mode, borderValue=0,
        )
        aligned_valid[y0:y1] = cv2.remap(
            valid, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        confidence[y0:y1] = _grid_stripe(confidence_grid, width, height, y0, y1)

    return aligned, np.clip(aligned_valid, 0.0, 1.0), np.clip(confidence, 0.0, 1.0)


def _difference_heatmap(reference: np.ndarray, aligned: np.ndarray, valid: np.ndarray) -> np.ndarray:
    difference = np.mean(np.abs(reference.astype(np.float32) - aligned.astype(np.float32)), axis=2) / 255.0
    difference *= valid
    visible = np.clip(difference * 4.0, 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.rint(visible * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def _border_constant(name: str) -> int:
    return {
        "black": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
    }[name]


def align_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    alignment_mode: str,
    global_model: str,
    analysis_max_side: int,
    match_ratio: float,
    min_matches: int,
    ransac_threshold: float,
    local_grid: int,
    local_smoothness: float,
    max_local_shift: float,
    border_mode_name: str,
    fail_behavior: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    reference_height, reference_width = reference.shape[:2]
    border_mode = _border_constant(border_mode_name)
    source, target, source_keypoints, target_keypoints = _mutual_sift_matches(
        moving, reference, analysis_max_side, match_ratio,
    )
    reference_scale = max(reference_height, reference_width) / float(analysis_max_side)
    full_ransac_threshold = ransac_threshold * max(1.0, reference_scale)

    try:
        selected = _estimate_global_transform(
            source, target, global_model, full_ransac_threshold, min_matches,
            reference_width, reference_height,
        )
        if selected["coverage"] < 0.125:
            raise ValueError(f"RANSAC inliers cover only {selected['coverage']:.1%} of the image grid")
    except ValueError as error:
        if fail_behavior == "error":
            raise
        matrix = np.array([
            [reference_width / moving.shape[1], 0.0, 0.0],
            [0.0, reference_height / moving.shape[0], 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        selected = {
            "name": "resize_only",
            "matrix": matrix,
            "inlier_mask": np.zeros(len(source), dtype=bool),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_error": None,
            "p95_error": None,
            "coverage": 0.0,
            "fallback_reason": str(error),
        }

    matrix = selected["matrix"]
    global_aligned = _warp_global(
        moving, matrix, reference_width, reference_height,
        cv2.INTER_LANCZOS4, border_mode,
    )
    moving_valid = np.ones(moving.shape[:2], np.float32)
    global_valid = _warp_global(
        moving_valid, matrix, reference_width, reference_height,
        cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
    )

    aligned = global_aligned
    valid = np.clip(global_valid, 0.0, 1.0)
    confidence = np.zeros((reference_height, reference_width), np.float32)
    local_used = False
    local_controls = 0
    local_median_shift = 0.0
    local_p95_shift = 0.0
    local_reason = "disabled"

    can_refine = alignment_mode in ("auto", "global_local") and selected["inliers"] > 0
    if can_refine:
        inliers = selected["inlier_mask"]
        predicted = _project(source, matrix)
        residuals = target - predicted
        magnitudes = np.linalg.norm(residuals, axis=1)
        inlier_magnitudes = magnitudes[inliers]
        median = float(np.median(inlier_magnitudes))
        mad = float(np.median(np.abs(inlier_magnitudes - median)))
        robust_limit = min(max_local_shift, median + 3.5 * max(mad, 0.5) + 1.0)
        controls = inliers & np.isfinite(magnitudes) & (magnitudes <= robust_limit)
        local_controls = int(controls.sum())
        control_coverage = _grid_coverage(target[controls], reference_width, reference_height)
        if local_controls >= max(10, min_matches // 2) and control_coverage >= 0.125:
            dx_grid, dy_grid, confidence_grid = _splat_local_grid(
                target[controls], residuals[controls], reference_width, reference_height,
                local_grid, local_smoothness, max_local_shift,
            )
            aligned, valid, confidence = _local_warp_striped(
                global_aligned, global_valid, dx_grid, dy_grid,
                confidence_grid, border_mode,
            )
            local_used = True
            local_reason = "applied"
            local_median_shift = float(np.median(magnitudes[controls]))
            local_p95_shift = float(np.percentile(magnitudes[controls], 95))
        else:
            local_reason = f"insufficient controls or coverage ({local_controls}, {control_coverage:.1%})"

    if not local_used and selected["inliers"] > 0:
        inlier_targets = target[selected["inlier_mask"]]
        _, _, confidence_grid = _splat_local_grid(
            inlier_targets, np.zeros_like(inlier_targets), reference_width, reference_height,
            local_grid, local_smoothness, max_local_shift,
        )
        for y0 in range(0, reference_height, 384):
            y1 = min(reference_height, y0 + 384)
            confidence[y0:y1] = _grid_stripe(confidence_grid, reference_width, reference_height, y0, y1)

    confidence *= valid
    difference = _difference_heatmap(reference, aligned, valid)
    report = {
        "reference_size": [reference_width, reference_height],
        "moving_size": [moving.shape[1], moving.shape[0]],
        "source_keypoints": source_keypoints,
        "reference_keypoints": target_keypoints,
        "mutual_matches": int(len(source)),
        "global_model": selected["name"],
        "global_inliers": selected["inliers"],
        "global_inlier_ratio": round(selected["inlier_ratio"], 5),
        "global_coverage": round(selected["coverage"], 5),
        "global_median_error_px": None if selected["median_error"] is None else round(selected["median_error"], 4),
        "global_p95_error_px": None if selected["p95_error"] is None else round(selected["p95_error"], 4),
        "local_refinement": local_used,
        "local_status": local_reason,
        "local_controls": local_controls,
        "local_median_shift_px": round(local_median_shift, 4),
        "local_p95_shift_px": round(local_p95_shift, 4),
        "valid_fraction": round(float((valid > 0.999).mean()), 5),
        "transform_moving_to_reference": np.round(matrix, 9).tolist(),
    }
    if "fallback_reason" in selected:
        report["fallback_reason"] = selected["fallback_reason"]
    return aligned, valid, confidence, difference, report


class PixelAccurateImageAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "moving_image": ("IMAGE",),
                "alignment_mode": (["auto", "global_local", "global_only"], {"default": "auto"}),
                "global_model": (["auto", "similarity", "affine", "homography"], {"default": "auto"}),
                "analysis_max_side": ("INT", {"default": 1600, "min": 512, "max": 3072, "step": 64}),
                "match_ratio": ("FLOAT", {"default": 0.75, "min": 0.55, "max": 0.90, "step": 0.01}),
                "min_matches": ("INT", {"default": 24, "min": 8, "max": 200, "step": 1}),
                "ransac_threshold": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.1}),
                "local_grid": ("INT", {"default": 32, "min": 8, "max": 64, "step": 2}),
                "local_smoothness": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 5.0, "step": 0.05}),
                "max_local_shift": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 128.0, "step": 1.0}),
                "border_mode": (["black", "reflect", "replicate"], {"default": "black"}),
                "fail_behavior": (["error", "resize_only"], {"default": "error"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("aligned_image", "valid_mask", "confidence_mask", "difference_preview", "alignment_report")
    FUNCTION = "align"
    CATEGORY = "image/alignment"
    DESCRIPTION = "Aligns a moving image to a reference with robust global registration and optional constrained local refinement."

    def align(
        self,
        reference_image: torch.Tensor,
        moving_image: torch.Tensor,
        alignment_mode: str,
        global_model: str,
        analysis_max_side: int,
        match_ratio: float,
        min_matches: int,
        ransac_threshold: float,
        local_grid: int,
        local_smoothness: float,
        max_local_shift: float,
        border_mode: str,
        fail_behavior: str,
    ):
        reference_batch = int(reference_image.shape[0])
        moving_batch = int(moving_image.shape[0])
        if reference_batch != moving_batch and reference_batch != 1 and moving_batch != 1:
            raise ValueError(
                f"Batch sizes must match or one batch must contain one image; got {reference_batch} and {moving_batch}"
            )
        batch_size = max(reference_batch, moving_batch)
        aligned_images = []
        valid_masks = []
        confidence_masks = []
        difference_images = []
        reports = []

        for index in range(batch_size):
            reference = _as_rgb_u8(reference_image[min(index, reference_batch - 1)])
            moving = _as_rgb_u8(moving_image[min(index, moving_batch - 1)])
            aligned, valid, confidence, difference, report = align_pair(
                reference, moving, alignment_mode, global_model,
                analysis_max_side, match_ratio, min_matches, ransac_threshold,
                local_grid, local_smoothness, max_local_shift,
                border_mode, fail_behavior,
            )
            report["batch_index"] = index
            aligned_images.append(torch.from_numpy(aligned.astype(np.float32) / 255.0))
            valid_masks.append(torch.from_numpy(valid.astype(np.float32)))
            confidence_masks.append(torch.from_numpy(confidence.astype(np.float32)))
            difference_images.append(torch.from_numpy(difference.astype(np.float32) / 255.0))
            reports.append(report)

        return (
            torch.stack(aligned_images),
            torch.stack(valid_masks),
            torch.stack(confidence_masks),
            torch.stack(difference_images),
            json.dumps(reports, ensure_ascii=False, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "PixelAccurateImageAlign": PixelAccurateImageAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelAccurateImageAlign": "Pixel Accurate Image Align / 像素级图像对齐",
}

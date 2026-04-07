from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, median_filter


@dataclass
class ProcessingConfig:
    dx_mm: float = 0.08
    dy_mm: float = 0.005615
    dz_mm: float = 0.0001
    stripe_height: int = 2048
    num_stripes: int = 3
    downsample_factor: int = 16
    seam_window: int = 64
    transition_window: int = 0
    smooth_sigma_x: float = 2.0
    seam_flatten_half_window: int = 2
    seam_flatten_sigma_x: float = 0.0
    seam_flatten_blend_width: int = 0
    seam_flatten_method: str = "linear"
    pre_smooth_x_sigma: float = 0.0
    gaussian_sigma: float = 0.064
    outlier_filter_mode: str = "off"
    outlier_window_size: int = 0
    outlier_threshold_mm: float = 0.0
    outlier_max_cluster_size: int = 0
    outlier_replace_mode: str = "median"
    crop_left_px: int = 0
    crop_right_px: int = 0

    @property
    def expected_height(self) -> int:
        return self.stripe_height * self.num_stripes


def load_height_png(path: Path) -> np.ndarray:
    image = Image.open(str(path))
    array = np.array(image)

    if array.ndim != 2:
        raise ValueError("Input image must be a single-channel 16-bit PNG.")

    if array.dtype == np.uint16:
        return array

    if array.dtype.kind in ("i", "u") and array.min() >= 0 and array.max() <= 65535:
        return array.astype(np.uint16)

    raise ValueError("Input image must be a 16-bit grayscale PNG.")


def save_height_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(array, dtype=np.uint16), mode="I;16")
    image.save(str(path))


def save_float_tiff(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    image.save(str(path))


def generate_synthetic_heightmap(
    width: int = 2048,
    stripe_height: int = 2048,
    num_stripes: int = 3,
    dz_mm: float = 0.0001,
    outlier_spike_count: int = 0,
    outlier_cluster_count: int = 0,
    outlier_cluster_size: int = 2,
    outlier_amplitude_mm: float = 0.06,
    outlier_seed: int = 17,
) -> np.ndarray:
    total_height = stripe_height * num_stripes
    x_mm = np.linspace(-6.0, 6.0, width, dtype=np.float64)
    y_mm = np.linspace(-4.0, 4.0, total_height, dtype=np.float64)
    xx, yy = np.meshgrid(x_mm, y_mm)

    base = 0.35 * np.sin(xx * 0.7) + 0.22 * np.cos(yy * 0.9)
    saddle = 0.012 * xx * yy
    bump = 0.55 * np.exp(-((xx + 1.8) ** 2 + (yy - 1.0) ** 2) / 2.8)
    pit = -0.48 * np.exp(-((xx - 2.1) ** 2 + (yy + 1.5) ** 2) / 1.7)
    ripple = 0.03 * np.sin(xx * 5.2 + yy * 2.6)

    surface_mm = base + saddle + bump + pit + ripple

    phase = np.linspace(0.0, 2.0 * np.pi, width, dtype=np.float64)
    delta1 = 0.55 + 0.10 * np.sin(phase) + 0.04 * np.linspace(-1.0, 1.0, width)
    delta2 = -0.42 + 0.06 * np.cos(phase * 0.7)

    surface_mm[stripe_height:stripe_height * 2] += delta1
    surface_mm[stripe_height * 2:] += delta1 + delta2

    # Add a narrow seam-local zigzag artifact instead of extending the zigzag
    # across the whole scan band. This better matches stitched real data where
    # the residual only appears on the seam line itself.
    seam_phase = np.arange(width, dtype=np.float64) / 6.0
    seam_frac = seam_phase - np.floor(seam_phase)
    seam_triangle = 2.0 * np.abs(2.0 * seam_frac - 1.0) - 1.0
    for seam_y, amplitude in ((stripe_height, 0.08), (stripe_height * 2, -0.07)):
        for row in range(seam_y - 3, seam_y + 4):
            if 0 <= row < total_height:
                weight = 1.0 - 0.18 * abs(row - seam_y)
                surface_mm[row, :] += amplitude * seam_triangle * weight

    rng = np.random.RandomState(7)
    surface_mm += rng.normal(scale=0.002, size=surface_mm.shape)

    if outlier_spike_count < 0 or outlier_cluster_count < 0:
        raise ValueError("Outlier counts must be non-negative.")
    if outlier_cluster_size <= 0:
        raise ValueError("outlier_cluster_size must be positive.")
    if outlier_amplitude_mm < 0:
        raise ValueError("outlier_amplitude_mm must be non-negative.")

    if outlier_spike_count > 0 or outlier_cluster_count > 0:
        outlier_rng = np.random.RandomState(outlier_seed)

        for _ in range(outlier_spike_count):
            row = int(outlier_rng.randint(0, total_height))
            col = int(outlier_rng.randint(0, width))
            sign = -1.0 if outlier_rng.rand() < 0.5 else 1.0
            scale = 0.8 + 0.4 * outlier_rng.rand()
            surface_mm[row, col] += sign * outlier_amplitude_mm * scale

        cluster_span = min(outlier_cluster_size, total_height, width)
        for _ in range(outlier_cluster_count):
            row = int(outlier_rng.randint(0, total_height - cluster_span + 1))
            col = int(outlier_rng.randint(0, width - cluster_span + 1))
            sign = -1.0 if outlier_rng.rand() < 0.5 else 1.0
            scale = 0.8 + 0.4 * outlier_rng.rand()
            surface_mm[row:row + cluster_span, col:col + cluster_span] += sign * outlier_amplitude_mm * scale

    surface_mm = surface_mm - surface_mm.min() + 0.25
    gray = np.rint(surface_mm / dz_mm)
    gray = np.clip(gray, 0, np.iinfo(np.uint16).max)
    return gray.astype(np.uint16)


_OUTLIER_FILTER_DEFAULTS = {
    "off": {"window_size": 0, "threshold_mm": 0.0, "max_cluster_size": 0},
    "conservative": {"window_size": 3, "threshold_mm": 0.02, "max_cluster_size": 1},
    "medium": {"window_size": 5, "threshold_mm": 0.01, "max_cluster_size": 4},
}


def _resolve_outlier_filter_params(
    mode: str,
    window_size: int,
    threshold_mm: float,
    max_cluster_size: int,
    replace_mode: str,
) -> Tuple[str, int, float, int, str]:
    normalized_mode = mode.lower()
    if normalized_mode not in _OUTLIER_FILTER_DEFAULTS:
        raise ValueError("Unsupported outlier filter mode: {}".format(mode))
    if replace_mode != "median":
        raise ValueError("Unsupported outlier replace mode: {}".format(replace_mode))
    if normalized_mode == "off":
        return normalized_mode, 0, 0.0, 0, replace_mode

    defaults = _OUTLIER_FILTER_DEFAULTS[normalized_mode]
    resolved_window_size = window_size if window_size > 0 else defaults["window_size"]
    resolved_threshold_mm = threshold_mm if threshold_mm > 0 else defaults["threshold_mm"]
    resolved_max_cluster_size = max_cluster_size if max_cluster_size > 0 else defaults["max_cluster_size"]

    if resolved_window_size <= 1:
        raise ValueError("outlier_window_size must be greater than 1 when outlier filtering is enabled.")
    if resolved_window_size % 2 == 0:
        raise ValueError("outlier_window_size must be an odd integer.")
    if resolved_threshold_mm <= 0:
        raise ValueError("outlier_threshold_mm must be positive when outlier filtering is enabled.")
    if resolved_max_cluster_size <= 0:
        raise ValueError("outlier_max_cluster_size must be positive when outlier filtering is enabled.")

    return (
        normalized_mode,
        resolved_window_size,
        resolved_threshold_mm,
        resolved_max_cluster_size,
        replace_mode,
    )


def filter_height_outliers(
    z_mm: np.ndarray,
    mode: str = "off",
    window_size: int = 0,
    threshold_mm: float = 0.0,
    max_cluster_size: int = 0,
    replace_mode: str = "median",
) -> np.ndarray:
    if z_mm.ndim != 2:
        raise ValueError("Height map must be a 2D array.")

    resolved_mode, resolved_window_size, resolved_threshold_mm, resolved_max_cluster_size, _ = (
        _resolve_outlier_filter_params(mode, window_size, threshold_mm, max_cluster_size, replace_mode)
    )
    if resolved_mode == "off":
        return z_mm.astype(np.float64, copy=True)

    local_median = median_filter(z_mm, size=(resolved_window_size, resolved_window_size), mode="nearest")
    residual = np.abs(z_mm - local_median)
    candidate_mask = residual >= resolved_threshold_mm
    if not np.any(candidate_mask):
        return z_mm.astype(np.float64, copy=True)

    # 组件面积使用“强离群核心”而不是整块候选区域来衡量，避免小团簇
    # 周围被阈值带出的弱响应像素把面积虚增，导致 2x2 一类的小异常
    # 无法按 medium 档被识别出来。
    core_mask = residual >= (resolved_threshold_mm * 1.5)
    structure = np.ones((3, 3), dtype=np.int8)
    labeled_mask, num_labels = label(candidate_mask, structure=structure)
    replace_mask = np.zeros_like(candidate_mask, dtype=bool)
    for component_id in range(1, num_labels + 1):
        component = labeled_mask == component_id
        core_size = int(np.count_nonzero(component & core_mask))
        if 0 < core_size <= resolved_max_cluster_size:
            replace_mask |= component

    filtered = z_mm.astype(np.float64, copy=True)
    filtered[replace_mask] = local_median[replace_mask]
    return filtered


def _project_window_to_target(rows: np.ndarray, target_row: float) -> np.ndarray:
    row_axis = rows[:, 0]
    values = rows[:, 1:]
    row_mean = row_axis.mean()
    centered_rows = row_axis - row_mean
    denom = np.sum(centered_rows * centered_rows)
    if denom <= 0:
        return np.median(values, axis=0)

    value_mean = values.mean(axis=0)
    slope = np.sum(centered_rows[:, np.newaxis] * (values - value_mean), axis=0) / denom
    return value_mean + slope * (target_row - row_mean)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _estimate_row_slope(image: np.ndarray, row: int) -> np.ndarray:
    if image.shape[0] == 1:
        return np.zeros(image.shape[1], dtype=np.float64)
    if row <= 0:
        return image[1, :] - image[0, :]
    if row >= image.shape[0] - 1:
        return image[-1, :] - image[-2, :]
    return 0.5 * (image[row + 1, :] - image[row - 1, :])


def _estimate_window_slope(image: np.ndarray, row_start: int, row_end: int) -> np.ndarray:
    row_start = max(0, row_start)
    row_end = min(image.shape[0] - 1, row_end)
    if row_end <= row_start:
        return np.zeros(image.shape[1], dtype=np.float64)

    rows = np.arange(row_start, row_end + 1, dtype=np.float64)
    values = image[row_start:row_end + 1, :].astype(np.float64)
    row_mean = rows.mean()
    centered_rows = rows - row_mean
    denom = np.sum(centered_rows * centered_rows)
    if denom <= 0:
        return np.zeros(image.shape[1], dtype=np.float64)

    value_mean = values.mean(axis=0)
    return np.sum(centered_rows[:, np.newaxis] * (values - value_mean), axis=0) / denom


def correct_scan_band_offsets(
    z_mm: np.ndarray,
    stripe_height: int,
    seam_window: int,
    transition_window: int,
    smooth_sigma_x: float,
    seam_flatten_half_window: int = 2,
    seam_flatten_sigma_x: float = 0.0,
    seam_flatten_blend_width: int = 0,
    seam_flatten_method: str = "linear",
    num_stripes: int | None = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if z_mm.ndim != 2:
        raise ValueError("Height map must be a 2D array.")
    if stripe_height <= 0:
        raise ValueError("stripe_height must be positive.")
    if z_mm.shape[0] % stripe_height != 0:
        raise ValueError("Image height must be divisible by stripe_height.")

    inferred_stripes = z_mm.shape[0] // stripe_height
    if num_stripes is None:
        num_stripes = inferred_stripes
    if num_stripes != inferred_stripes:
        raise ValueError("Configured stripe count does not match input height.")

    width = z_mm.shape[1]
    zero_offset = np.zeros(width, dtype=np.float64)
    cumulative_offsets = [zero_offset]
    seam_offsets = []
    running = zero_offset.copy()

    for stripe_index in range(1, num_stripes):
        seam_y = stripe_index * stripe_height
        above_y0 = max(0, seam_y - seam_window)
        below_y1 = min(z_mm.shape[0], seam_y + seam_window)
        above = z_mm[above_y0:seam_y, :]
        below = z_mm[seam_y:below_y1, :]
        if above.size == 0 or below.size == 0:
            raise ValueError("Seam window exceeds image bounds.")

        # 这里不是给整条接缝只算一个偏移量，而是对每个 X 列单独估计
        # 一次，最终得到长度等于图像宽度的 Δ(x)。
        target_row = seam_y - 0.5
        above_rows = np.column_stack(
            [np.arange(above_y0, seam_y, dtype=np.float64), above.astype(np.float64)]
        )
        below_rows = np.column_stack(
            [np.arange(seam_y, below_y1, dtype=np.float64), below.astype(np.float64)]
        )
        delta = _project_window_to_target(below_rows, target_row) - _project_window_to_target(
            above_rows, target_row
        )
        # 只做轻微的 X 向平滑。若平滑过强，会把真实的列间偏移差异抹掉，
        # 反而在接缝处留下残差，并在 grad_y / curv_y 中被放大。
        if smooth_sigma_x > 0:
            delta = gaussian_filter1d(delta, sigma=smooth_sigma_x, mode="nearest")
        seam_offsets.append(delta)
        running = running + delta
        cumulative_offsets.append(running.copy())

    correction = np.zeros_like(z_mm, dtype=np.float64)
    for stripe_index, stripe_offset in enumerate(cumulative_offsets):
        start = stripe_index * stripe_height
        end = start + stripe_height
        correction[start:end, :] = stripe_offset

    # 整带补偿：第 2 道减 Δ1(x)，第 3 道减 Δ1(x)+Δ2(x)。
    corrected = z_mm - correction
    if seam_flatten_half_window > 0:
        # 接缝窄带重建发生在重采样前，目的是在原始高分辨率网格上先去掉
        # seam-local 的锯齿残差，再做 Y 向分块采样。
        corrected = flatten_seam_artifacts(
            corrected,
            stripe_height=stripe_height,
            num_stripes=num_stripes,
            half_window=seam_flatten_half_window,
            sigma_x=seam_flatten_sigma_x,
            blend_width=seam_flatten_blend_width,
            method=seam_flatten_method,
        )
    return corrected, seam_offsets


def flatten_seam_artifacts(
    z_mm: np.ndarray,
    stripe_height: int,
    num_stripes: int,
    half_window: int = 2,
    sigma_x: float = 0.0,
    blend_width: int = 0,
    method: str = "linear",
) -> np.ndarray:
    if half_window <= 0:
        return z_mm.copy()
    if method not in ("linear", "quadratic", "cubic"):
        raise ValueError("Unsupported seam flatten method: {}".format(method))

    flattened = z_mm.copy()
    for stripe_index in range(1, num_stripes):
        seam_y = stripe_index * stripe_height
        core_start = max(1, seam_y - half_window)
        core_end = min(flattened.shape[0] - 1, seam_y + half_window)
        band_start = max(0, core_start - blend_width)
        band_end = min(flattened.shape[0], core_end + blend_width)
        top_anchor = max(0, band_start - 1)
        bottom_anchor = min(flattened.shape[0] - 1, band_end)
        if core_end <= core_start or bottom_anchor <= top_anchor:
            continue

        top_line = flattened[top_anchor, :].copy()
        bottom_line = flattened[bottom_anchor, :].copy()
        slope_window = max(4, half_window + 4)
        top_slope = _estimate_window_slope(flattened, top_anchor - slope_window + 1, top_anchor)
        bottom_slope = _estimate_window_slope(flattened, bottom_anchor, bottom_anchor + slope_window - 1)
        if sigma_x > 0:
            top_line = gaussian_filter1d(top_line, sigma=sigma_x, mode="nearest")
            bottom_line = gaussian_filter1d(bottom_line, sigma=sigma_x, mode="nearest")
            top_slope = gaussian_filter1d(top_slope, sigma=sigma_x, mode="nearest")
            bottom_slope = gaussian_filter1d(bottom_slope, sigma=sigma_x, mode="nearest")

        # 核心带使用 Hermite 三次插值，匹配上下边界的高度和斜率。
        # 这样不仅高度连续，一阶导数也更连续，能明显减轻 half_window
        # 边界处再次出现的过渡感。
        span = float(bottom_anchor - top_anchor)
        original_band = flattened[band_start:band_end, :].copy()

        rebuilt_rows = []
        for row in range(band_start, band_end):
            t = (row - top_anchor) / span
            if method == "linear":
                rebuilt = top_line * (1.0 - t) + bottom_line * t
            elif method == "quadratic":
                # quadratic 模式要求每个 X 列都保持常二阶导，并让
                # 斜率沿 Y 方向做均匀变化。这里先由上下锚点高度确定
                # 割线斜率，再用上下边界斜率差来决定“从低到高”的
                # 线性变化幅度。这样既保持高度约束，又不会把原本
                # 低梯度的一侧抬高、高梯度的一侧压低到反转。
                effective_sigma_x = max(float(sigma_x), 2.0)
                secant_slope = (bottom_line - top_line) / span
                secant_slope = gaussian_filter1d(secant_slope, sigma=effective_sigma_x, mode="nearest")
                slope_delta = bottom_slope - top_slope
                slope_delta = gaussian_filter1d(slope_delta, sigma=effective_sigma_x, mode="nearest")
                top_rebuilt_slope = secant_slope - 0.5 * slope_delta
                a = 0.5 * slope_delta / span
                delta = row - top_anchor
                rebuilt = a * delta * delta + top_rebuilt_slope * delta + top_line
            else:
                h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0
                h10 = t * t * t - 2.0 * t * t + t
                h01 = -2.0 * t * t * t + 3.0 * t * t
                h11 = t * t * t - t * t
                rebuilt = (
                    h00 * top_line
                    + h10 * (top_slope * span)
                    + h01 * bottom_line
                    + h11 * (bottom_slope * span)
                )
            rebuilt_rows.append(rebuilt)

        rebuilt_band = np.vstack(rebuilt_rows)
        # 在核心带外再加一段混合区，让“重建结果”和“原始图”平滑交接。
        weights = np.zeros(band_end - band_start, dtype=np.float64)
        for idx, row in enumerate(range(band_start, band_end)):
            if core_start <= row < core_end:
                weights[idx] = 1.0
            elif row < core_start and blend_width > 0:
                weights[idx] = _smoothstep(np.array([(row - band_start + 1) / float(blend_width + 1)]))[0]
            elif row >= core_end and blend_width > 0:
                weights[idx] = _smoothstep(
                    np.array([(band_end - row) / float(blend_width + 1)])
                )[0]

        flattened[band_start:band_end, :] = (
            original_band * (1.0 - weights[:, np.newaxis])
            + rebuilt_band * weights[:, np.newaxis]
        )

    return flattened


def downsample_y(z_mm: np.ndarray, factor: int, dy_mm: float) -> Tuple[np.ndarray, float]:
    if factor <= 0:
        raise ValueError("Downsample factor must be positive.")
    if z_mm.shape[0] % factor != 0:
        raise ValueError("Image height must be divisible by the downsample factor.")
    if factor == 1:
        return z_mm.copy(), dy_mm

    reshaped = z_mm.reshape(z_mm.shape[0] // factor, factor, z_mm.shape[1])
    return reshaped.mean(axis=1), dy_mm * factor


def sample_y_block_centers(array: np.ndarray, factor: int, dy_mm: float) -> Tuple[np.ndarray, float]:
    if factor <= 0:
        raise ValueError("Downsample factor must be positive.")
    if array.shape[0] % factor != 0:
        raise ValueError("Image height must be divisible by the downsample factor.")
    if factor == 1:
        return array.copy(), dy_mm

    indices = np.arange(array.shape[0] // factor, dtype=np.int64) * factor + (factor // 2)
    indices = np.clip(indices, 0, array.shape[0] - 1)
    return array[indices, :].copy(), dy_mm * factor


def compute_surface_maps(
    z_mm: np.ndarray,
    dx_mm: float,
    dy_mm: float,
    gaussian_sigma: float = 0.064,
    pre_smooth_x_sigma: float = 0.0,
    outlier_filter_mode: str = "off",
    outlier_window_size: int = 0,
    outlier_threshold_mm: float = 0.0,
    outlier_max_cluster_size: int = 0,
    outlier_replace_mode: str = "median",
) -> Dict[str, np.ndarray]:
    if z_mm.ndim != 2:
        raise ValueError("Height map must be a 2D array.")

    z_used = filter_height_outliers(
        z_mm,
        mode=outlier_filter_mode,
        window_size=outlier_window_size,
        threshold_mm=outlier_threshold_mm,
        max_cluster_size=outlier_max_cluster_size,
        replace_mode=outlier_replace_mode,
    )

    # 仅在 X 方向做轻量平滑，用来压制列间高频条纹。sigma 使用物理
    # 长度 mm 表示，再按 dx 换算成像素，避免采样率变化时平滑宽度失真。
    if pre_smooth_x_sigma > 0:
        sigma_x_px = pre_smooth_x_sigma / dx_mm
        z_used = gaussian_filter1d(z_used, sigma=sigma_x_px, axis=1, mode="nearest")

    # 各向平滑同样以物理长度 mm 定义，再分别按 dy/dx 换算成像素。
    if gaussian_sigma > 0:
        sigma_y_px = gaussian_sigma / dy_mm
        sigma_x_px = gaussian_sigma / dx_mm
        z_used = gaussian_filter(z_used, sigma=(sigma_y_px, sigma_x_px), mode="nearest")

    edge_order = 2 if min(z_used.shape) >= 3 else 1
    grad_y, grad_x = np.gradient(z_used, dy_mm, dx_mm, edge_order=edge_order)
    curv2_x = np.gradient(grad_x, dx_mm, axis=1, edge_order=edge_order)
    curv2_y = np.gradient(grad_y, dy_mm, axis=0, edge_order=edge_order)
    curve_x = curv2_x / np.power(1.0 + grad_x * grad_x, 1.5)
    curve_y = curv2_y / np.power(1.0 + grad_y * grad_y, 1.5)

    return {
        "smoothed_height_mm": z_used,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "curv2_x": curv2_x,
        "curv2_y": curv2_y,
        "curve_x": curve_x,
        "curve_y": curve_y,
    }


def compute_resampled_surface_maps(
    z_mm: np.ndarray,
    dx_mm: float,
    dy_mm: float,
    downsample_factor: int,
    gaussian_sigma: float = 0.064,
    pre_smooth_x_sigma: float = 0.0,
    outlier_filter_mode: str = "off",
    outlier_window_size: int = 0,
    outlier_threshold_mm: float = 0.0,
    outlier_max_cluster_size: int = 0,
    outlier_replace_mode: str = "median",
) -> Tuple[Dict[str, np.ndarray], float]:
    full_resolution_maps = compute_surface_maps(
        z_mm,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        gaussian_sigma=gaussian_sigma,
        pre_smooth_x_sigma=pre_smooth_x_sigma,
        outlier_filter_mode=outlier_filter_mode,
        outlier_window_size=outlier_window_size,
        outlier_threshold_mm=outlier_threshold_mm,
        outlier_max_cluster_size=outlier_max_cluster_size,
        outlier_replace_mode=outlier_replace_mode,
    )

    resampled_maps: Dict[str, np.ndarray] = {}
    resampled_dy_mm = dy_mm
    for key, value in full_resolution_maps.items():
        resampled_value, current_resampled_dy_mm = sample_y_block_centers(
            value,
            factor=downsample_factor,
            dy_mm=dy_mm,
        )
        resampled_maps[key] = resampled_value
        resampled_dy_mm = current_resampled_dy_mm

    return resampled_maps, resampled_dy_mm


def save_csv(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), array, delimiter=",", fmt="%.8f")


def save_xyz_csv(array: np.ndarray, dx_mm: float, dy_mm: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    x_coords = np.arange(width, dtype=np.float64) * dx_mm
    y_coords = np.arange(height, dtype=np.float64) * dy_mm
    xx, yy = np.meshgrid(x_coords, y_coords)
    xyz = np.column_stack([xx.ravel(), yy.ravel(), array.astype(np.float64, copy=False).ravel()])
    np.savetxt(str(path), xyz, delimiter=",", fmt="%.8f", header="X,Y,Z", comments="")


def _percentile_limits(array: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> Tuple[float, float]:
    vmin = float(np.percentile(array, lower))
    vmax = float(np.percentile(array, upper))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    return vmin, vmax


def save_preview_png(
    array: np.ndarray,
    path: Path,
    cmap: str,
    center_zero: bool,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    if center_zero:
        _, vmax_pct = _percentile_limits(np.abs(array), lower=2.0, upper=98.0)
        vmax = max(vmax_pct, 1e-9)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        image = ax.imshow(array, cmap=cmap, norm=norm, aspect="auto")
    else:
        vmin, vmax = _percentile_limits(array)
        image = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def save_overview_png(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    items = [
        ("Raw Height", "height_raw_preview.png"),
        ("Corrected Height", "height_corrected_preview.png"),
        ("Resampled Height", "height_resampled_preview.png"),
        ("Grad Y", "grad_y.png"),
        ("Curv2 Y", "curv2_y.png"),
        ("Curve Y", "curve_y.png"),
    ]

    cards = []
    target_width = 560
    for title, filename in items:
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError("Overview source image not found: {}".format(path))

        image = Image.open(str(path)).convert("RGB")
        resized_height = max(1, int(image.height * target_width / float(image.width)))
        image = image.resize((target_width, resized_height))

        card = Image.new("RGB", (target_width, resized_height + 40), "white")
        card.paste(image, (0, 40))
        draw = ImageDraw.Draw(card)
        draw.text((10, 10), title, fill="black")
        cards.append(card)

    rows = []
    for index in range(0, len(cards), 2):
        left = cards[index]
        right = cards[index + 1] if index + 1 < len(cards) else Image.new("RGB", left.size, "white")
        row_height = max(left.height, right.height)
        row = Image.new("RGB", (left.width + right.width + 20, row_height), (245, 245, 245))
        row.paste(left, (0, 0))
        row.paste(right, (left.width + 20, 0))
        rows.append(row)

    total_width = max(row.width for row in rows)
    total_height = sum(row.height for row in rows) + 20 * (len(rows) - 1)
    overview = Image.new("RGB", (total_width, total_height), (230, 230, 230))
    y_offset = 0
    for row in rows:
        overview.paste(row, (0, y_offset))
        y_offset += row.height + 20

    overview.save(str(output_dir / "overview.png"))


def process_heightmap(input_path: Path, output_dir: Path, config: ProcessingConfig) -> Dict[str, np.ndarray]:
    gray = load_height_png(Path(input_path))
    if gray.shape[0] != config.expected_height:
        raise ValueError(
            "Input image height {} does not match configured stripe layout {}.".format(
                gray.shape[0], config.expected_height
            )
        )
    if config.crop_left_px < 0 or config.crop_right_px < 0:
        raise ValueError("Crop pixels must be non-negative.")
    if config.crop_left_px + config.crop_right_px >= gray.shape[1]:
        raise ValueError("Left/right crop removes the full image width.")

    # 左右裁切发生在所有计算之前，用于切掉原图中不需要参与分析的区域。
    # 裁切只影响 X 宽度，不改变扫描带高度与接缝位置。
    x_start = config.crop_left_px
    x_end = gray.shape[1] - config.crop_right_px if config.crop_right_px > 0 else gray.shape[1]
    gray = gray[:, x_start:x_end]
    if gray.shape[0] % config.downsample_factor != 0:
        raise ValueError("Image height must be divisible by the downsample factor.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_height_mm = gray.astype(np.float64) * config.dz_mm
    corrected_height_mm, seam_offsets = correct_scan_band_offsets(
        raw_height_mm,
        stripe_height=config.stripe_height,
        seam_window=config.seam_window,
        transition_window=config.transition_window,
        smooth_sigma_x=config.smooth_sigma_x,
        seam_flatten_half_window=config.seam_flatten_half_window,
        seam_flatten_sigma_x=config.seam_flatten_sigma_x,
        seam_flatten_blend_width=config.seam_flatten_blend_width,
        seam_flatten_method=config.seam_flatten_method,
        num_stripes=config.num_stripes,
    )
    resampled_height_mm, _ = downsample_y(corrected_height_mm, factor=config.downsample_factor, dy_mm=config.dy_mm)
    maps, resampled_dy_mm = compute_resampled_surface_maps(
        corrected_height_mm,
        dx_mm=config.dx_mm,
        dy_mm=config.dy_mm,
        downsample_factor=config.downsample_factor,
        gaussian_sigma=config.gaussian_sigma,
        pre_smooth_x_sigma=config.pre_smooth_x_sigma,
        outlier_filter_mode=config.outlier_filter_mode,
        outlier_window_size=config.outlier_window_size,
        outlier_threshold_mm=config.outlier_threshold_mm,
        outlier_max_cluster_size=config.outlier_max_cluster_size,
        outlier_replace_mode=config.outlier_replace_mode,
    )

    save_preview_png(raw_height_mm, output_dir / "height_raw_preview.png", "viridis", False, "Raw Height (mm)")
    save_preview_png(
        corrected_height_mm,
        output_dir / "height_corrected_preview.png",
        "viridis",
        False,
        "Corrected Height (mm)",
    )
    save_preview_png(
        resampled_height_mm,
        output_dir / "height_resampled_preview.png",
        "viridis",
        False,
        "Resampled Height (mm)",
    )
    save_float_tiff(corrected_height_mm, output_dir / "height_corrected_mm.tiff")
    save_float_tiff(resampled_height_mm, output_dir / "height_resampled_mm.tiff")
    save_xyz_csv(
        resampled_height_mm,
        dx_mm=config.dx_mm,
        dy_mm=resampled_dy_mm,
        path=output_dir / "height_resampled_xyz.csv",
    )

    for key in ("grad_x", "grad_y", "curv2_x", "curv2_y", "curve_x", "curve_y"):
        save_csv(maps[key], output_dir / (key + ".csv"))
        save_xyz_csv(maps[key], dx_mm=config.dx_mm, dy_mm=resampled_dy_mm, path=output_dir / (key + "_xyz.csv"))
        save_float_tiff(maps[key], output_dir / (key + ".tiff"))
        save_preview_png(maps[key], output_dir / (key + ".png"), "coolwarm", True, key)

    # 生成总览图，便于快速检查高度图修正前后以及 Y 向结果是否仍有接缝伪影。
    save_overview_png(output_dir)

    results = {
        "raw_height_mm": raw_height_mm,
        "corrected_height_mm": corrected_height_mm,
        "resampled_height_mm": resampled_height_mm,
        "seam_offsets": seam_offsets,
        "resampled_dy_mm": np.array([resampled_dy_mm], dtype=np.float64),
    }
    results.update(maps)
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="处理 16-bit 单通道光谱共焦 PNG 高度图，输出高度图、梯度图、曲率图。")
    parser.add_argument("--input", type=Path, help="输入的 16-bit 单通道 PNG 高度图路径。")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录，PNG/CSV/TIFF 结果都会写入这里。")
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="先生成一张假的高度图，再运行整条处理流程。",
    )
    parser.add_argument("--width", type=int, default=2048, help="假图宽度，单位像素。")
    parser.add_argument("--stripe-height", type=int, default=2048, help="每一道扫描带的高度，单位像素。")
    parser.add_argument("--num-stripes", type=int, default=3, help="扫描带数量。")
    parser.add_argument("--downsample-factor", type=int, default=16, help="Y 方向分步采样倍率，例如 16 表示 16 行取一组平均。")
    parser.add_argument("--dx-mm", type=float, default=0.08, help="X 方向物理分辨率，单位 mm/pixel。")
    parser.add_argument("--dy-mm", type=float, default=0.005615, help="Y 方向物理分辨率，单位 mm/pixel。")
    parser.add_argument("--dz-mm", type=float, default=0.0001, help="Z 方向高度分辨率，单位 mm/gray。")
    parser.add_argument("--crop-left-px", type=int, default=0, help="处理前从图像左侧裁掉的像素数。")
    parser.add_argument("--crop-right-px", type=int, default=0, help="处理前从图像右侧裁掉的像素数。")
    parser.add_argument("--seam-window", type=int, default=64, help="估计每条接缝整带偏移时，接缝上下各取多少行。")
    parser.add_argument(
        "--transition-window",
        type=int,
        default=0,
        help="旧版接缝过渡窗口参数，默认 0。当前主流程不再依赖这一步，保留仅为兼容。",
    )
    parser.add_argument(
        "--smooth-sigma-x",
        type=float,
        default=2.0,
        help="对逐列偏移 Δ(x) 做 X 向轻平滑的强度，值越大越平缓，但也更容易抹掉真实列间差异。",
    )
    parser.add_argument(
        "--seam-flatten-half-window",
        type=int,
        default=2,
        help="接缝窄带重建的半宽，单位是原始 Y 像素。默认只修 seam 附近极少几行，避免人为过渡带过宽。",
    )
    parser.add_argument(
        "--seam-flatten-sigma-x",
        type=float,
        default=0.0,
        help="接缝窄带重建时，对上下锚线做 X 向轻平滑的强度。默认关闭，优先保留真实列间差异。",
    )
    parser.add_argument(
        "--seam-flatten-blend-width",
        type=int,
        default=0,
        help="接缝核心带外侧的混合宽度。默认关闭，避免把修补区域扩展成可见的过渡带。",
    )
    parser.add_argument(
        "--seam-flatten-method",
        choices=["linear", "quadratic", "cubic"],
        default="linear",
        help="接缝窄带重建方法。linear 为旧版简单插值；quadratic 为固定二阶导风格过渡；cubic 为三次插值。",
    )
    parser.add_argument(
        "--pre-smooth-x-sigma",
        type=float,
        default=0.0,
        help="求导前仅对 X 方向做轻量高斯平滑，单位 mm，用于抑制 grad_x 中的纵向条纹。",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.064,
        help="求导前对整图做轻量高斯平滑的物理尺度，单位 mm，用于抑制噪声。",
    )
    parser.add_argument(
        "--outlier-filter-mode",
        choices=["off", "conservative", "medium"],
        default="off",
        help="离群噪点滤除模式。off 关闭；conservative 只处理孤立尖刺；medium 额外处理小团簇离群点。",
    )
    parser.add_argument(
        "--outlier-window-size",
        type=int,
        default=0,
        help="离群检测局部窗口尺寸，必须是奇数。设为 0 时按模式使用默认值。",
    )
    parser.add_argument(
        "--outlier-threshold-mm",
        type=float,
        default=0.0,
        help="离群点相对局部中值的高度阈值，单位 mm。设为 0 时按模式使用默认值。",
    )
    parser.add_argument(
        "--outlier-max-cluster-size",
        type=int,
        default=0,
        help="允许被替换的小连通域最大面积，单位像素。设为 0 时按模式使用默认值。",
    )
    parser.add_argument(
        "--outlier-replace-mode",
        choices=["median"],
        default="median",
        help="离群点替换方式。当前仅支持 median。",
    )
    parser.add_argument(
        "--synthetic-outlier-spike-count",
        type=int,
        default=0,
        help="生成假图时注入的单像素尖刺噪点数量。",
    )
    parser.add_argument(
        "--synthetic-outlier-cluster-count",
        type=int,
        default=0,
        help="生成假图时注入的小团簇离群点数量。",
    )
    parser.add_argument(
        "--synthetic-outlier-cluster-size",
        type=int,
        default=2,
        help="生成假图时每个小团簇的边长，单位像素。",
    )
    parser.add_argument(
        "--synthetic-outlier-amplitude-mm",
        type=float,
        default=0.06,
        help="生成假图时离群噪点的幅值，单位 mm。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = ProcessingConfig(
        dx_mm=args.dx_mm,
        dy_mm=args.dy_mm,
        dz_mm=args.dz_mm,
        stripe_height=args.stripe_height,
        num_stripes=args.num_stripes,
        downsample_factor=args.downsample_factor,
        crop_left_px=args.crop_left_px,
        crop_right_px=args.crop_right_px,
        seam_window=args.seam_window,
        transition_window=args.transition_window,
        smooth_sigma_x=args.smooth_sigma_x,
        seam_flatten_half_window=args.seam_flatten_half_window,
        seam_flatten_sigma_x=args.seam_flatten_sigma_x,
        seam_flatten_blend_width=args.seam_flatten_blend_width,
        seam_flatten_method=args.seam_flatten_method,
        pre_smooth_x_sigma=args.pre_smooth_x_sigma,
        gaussian_sigma=args.gaussian_sigma,
        outlier_filter_mode=args.outlier_filter_mode,
        outlier_window_size=args.outlier_window_size,
        outlier_threshold_mm=args.outlier_threshold_mm,
        outlier_max_cluster_size=args.outlier_max_cluster_size,
        outlier_replace_mode=args.outlier_replace_mode,
    )

    output_dir = Path(args.output_dir)
    input_path = args.input
    if args.generate_synthetic:
        synthetic = generate_synthetic_heightmap(
            width=args.width,
            stripe_height=args.stripe_height,
            num_stripes=args.num_stripes,
            dz_mm=args.dz_mm,
            outlier_spike_count=args.synthetic_outlier_spike_count,
            outlier_cluster_count=args.synthetic_outlier_cluster_count,
            outlier_cluster_size=args.synthetic_outlier_cluster_size,
            outlier_amplitude_mm=args.synthetic_outlier_amplitude_mm,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "synthetic_input.png"
        save_height_png(synthetic, input_path)
    elif input_path is None:
        parser.error("请通过 --input 指定输入 PNG，或使用 --generate-synthetic 生成假图。")

    process_heightmap(input_path, output_dir, config)
    print("处理完成，结果已输出到 {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

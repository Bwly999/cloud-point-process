import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from cloud_point_process.processor import (
    ProcessingConfig,
    _estimate_window_slope,
    compute_resampled_surface_maps,
    compute_surface_maps,
    correct_scan_band_offsets,
    downsample_y,
    flatten_seam_artifacts,
    generate_synthetic_heightmap,
    process_heightmap,
    save_height_png,
)


class TestPointCloudProcessor(unittest.TestCase):
    def test_estimate_window_slope_ignores_seam_side_spikes(self):
        stripe_height = 64
        width = 96
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        image = 0.0008 * np.minimum(y, stripe_height) + 0.0032 * np.maximum(y - stripe_height, 0)
        image += 0.01 * np.sin(x / 10.0)

        seam_y = stripe_height
        phase = np.arange(width, dtype=np.float64) / 4.0
        frac = phase - np.floor(phase)
        triangle = 2.0 * np.abs(2.0 * frac - 1.0) - 1.0
        for row in range(seam_y - 2, seam_y + 3):
            weight = 1.0 - 0.2 * abs(row - seam_y)
            image[row, :] += 0.05 * triangle * weight

        centered_top = 0.5 * (image[seam_y - 2, :] - image[seam_y - 4, :])
        outward_top = _estimate_window_slope(image, seam_y - 5, seam_y - 2)
        centered_bottom = 0.5 * (image[seam_y + 3, :] - image[seam_y + 1, :])
        outward_bottom = _estimate_window_slope(image, seam_y + 2, seam_y + 5)

        true_top = np.full(width, 0.0008, dtype=np.float64)
        true_bottom = np.full(width, 0.0032, dtype=np.float64)

        self.assertLess(np.median(np.abs(outward_top - true_top)), np.median(np.abs(centered_top - true_top)))
        self.assertLess(
            np.median(np.abs(outward_bottom - true_bottom)),
            np.median(np.abs(centered_bottom - true_bottom)),
        )

    def test_generate_synthetic_heightmap_returns_uint16_image(self):
        image = generate_synthetic_heightmap(width=64, stripe_height=32, num_stripes=3)

        self.assertEqual(image.shape, (96, 64))
        self.assertEqual(image.dtype, np.uint16)
        self.assertGreater(int(image.max()), int(image.min()))

    def test_correct_scan_band_offsets_aligns_seams(self):
        stripe_height = 32
        y, x = np.indices((stripe_height * 3, 48), dtype=np.float64)
        base = 0.02 * x + 0.05 * np.sin(x / 9.0) + 0.01 * np.cos(y / 5.0)
        delta1 = 0.8 + 0.1 * np.sin(np.linspace(0.0, np.pi, base.shape[1]))
        delta2 = -0.5 + 0.08 * np.cos(np.linspace(0.0, np.pi, base.shape[1]))

        raw = base.copy()
        raw[stripe_height:stripe_height * 2] += delta1
        raw[stripe_height * 2:] += delta1 + delta2

        corrected, offsets = correct_scan_band_offsets(
            raw,
            stripe_height=stripe_height,
            seam_window=8,
            transition_window=4,
            smooth_sigma_x=2.0,
        )

        before_first = np.median(raw[stripe_height + 4:stripe_height + 8], axis=0) - np.median(
            raw[stripe_height - 8:stripe_height - 4], axis=0
        )
        after_first = np.median(
            corrected[stripe_height + 4:stripe_height + 8], axis=0
        ) - np.median(corrected[stripe_height - 8:stripe_height - 4], axis=0)
        before_second = np.median(
            raw[stripe_height * 2 + 4:stripe_height * 2 + 8], axis=0
        ) - np.median(raw[stripe_height * 2 - 8:stripe_height * 2 - 4], axis=0)
        after_second = np.median(
            corrected[stripe_height * 2 + 4:stripe_height * 2 + 8], axis=0
        ) - np.median(corrected[stripe_height * 2 - 8:stripe_height * 2 - 4], axis=0)

        self.assertEqual(len(offsets), 2)
        self.assertLess(np.median(np.abs(after_first)), np.median(np.abs(before_first)) * 0.1)
        self.assertLess(np.median(np.abs(after_second)), np.median(np.abs(before_second)) * 0.1)
        self.assertLess(np.median(np.abs(after_first)), 0.03)
        self.assertLess(np.median(np.abs(after_second)), 0.03)

    def test_downsample_y_averages_blocks(self):
        image = np.arange(16, dtype=np.float64).reshape(8, 2)

        downsampled, dy = downsample_y(image, factor=4, dy_mm=0.5)

        expected = np.array([[3.0, 4.0], [11.0, 12.0]])
        np.testing.assert_allclose(downsampled, expected)
        self.assertEqual(dy, 2.0)

    def test_compute_surface_maps_matches_plane(self):
        dx = 0.5
        dy = 0.25
        x = np.arange(20, dtype=np.float64) * dx
        y = np.arange(18, dtype=np.float64) * dy
        xx, yy = np.meshgrid(x, y)
        z = 2.0 * xx + 3.0 * yy

        maps = compute_surface_maps(z, dx_mm=dx, dy_mm=dy, gaussian_sigma=0.0)

        np.testing.assert_allclose(maps["grad_x"], 2.0, atol=1e-10)
        np.testing.assert_allclose(maps["grad_y"], 3.0, atol=1e-10)
        np.testing.assert_allclose(maps["curv2_x"], 0.0, atol=1e-10)
        np.testing.assert_allclose(maps["curv2_y"], 0.0, atol=1e-10)
        np.testing.assert_allclose(maps["curve_x"], 0.0, atol=1e-10)
        np.testing.assert_allclose(maps["curve_y"], 0.0, atol=1e-10)

    def test_compute_resampled_surface_maps_keeps_curve_y_stable_across_downsample_factors(self):
        dx = 0.08
        dy = 0.005615
        width = 24
        height = 4096
        y = np.arange(height, dtype=np.float64) * dy
        profile = 0.05 * np.sin(2.0 * np.pi * y / 0.3)
        z = np.repeat(profile[:, np.newaxis], width, axis=1)

        maps_factor_1, resampled_dy_1 = compute_resampled_surface_maps(
            z,
            dx_mm=dx,
            dy_mm=dy,
            downsample_factor=1,
            gaussian_sigma=0.064,
        )
        maps_factor_8, resampled_dy_8 = compute_resampled_surface_maps(
            z,
            dx_mm=dx,
            dy_mm=dy,
            downsample_factor=8,
            gaussian_sigma=0.064,
        )

        self.assertEqual(resampled_dy_1, dy)
        self.assertEqual(resampled_dy_8, dy * 8)

        grad_y_max_1 = float(np.percentile(np.abs(maps_factor_1["grad_y"]), 95))
        grad_y_max_8 = float(np.percentile(np.abs(maps_factor_8["grad_y"]), 95))
        curve_y_max_1 = float(np.percentile(np.abs(maps_factor_1["curve_y"]), 95))
        curve_y_max_8 = float(np.percentile(np.abs(maps_factor_8["curve_y"]), 95))

        self.assertLess(abs(grad_y_max_8 - grad_y_max_1) / grad_y_max_1, 0.1)
        self.assertLess(abs(curve_y_max_8 - curve_y_max_1) / curve_y_max_1, 0.1)

    def test_compute_surface_maps_uses_gaussian_sigma_in_physical_mm(self):
        dx = 0.08
        dy = 0.005615
        sigma_mm = 0.08
        y, x = np.indices((96, 64), dtype=np.float64)
        z = 0.03 * np.sin(x / 5.0) + 0.04 * np.cos(y / 9.0) + 0.002 * y

        maps = compute_surface_maps(z, dx_mm=dx, dy_mm=dy, gaussian_sigma=sigma_mm)
        expected = gaussian_filter(z, sigma=(sigma_mm / dy, sigma_mm / dx), mode="nearest")

        np.testing.assert_allclose(maps["smoothed_height_mm"], expected, rtol=1e-10, atol=1e-10)

    def test_pre_smooth_x_sigma_reduces_vertical_stripes_in_grad_x(self):
        width = 96
        height = 72
        y, x = np.indices((height, width), dtype=np.float64)
        stripe_noise = 0.08 * np.sign(np.sin(np.arange(width, dtype=np.float64) * 1.7))
        z = 0.002 * y + stripe_noise[np.newaxis, :]

        raw_maps = compute_surface_maps(
            z,
            dx_mm=0.08,
            dy_mm=0.08,
            gaussian_sigma=0.0,
            pre_smooth_x_sigma=0.0,
        )
        smooth_maps = compute_surface_maps(
            z,
            dx_mm=0.08,
            dy_mm=0.08,
            gaussian_sigma=0.0,
            pre_smooth_x_sigma=0.064,
        )

        raw_grad_x = np.median(np.abs(raw_maps["grad_x"]))
        smooth_grad_x = np.median(np.abs(smooth_maps["grad_x"]))
        raw_grad_y = np.median(np.abs(raw_maps["grad_y"]))
        smooth_grad_y = np.median(np.abs(smooth_maps["grad_y"]))

        self.assertLess(smooth_grad_x, raw_grad_x * 0.75)
        np.testing.assert_allclose(smooth_grad_y, raw_grad_y, rtol=0.05, atol=1e-6)

    def test_outlier_filter_off_preserves_height_without_gaussian(self):
        y, x = np.indices((24, 20), dtype=np.float64)
        z = 0.002 * x + 0.003 * y + 0.01 * np.sin(x / 5.0)

        maps = compute_surface_maps(
            z,
            dx_mm=0.08,
            dy_mm=0.08,
            gaussian_sigma=0.0,
            outlier_filter_mode="off",
        )

        np.testing.assert_allclose(maps["smoothed_height_mm"], z, atol=1e-12)

    def test_conservative_outlier_filter_replaces_isolated_spikes(self):
        height = 48
        width = 40
        y, x = np.indices((height, width), dtype=np.float64)
        base = 0.0015 * x + 0.0025 * y + 0.006 * np.sin(x / 7.0)
        z = base.copy()
        spikes = [(10, 8, 0.09), (20, 24, -0.08), (33, 15, 0.07)]
        for row, col, delta in spikes:
            z[row, col] += delta

        raw_maps = compute_surface_maps(
            z,
            dx_mm=0.08,
            dy_mm=0.08,
            gaussian_sigma=0.0,
            outlier_filter_mode="off",
        )
        filtered_maps = compute_surface_maps(
            z,
            dx_mm=0.08,
            dy_mm=0.08,
            gaussian_sigma=0.0,
            outlier_filter_mode="conservative",
            outlier_window_size=3,
            outlier_threshold_mm=0.03,
            outlier_max_cluster_size=1,
        )

        stable_mask = np.ones_like(z, dtype=bool)
        for row, col, _ in spikes:
            stable_mask[max(0, row - 1):min(height, row + 2), max(0, col - 1):min(width, col + 2)] = False
            raw_peak = float(
                np.max(np.abs(raw_maps["grad_x"][max(0, row - 1):min(height, row + 2), max(0, col - 1):min(width, col + 2)]))
            )
            filtered_peak = float(
                np.max(
                    np.abs(
                        filtered_maps["grad_x"][
                            max(0, row - 1):min(height, row + 2),
                            max(0, col - 1):min(width, col + 2),
                        ]
                    )
                )
            )
            self.assertLess(filtered_peak, raw_peak * 0.1)
            self.assertLess(
                abs(filtered_maps["smoothed_height_mm"][row, col] - base[row, col]),
                abs(z[row, col] - base[row, col]) * 0.1,
            )

        np.testing.assert_allclose(filtered_maps["smoothed_height_mm"][stable_mask], z[stable_mask], atol=1e-12)

    def test_medium_outlier_filter_removes_small_clusters_beyond_conservative(self):
        y, x = np.indices((40, 40), dtype=np.float64)
        base = 0.002 * x + 0.001 * y + 0.004 * np.cos(y / 6.0)
        z = base.copy()
        z[14:16, 18:20] += 0.06

        conservative_maps = compute_surface_maps(
            z,
            dx_mm=0.08,
            dy_mm=0.08,
            gaussian_sigma=0.0,
            outlier_filter_mode="conservative",
            outlier_window_size=3,
            outlier_threshold_mm=0.02,
            outlier_max_cluster_size=1,
        )
        medium_maps = compute_surface_maps(
            z,
            dx_mm=0.08,
            dy_mm=0.08,
            gaussian_sigma=0.0,
            outlier_filter_mode="medium",
            outlier_window_size=5,
            outlier_threshold_mm=0.02,
            outlier_max_cluster_size=4,
        )

        cluster = np.s_[14:16, 18:20]
        conservative_error = float(
            np.median(np.abs(conservative_maps["smoothed_height_mm"][cluster] - base[cluster]))
        )
        medium_error = float(np.median(np.abs(medium_maps["smoothed_height_mm"][cluster] - base[cluster])))

        self.assertGreater(conservative_error, 0.03)
        self.assertLess(medium_error, conservative_error * 0.2)

    def test_seam_correction_does_not_introduce_y_derivative_artifacts(self):
        stripe_height = 32
        width = 80
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        base = 0.01 * np.sin(x / 8.0) + 0.015 * np.cos(y / 7.0) + 0.003 * x
        delta1 = 0.55 + 0.05 * np.sin(np.linspace(0.0, np.pi, width))
        delta2 = -0.38 + 0.04 * np.cos(np.linspace(0.0, np.pi, width))

        raw = base.copy()
        raw[stripe_height:stripe_height * 2] += delta1
        raw[stripe_height * 2:] += delta1 + delta2

        corrected, _ = correct_scan_band_offsets(
            raw,
            stripe_height=stripe_height,
            seam_window=8,
            transition_window=4,
            smooth_sigma_x=2.0,
        )
        resampled, dy_mm = downsample_y(corrected, factor=4, dy_mm=0.5)
        maps = compute_surface_maps(resampled, dx_mm=0.8, dy_mm=dy_mm, gaussian_sigma=0.064)

        for seam in (stripe_height, stripe_height * 2):
            seam_row = seam // 4
            seam_slice = slice(max(0, seam_row - 2), min(maps["grad_y"].shape[0], seam_row + 2))
            background = np.concatenate(
                [
                    np.abs(maps["grad_y"])[max(0, seam_row - 8):max(0, seam_row - 4), :].ravel(),
                    np.abs(maps["grad_y"])[min(maps["grad_y"].shape[0], seam_row + 4):min(maps["grad_y"].shape[0], seam_row + 8), :].ravel(),
                ]
            )
            background_curv2 = np.concatenate(
                [
                    np.abs(maps["curv2_y"])[max(0, seam_row - 8):max(0, seam_row - 4), :].ravel(),
                    np.abs(maps["curv2_y"])[min(maps["curv2_y"].shape[0], seam_row + 4):min(maps["curv2_y"].shape[0], seam_row + 8), :].ravel(),
                ]
            )
            background_curve = np.concatenate(
                [
                    np.abs(maps["curve_y"])[max(0, seam_row - 8):max(0, seam_row - 4), :].ravel(),
                    np.abs(maps["curve_y"])[min(maps["curve_y"].shape[0], seam_row + 4):min(maps["curve_y"].shape[0], seam_row + 8), :].ravel(),
                ]
            )

            seam_grad = np.median(np.abs(maps["grad_y"])[seam_slice, :])
            seam_curv2 = np.median(np.abs(maps["curv2_y"])[seam_slice, :])
            seam_curve = np.median(np.abs(maps["curve_y"])[seam_slice, :])

            self.assertLess(seam_grad, np.median(background) * 2.0)
            self.assertLess(seam_curv2, np.median(background_curv2) * 5.0)
            self.assertLess(seam_curve, np.median(background_curve) * 5.0)

    def test_zigzag_band_offsets_do_not_leave_y_seam_after_resample(self):
        config = ProcessingConfig()
        stripe_height = 128
        width = 256
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        base = 0.03 * np.sin(x / 15.0) + 0.01 * np.cos(y / 11.0) + 0.002 * x
        phase = np.linspace(0.0, 8.0 * np.pi, width)
        delta1 = 0.35 + 0.12 * np.sign(np.sin(phase)) + 0.06 * np.sin(phase * 0.7)
        delta2 = -0.28 + 0.10 * np.sign(np.sin(phase * 1.1 + 0.4))

        raw = base.copy()
        raw[stripe_height:stripe_height * 2] += delta1
        raw[stripe_height * 2:] += delta1 + delta2

        corrected, _ = correct_scan_band_offsets(
            raw,
            stripe_height=stripe_height,
            seam_window=16,
            transition_window=config.transition_window,
            smooth_sigma_x=config.smooth_sigma_x,
        )
        resampled, dy_mm = downsample_y(corrected, factor=8, dy_mm=0.5)
        maps = compute_surface_maps(resampled, dx_mm=0.8, dy_mm=dy_mm, gaussian_sigma=0.064)

        for seam in (stripe_height, stripe_height * 2):
            seam_row = seam // 8
            seam_grad = np.median(np.abs(maps["grad_y"])[seam_row - 2:seam_row + 2, :])
            bg_grad = np.median(
                np.concatenate(
                    [
                        np.abs(maps["grad_y"])[seam_row - 10:seam_row - 5, :].ravel(),
                        np.abs(maps["grad_y"])[seam_row + 5:seam_row + 10, :].ravel(),
                    ]
                )
            )
            seam_curv2 = np.median(np.abs(maps["curv2_y"])[seam_row - 2:seam_row + 2, :])
            bg_curv2 = np.median(
                np.concatenate(
                    [
                        np.abs(maps["curv2_y"])[seam_row - 10:seam_row - 5, :].ravel(),
                        np.abs(maps["curv2_y"])[seam_row + 5:seam_row + 10, :].ravel(),
                    ]
                )
            )

            self.assertLess(seam_grad, bg_grad * 1.6)
            self.assertLess(seam_curv2, bg_curv2 * 1.4)

    def test_seam_flattening_avoids_half_window_edge_spikes(self):
        stripe_height = 48
        width = 160
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        base = 0.015 * np.sin(x / 9.0) + 0.008 * np.cos(y / 7.0) + 0.001 * x + 0.0008 * y
        phase = np.arange(width, dtype=np.float64) / 4.0
        frac = phase - np.floor(phase)
        triangle = 2.0 * np.abs(2.0 * frac - 1.0) - 1.0

        seam_y = stripe_height
        image = base.copy()
        for row in range(seam_y - 3, seam_y + 4):
            weight = 1.0 - 0.18 * abs(row - seam_y)
            image[row, :] += 0.12 * triangle * weight

        flattened = flatten_seam_artifacts(
            image,
            stripe_height=stripe_height,
            num_stripes=3,
            half_window=4,
            sigma_x=0.0,
            blend_width=2,
            method="cubic",
        )

        grad_before = np.abs(np.gradient(image, axis=0))
        grad_after = np.abs(np.gradient(flattened, axis=0))
        edge_rows = [seam_y - 4, seam_y + 3]
        bg_before = np.median(grad_before[seam_y - 12:seam_y - 8, :])
        bg_after = np.median(grad_after[seam_y - 12:seam_y - 8, :])

        for row in edge_rows:
            self.assertLess(np.median(grad_after[row:row + 1, :]), np.median(grad_before[row:row + 1, :]) * 0.85)
            self.assertLess(np.median(grad_after[row:row + 1, :]), bg_after * 2.5)

    def test_flatten_seam_artifacts_supports_linear_mode(self):
        stripe_height = 32
        width = 96
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        image = 0.01 * np.sin(x / 8.0) + 0.004 * np.cos(y / 5.0) + 0.001 * x
        image[stripe_height - 1:stripe_height + 2, :] += 0.08

        flattened = flatten_seam_artifacts(
            image,
            stripe_height=stripe_height,
            num_stripes=3,
            half_window=2,
            sigma_x=0.0,
            blend_width=0,
            method="linear",
        )

        self.assertEqual(flattened.shape, image.shape)
        seam_grad_before = np.median(np.abs(np.gradient(image, axis=0))[stripe_height - 1:stripe_height + 1, :])
        seam_grad_after = np.median(np.abs(np.gradient(flattened, axis=0))[stripe_height - 1:stripe_height + 1, :])
        self.assertLess(seam_grad_after, seam_grad_before)

    def test_flatten_seam_artifacts_supports_quadratic_mode(self):
        stripe_height = 32
        width = 96
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        image = 0.01 * np.sin(x / 8.0) + 0.004 * np.cos(y / 5.0) + 0.001 * x
        image[stripe_height - 1:stripe_height + 2, :] += 0.08

        flattened = flatten_seam_artifacts(
            image,
            stripe_height=stripe_height,
            num_stripes=3,
            half_window=2,
            sigma_x=0.0,
            blend_width=0,
            method="quadratic",
        )

        self.assertEqual(flattened.shape, image.shape)
        seam_grad_before = np.median(np.abs(np.gradient(image, axis=0))[stripe_height - 1:stripe_height + 1, :])
        seam_grad_after = np.median(np.abs(np.gradient(flattened, axis=0))[stripe_height - 1:stripe_height + 1, :])
        self.assertLess(seam_grad_after, seam_grad_before)

    def test_quadratic_mode_keeps_curve_y_smoother_than_linear_at_seam(self):
        stripe_height = 128
        width = 256
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        base = 0.02 * np.sin(x / 11.0) + 0.008 * np.cos(y / 9.0) + 0.0015 * x + 0.0007 * y
        phase = np.arange(width, dtype=np.float64) / 4.0
        frac = phase - np.floor(phase)
        triangle = 2.0 * np.abs(2.0 * frac - 1.0) - 1.0

        image = base.copy()
        seam_y = stripe_height
        for row in range(seam_y - 3, seam_y + 4):
            weight = 1.0 - 0.18 * abs(row - seam_y)
            image[row, :] += 0.10 * triangle * weight

        flattened_linear = flatten_seam_artifacts(
            image,
            stripe_height=stripe_height,
            num_stripes=3,
            half_window=2,
            sigma_x=0.0,
            blend_width=0,
            method="linear",
        )
        flattened_quadratic = flatten_seam_artifacts(
            image,
            stripe_height=stripe_height,
            num_stripes=3,
            half_window=2,
            sigma_x=0.0,
            blend_width=0,
            method="quadratic",
        )

        linear_resampled, dy_mm = downsample_y(flattened_linear, factor=8, dy_mm=0.5)
        quadratic_resampled, _ = downsample_y(flattened_quadratic, factor=8, dy_mm=0.5)
        linear_maps = compute_surface_maps(linear_resampled, dx_mm=0.8, dy_mm=dy_mm, gaussian_sigma=0.064)
        quadratic_maps = compute_surface_maps(quadratic_resampled, dx_mm=0.8, dy_mm=dy_mm, gaussian_sigma=0.064)

        seam_row = seam_y // 8
        linear_curve = linear_maps["curve_y"][seam_row, :]
        quadratic_curve = quadratic_maps["curve_y"][seam_row, :]
        linear_roughness = np.median(np.abs(np.diff(linear_curve)))
        quadratic_roughness = np.median(np.abs(np.diff(quadratic_curve)))
        linear_magnitude = np.median(np.abs(linear_curve))
        quadratic_magnitude = np.median(np.abs(quadratic_curve))

        self.assertLessEqual(quadratic_roughness, linear_roughness * 1.02)
        self.assertLessEqual(quadratic_magnitude, linear_magnitude * 1.05)

    def test_quadratic_mode_preserves_boundary_slope_order(self):
        stripe_height = 64
        width = 128
        y, x = np.indices((stripe_height * 3, width), dtype=np.float64)
        base = 0.0008 * np.minimum(y, stripe_height) + 0.0032 * np.maximum(y - stripe_height, 0)
        base += 0.01 * np.sin(x / 10.0)

        image = base.copy()
        seam_y = stripe_height
        phase = np.arange(width, dtype=np.float64) / 4.0
        frac = phase - np.floor(phase)
        triangle = 2.0 * np.abs(2.0 * frac - 1.0) - 1.0
        for row in range(seam_y - 2, seam_y + 3):
            weight = 1.0 - 0.2 * abs(row - seam_y)
            image[row, :] += 0.05 * triangle * weight

        flattened = flatten_seam_artifacts(
            image,
            stripe_height=stripe_height,
            num_stripes=3,
            half_window=2,
            sigma_x=0.0,
            blend_width=0,
            method="quadratic",
        )

        core_start = seam_y - 2
        core_end = seam_y + 2
        top_local_slope = flattened[core_start, :] - flattened[core_start - 1, :]
        bottom_local_slope = flattened[core_end, :] - flattened[core_end - 1, :]

        self.assertLess(np.median(top_local_slope), np.median(bottom_local_slope))
        self.assertGreaterEqual(np.mean(top_local_slope <= bottom_local_slope), 0.7)

    def test_process_heightmap_exports_png_and_csv(self):
        config = ProcessingConfig(
            dx_mm=0.08,
            dy_mm=0.005615,
            dz_mm=0.0001,
            stripe_height=32,
            num_stripes=3,
            downsample_factor=4,
            seam_window=8,
            transition_window=4,
            smooth_sigma_x=2.0,
            seam_flatten_blend_width=1,
            gaussian_sigma=0.064,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "synthetic.png"
            output_dir = tmp_path / "outputs"

            image = generate_synthetic_heightmap(width=64, stripe_height=32, num_stripes=3)
            save_height_png(image, input_path)

            results = process_heightmap(input_path, output_dir, config)

            expected_files = [
                "height_raw_preview.png",
                "height_corrected_preview.png",
                "height_resampled_preview.png",
                "overview.png",
                "height_corrected_mm.tiff",
                "height_resampled_mm.tiff",
                "height_resampled_xyz.csv",
                "grad_x.png",
                "grad_x.csv",
                "grad_x_xyz.csv",
                "grad_x.tiff",
                "grad_y.png",
                "grad_y.csv",
                "grad_y_xyz.csv",
                "grad_y.tiff",
                "curv2_x.png",
                "curv2_x.csv",
                "curv2_x_xyz.csv",
                "curv2_x.tiff",
                "curv2_y.png",
                "curv2_y.csv",
                "curv2_y_xyz.csv",
                "curv2_y.tiff",
                "curve_x.png",
                "curve_x.csv",
                "curve_x_xyz.csv",
                "curve_x.tiff",
                "curve_y.png",
                "curve_y.csv",
                "curve_y_xyz.csv",
                "curve_y.tiff",
            ]

            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)

            self.assertEqual(results["raw_height_mm"].shape, (96, 64))
            self.assertEqual(results["resampled_height_mm"].shape, (24, 64))

            xyz = np.loadtxt(output_dir / "height_resampled_xyz.csv", delimiter=",", skiprows=1)
            self.assertEqual(xyz.shape, (24 * 64, 3))
            np.testing.assert_allclose(xyz[0], [0.0, 0.0, results["resampled_height_mm"][0, 0]], atol=1e-8)
            np.testing.assert_allclose(
                xyz[-1],
                [
                    (results["resampled_height_mm"].shape[1] - 1) * config.dx_mm,
                    (results["resampled_height_mm"].shape[0] - 1) * (config.dy_mm * config.downsample_factor),
                    results["resampled_height_mm"][-1, -1],
                ],
                atol=1e-8,
            )

            grad_xyz = np.loadtxt(output_dir / "grad_y_xyz.csv", delimiter=",", skiprows=1)
            self.assertEqual(grad_xyz.shape, (24 * 64, 3))
            np.testing.assert_allclose(grad_xyz[0], [0.0, 0.0, results["grad_y"][0, 0]], atol=1e-8)

    def test_process_heightmap_supports_left_right_crop(self):
        config = ProcessingConfig(
            stripe_height=32,
            num_stripes=3,
            downsample_factor=4,
            crop_left_px=5,
            crop_right_px=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "synthetic.png"
            output_dir = tmp_path / "outputs"

            image = generate_synthetic_heightmap(width=64, stripe_height=32, num_stripes=3)
            save_height_png(image, input_path)

            results = process_heightmap(input_path, output_dir, config)

            self.assertEqual(results["raw_height_mm"].shape, (96, 52))
            self.assertEqual(results["corrected_height_mm"].shape, (96, 52))
            self.assertEqual(results["resampled_height_mm"].shape, (24, 52))

    def test_process_heightmap_reduces_outlier_artifacts_for_noisy_synthetic_input(self):
        off_config = ProcessingConfig(
            stripe_height=96,
            num_stripes=1,
            downsample_factor=1,
            gaussian_sigma=0.0,
            outlier_filter_mode="off",
        )
        filtered_config = ProcessingConfig(
            stripe_height=96,
            num_stripes=1,
            downsample_factor=1,
            gaussian_sigma=0.0,
            outlier_filter_mode="medium",
            outlier_window_size=5,
            outlier_threshold_mm=0.02,
            outlier_max_cluster_size=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            clean_input_path = tmp_path / "synthetic_clean.png"
            input_path = tmp_path / "synthetic_noisy.png"
            clean_output_dir = tmp_path / "clean"
            off_output_dir = tmp_path / "off"
            filtered_output_dir = tmp_path / "filtered"

            clean_image = generate_synthetic_heightmap(
                width=64,
                stripe_height=96,
                num_stripes=1,
            )
            image = clean_image.astype(np.int32, copy=True)
            delta_gray = int(round(0.08 / off_config.dz_mm))
            image[12, 10] += delta_gray
            image[30, 26] -= delta_gray
            image[54:56, 40:42] += delta_gray
            image = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            save_height_png(clean_image, clean_input_path)
            save_height_png(image, input_path)

            clean_results = process_heightmap(clean_input_path, clean_output_dir, off_config)
            off_results = process_heightmap(input_path, off_output_dir, off_config)
            filtered_results = process_heightmap(input_path, filtered_output_dir, filtered_config)

            self.assertTrue((filtered_output_dir / "overview.png").exists())
            self.assertTrue((filtered_output_dir / "grad_y.png").exists())

            noisy_mask = np.abs(off_results["raw_height_mm"] - clean_results["raw_height_mm"]) > 1e-12
            off_error = float(
                np.max(
                    np.abs(
                        off_results["smoothed_height_mm"][noisy_mask]
                        - clean_results["smoothed_height_mm"][noisy_mask]
                    )
                )
            )
            filtered_error = float(
                np.max(
                    np.abs(
                        filtered_results["smoothed_height_mm"][noisy_mask]
                        - clean_results["smoothed_height_mm"][noisy_mask]
                    )
                )
            )
            off_median_error = float(
                np.median(
                    np.abs(
                        off_results["smoothed_height_mm"][noisy_mask]
                        - clean_results["smoothed_height_mm"][noisy_mask]
                    )
                )
            )
            filtered_median_error = float(
                np.median(
                    np.abs(
                        filtered_results["smoothed_height_mm"][noisy_mask]
                        - clean_results["smoothed_height_mm"][noisy_mask]
                    )
                )
            )

            self.assertLess(filtered_error, off_error * 0.35)
            self.assertLess(filtered_median_error, off_median_error * 0.2)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cloud_point_process.processor import (
    ProcessingConfig,
    compute_surface_maps,
    correct_scan_band_offsets,
    downsample_y,
    flatten_seam_artifacts,
    generate_synthetic_heightmap,
    process_heightmap,
    save_height_png,
)


class TestPointCloudProcessor(unittest.TestCase):
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
            pre_smooth_x_sigma=0.8,
        )

        raw_grad_x = np.median(np.abs(raw_maps["grad_x"]))
        smooth_grad_x = np.median(np.abs(smooth_maps["grad_x"]))
        raw_grad_y = np.median(np.abs(raw_maps["grad_y"]))
        smooth_grad_y = np.median(np.abs(smooth_maps["grad_y"]))

        self.assertLess(smooth_grad_x, raw_grad_x * 0.75)
        np.testing.assert_allclose(smooth_grad_y, raw_grad_y, rtol=0.05, atol=1e-6)

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
        maps = compute_surface_maps(resampled, dx_mm=0.8, dy_mm=dy_mm, gaussian_sigma=0.8)

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
        maps = compute_surface_maps(resampled, dx_mm=0.8, dy_mm=dy_mm, gaussian_sigma=0.8)

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
            gaussian_sigma=0.8,
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
                "grad_x.png",
                "grad_x.csv",
                "grad_x.tiff",
                "grad_y.png",
                "grad_y.csv",
                "grad_y.tiff",
                "curv2_x.png",
                "curv2_x.csv",
                "curv2_x.tiff",
                "curv2_y.png",
                "curv2_y.csv",
                "curv2_y.tiff",
                "curve_x.png",
                "curve_x.csv",
                "curve_x.tiff",
                "curve_y.png",
                "curve_y.csv",
                "curve_y.tiff",
            ]

            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)

            self.assertEqual(results["raw_height_mm"].shape, (96, 64))
            self.assertEqual(results["resampled_height_mm"].shape, (24, 64))

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


if __name__ == "__main__":
    unittest.main()

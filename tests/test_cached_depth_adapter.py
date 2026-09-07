import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from cached_depth_adapter import depth_path, fit_one, resize_depth_and_mask, scaled_camera_fields


class CachedDepthTest(unittest.TestCase):
    def test_uses_original_frame_not_keyframe_position(self):
        self.assertEqual(depth_path("cache", np.int64(599)).name, "frame_000599_foundation_depth_m.npy")
        for index in (-1, 1800, 1.5):
            with self.assertRaises(ValueError):
                depth_path("cache", index)

    def test_fully_valid_resize_matches_native_exactly_without_unit_scaling(self):
        depth = np.linspace(.1, 2., 48, dtype=np.float32).reshape(6, 8)
        mask = np.zeros((6, 8), bool)
        mask[2:5, 3:6] = True
        resized, excluded, valid = resize_depth_and_mask(depth, mask, (3, 5))
        np.testing.assert_array_equal(resized, cv2.resize(depth, (5, 3)))
        np.testing.assert_array_equal(excluded, cv2.resize(mask.astype(np.uint8), (5, 3)))
        self.assertTrue(valid.all())

    def test_invalid_depth_is_not_zero_filled_support(self):
        depth = np.ones((4, 4), np.float32)
        depth[0, 0], depth[0, 2], depth[2, 0] = np.nan, 0., -1.
        resized, excluded, valid = resize_depth_and_mask(depth, np.zeros((4, 4)), (2, 2))
        np.testing.assert_array_equal(valid, [[False, False], [False, True]])
        self.assertTrue(np.isnan(resized[~valid]).all())
        self.assertTrue((excluded[~valid] == 1).all())
        self.assertEqual(resized[1, 1], 1.)

    def test_missing_background_or_wrong_dtype_fails(self):
        for depth, mask in ((np.full((2, 2), np.nan, np.float32), np.zeros((2, 2))),
                            (np.ones((2, 2), np.float32), np.ones((2, 2))),
                            (np.ones((2, 2), np.float64), np.zeros((2, 2)))):
            with self.assertRaises(ValueError):
                resize_depth_and_mask(depth, mask, (2, 2))

    def test_expansion_carries_thresholds_and_keeps_native_sigma(self):
        calls = []
        def estimator(slam, depth, **kwargs):
            calls.append(kwargs)
            return np.nan if len(calls) == 1 else .5
        scale, near, far, audit = fit_one(np.ones((2, 2), np.float32), np.full((2, 2), .5, np.float32),
                                        np.zeros((2, 2)), estimator=estimator)
        self.assertEqual(scale, .5)
        self.assertAlmostEqual(near, .3)
        self.assertAlmostEqual(far, .8)
        self.assertEqual(audit["retries"], 1)
        self.assertEqual(calls[0]["sigma"], .5)

    def test_empty_band_is_expanded_without_empty_optimizer_call(self):
        calls = []
        def estimator(slam, depth, **kwargs):
            calls.append(kwargs)
            return 2.
        _, _, _, audit = fit_one(np.ones((2, 2), np.float32), np.full((2, 2), .9, np.float32),
                                np.zeros((2, 2)), estimator=estimator)
        self.assertGreater(audit["empty_bands_skipped"], 0)
        self.assertEqual(len(calls), 1)

    def test_no_silent_scale_fallback(self):
        with self.assertRaises(RuntimeError):
            fit_one(np.ones((2, 2), np.float32), np.full((2, 2), .5, np.float32), np.zeros((2, 2)),
                    estimator=lambda *a, **k: np.nan, max_attempts=2)

    def test_original_scale_fitter_recovers_known_metric_ratio(self):
        depth = np.full((4, 4), .5, np.float32)
        depth[0, 0] = np.nan
        resized, excluded, _ = resize_depth_and_mask(depth, np.zeros((4, 4)), (4, 4))
        scale, _, _, audit = fit_one(np.full((4, 4), 2., np.float32), resized, excluded)
        self.assertAlmostEqual(scale, .25, places=6)
        self.assertEqual(audit["positive_background_support"], 15)
        self.assertEqual(audit["retries"], 0)

    def test_scale_saved_separately_leaves_raw_trajectory_and_rotation_unchanged(self):
        raw = {"traj": np.arange(21, dtype=np.float32).reshape(3, 7), "frame_idx": np.arange(3)}
        original = raw["traj"].copy()
        result = scaled_camera_fields(raw, 2., np.eye(3))
        np.testing.assert_array_equal(result["traj"], original)
        self.assertEqual(result["scale"], 2.)
        self.assertNotIn("scale", raw)


if __name__ == "__main__":
    unittest.main()

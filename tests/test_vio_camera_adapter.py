import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from vio_camera_adapter import adapt_camera_poses


class VioAdapterTest(unittest.TestCase):
    def fixture(self, selected=(0, 1, 2, 3)):
        time = np.array([10., 10.02, 10.06, 10.1])
        transforms = np.tile(np.eye(4), (4, 1, 1))
        transforms[:, :3, :3] = Rotation.from_euler("z", [0, 10, 30, 50], degrees=True).as_matrix()
        transforms[:, 0, 3] = [0, 1, 3, 5]
        ids = np.asarray(selected)
        return time, np.arange(100, 104), {"timestamp_camera": time[ids],
            "frame_index": np.arange(100, 104)[ids], "T_world_camera": transforms[ids]}

    def test_exact_poses_are_not_rescaled_or_shifted(self):
        time, frames, source = self.fixture()
        result = adapt_camera_poses(time, frames, source, np.eye(3))
        np.testing.assert_array_equal(result["T_world_rect"], source["T_world_camera"])
        self.assertEqual(result["scale"], 1)
        self.assertTrue(result["camera_pose_from_source"].all())
        self.assertFalse(result["camera_pose_interpolated"].any())

    def test_gap_uses_timestamp_weight_and_slerp_without_row_shift(self):
        time, frames, source = self.fixture((0, 2, 3))
        result = adapt_camera_poses(time, frames, source, np.eye(3))
        np.testing.assert_allclose(result["traj"][:, 0], [0, 1, 3, 5], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(result["traj"][:, 3:]).as_euler("xyz", degrees=True)[:, 2], [0, 10, 30, 50])
        np.testing.assert_array_equal(result["source_pose_row"], [0, -1, 1, 2])
        np.testing.assert_array_equal(result["interpolation_anchor_frames"][1], [0, 2])
        np.testing.assert_array_equal(result["camera_pose_interpolated"], [False, True, False, False])
        self.assertFalse(result["camera_pose_missing"].any())

    def test_rectification_is_right_multiplied_inverse_rotation_only(self):
        time, frames, source = self.fixture()
        r1 = Rotation.from_euler("x", 23, degrees=True).as_matrix()
        result = adapt_camera_poses(time, frames, source, r1)
        np.testing.assert_allclose(result["T_world_rect"][:, :3, :3], source["T_world_camera"][:, :3, :3] @ r1.T)
        np.testing.assert_array_equal(result["traj"][:, :3], source["T_world_camera"][:, :3, 3])
        point_raw = np.array([.1, .2, 1.])
        np.testing.assert_allclose(result["T_world_rect"][2, :3, :3] @ (r1 @ point_raw), source["T_world_camera"][2, :3, :3] @ point_raw)

    def test_matching_time_wrong_identity_is_rejected(self):
        time, frames, source = self.fixture()
        source["frame_index"][1] = 999
        with self.assertRaisesRegex(ValueError, "frame ID"):
            adapt_camera_poses(time, frames, source, np.eye(3))

    def test_nonisolated_and_endpoint_gaps_are_rejected(self):
        for selected in ((0, 3), (1, 2, 3), (0, 1, 2)):
            time, frames, source = self.fixture(selected)
            with self.assertRaisesRegex(ValueError, "isolated"):
                adapt_camera_poses(time, frames, source, np.eye(3))

    def test_invalid_rotation_is_rejected_not_projected_to_so3(self):
        time, frames, source = self.fixture()
        source["T_world_camera"][0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "proper SE"):
            adapt_camera_poses(time, frames, source, np.eye(3))

    def test_duplicate_timestamps_are_rejected(self):
        time, frames, source = self.fixture()
        source["timestamp_camera"][1] = source["timestamp_camera"][0]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            adapt_camera_poses(time, frames, source, np.eye(3))


if __name__ == "__main__":
    unittest.main()

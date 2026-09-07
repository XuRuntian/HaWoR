import copy
import unittest

from scripts.frontend_comparison_adapter import read_frontend, optimized_groups


def fixture():
    obs = {"candidate_id": 3, "original_candidate_id": 17, "baseline_candidate_id": 3,
           "handedness": "right", "is_right": 1, "official_gate_passed": True,
           "bbox_xyxy": [10, 20, 30, 40], "person_score": 0.9,
           "vitpose_keypoints_2d": [[20, 30, 0.9]] * 21,
           "mano_params": {"must_not_pass": 123}, "crop": {"must_not_pass": 456},
           "pred_cam_t_full": [1, 2, 3]}
    track = {"frame_idx": 0, "track_id": 0, "state": "observed",
             "selected_candidate_id": 17, "vitpose_side": "right", "track_fragment_id": 5,
             "dominant_track_handedness": "left", "handedness_flip": True,
             "identity_switch_suspected": True}
    missing = {**track, "track_id": 1, "state": "missing", "selected_candidate_id": None}
    return {"schema_version": "1.0", "sources": {"video": "/rectified.mp4"},
            "video_info": {"width": 100, "height": 80}, "association_summary": {},
            "frames": [{"frame_idx": 0, "time_s": 0, "observations": [obs],
                        "temporal_tracks": [track, missing]}]}


class FrontendAdapterTest(unittest.TestCase):
    def test_maps_original_id_not_array_position_or_baseline_id(self):
        data = fixture()
        before = copy.deepcopy(data)
        result = read_frontend(data, "/rectified.mp4", (100, 80), 1)
        selected = result["selected"][0]
        self.assertEqual(selected["observation_id"], 17)
        self.assertEqual(selected["baseline_candidate_id"], 3)
        self.assertEqual(selected["physical_track_id"], 0)
        self.assertEqual(selected["hawor_side"], 1)
        self.assertEqual(selected["handedness"], "right")
        self.assertFalse({"mano_params", "crop", "pred_cam_t_full"} & selected.keys())
        self.assertEqual(data, before)
        self.assertEqual(result["states"][1]["state"], "missing")

    def test_unknown_or_disagreeing_side_is_rejected_not_guessed(self):
        data = fixture()
        data["frames"][0]["temporal_tracks"][0]["vitpose_side"] = "left"
        with self.assertRaisesRegex(ValueError, "handedness"):
            read_frontend(data, "/rectified.mp4", (100, 80), 1)

    def test_missing_frames_do_not_become_generated_observations(self):
        result = read_frontend(fixture(), "/rectified.mp4", (100, 80), 1)
        groups = optimized_groups(result["selected"])
        self.assertEqual(list(groups), ["t00_f005_right"])
        self.assertEqual(len(groups["t00_f005_right"]), 1)
        self.assertTrue(groups["t00_f005_right"][0]["det"])

    def test_same_side_physical_fragments_never_merge(self):
        first = read_frontend(fixture(), "/rectified.mp4", (100, 80), 1)["selected"][0]
        second = {**first, "frame_idx": 9, "physical_track_fragment_id": 6}
        third = {**first, "physical_track_id": 1}
        groups = optimized_groups([first, second, third])
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups["t00_f006_right"][0]["frame"], 9)
        self.assertEqual(len(optimized_groups([first, second], {9})), 1)

    def test_wrong_video_geometry_or_timeline_rejected(self):
        for video, size, count in (("/raw.mp4", (100, 80), 1),
                                   ("/rectified.mp4", (80, 100), 1),
                                   ("/rectified.mp4", (100, 80), 2)):
            with self.assertRaises(ValueError):
                read_frontend(fixture(), video, size, count)

    def test_preserves_zero_width_bbox_supported_by_native_square_crop(self):
        data = fixture()
        data["frames"][0]["observations"][0]["bbox_xyxy"] = [10, 20, 10, 40]
        selected = read_frontend(data, "/rectified.mp4", (100, 80), 1)["selected"][0]
        self.assertEqual(selected["bbox_xyxy"], [10, 20, 10, 40])
        self.assertTrue(selected["degenerate_bbox_axis"])

    def test_track_exclusivity_and_missing_id_checked(self):
        data = fixture()
        data["frames"][0]["temporal_tracks"][1] = {
            **data["frames"][0]["temporal_tracks"][0], "track_id": 1}
        with self.assertRaisesRegex(ValueError, "Non-exclusive"):
            read_frontend(data, "/rectified.mp4", (100, 80), 1)
        data = fixture()
        data["frames"][0]["temporal_tracks"][0]["selected_candidate_id"] = 3
        with self.assertRaisesRegex(ValueError, "unresolved"):
            read_frontend(data, "/rectified.mp4", (100, 80), 1)


if __name__ == "__main__":
    unittest.main()

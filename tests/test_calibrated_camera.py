import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate

from lib.datasets.track_dataset import TrackDatasetEval
from lib.models.hawor import HAWOR


class CalibratedCameraTest(unittest.TestCase):
    def dataset(self, flipped, calibrated):
        return TrackDatasetEval(["fixture.jpg"], np.array([[20., 15., 80., 85., .9]]),
                                img_focal=80., img_center=[50.25, 44.5],
                                do_flip=flipped, calibrated_camera=calibrated)

    def item(self, dataset):
        image = np.arange(96 * 128 * 3, dtype=np.uint8).reshape(96, 128, 3)
        with patch("lib.datasets.track_dataset.cv2.imread", return_value=image):
            return dataset[0]

    def test_camera_metadata_does_not_change_crop_pixels(self):
        for flipped in (False, True):
            legacy = self.item(self.dataset(flipped, False))
            calibrated = self.item(self.dataset(flipped, True))
            self.assertTrue(torch.equal(legacy["img"], calibrated["img"]))
            self.assertNotIn("original_img_center", legacy)

    def test_calibrated_mirror_is_repeatable_and_uses_pixel_width(self):
        dataset = self.dataset(True, True)
        first, second = self.item(dataset), self.item(dataset)
        self.assertTrue(torch.equal(first["img"], second["img"]))
        self.assertTrue(torch.equal(first["center"], second["center"]))
        np.testing.assert_allclose(first["center"], [77, 50])
        np.testing.assert_allclose(first["img_center"], [76.75, 44.5])
        np.testing.assert_allclose(first["original_img_center"], [50.25, 44.5])

    def test_real_forward_geometry_restores_original_principal_point(self):
        for flipped in (False, True):
            captured = {}

            def bbox(center, scale, focal, principal):
                captured["conditioning_offset"] = (center - principal).clone()
                return torch.zeros(len(center), 3)

            def head(feature):
                n = len(feature)
                pose = torch.eye(3)[:, :2].reshape(1, 6).repeat(n, 16)
                return pose, torch.zeros(n, 10), torch.tensor([[1., .1, .2]]).repeat(n, 1)

            def project(points, camera, center, scale, focal, principal):
                captured["output_center"] = center.clone()
                captured["output_principal"] = principal.clone()
                return torch.zeros(len(points), 21, 2)

            stub = SimpleNamespace(bbox_est=bbox, backbone=lambda x: x,
                                   st_module=None, motion_module=None, mano_head=head,
                                   pose_num=16, crop_size=256, project=project,
                                   mano=SimpleNamespace(query=lambda out: SimpleNamespace(joints=torch.zeros(1, 21, 3))))
            stub.get_trans = lambda *args: HAWOR.get_trans(stub, *args)
            item = self.item(self.dataset(flipped, True))
            batch = {key: value[None] for key, value in default_collate([item]).items()}
            output = HAWOR.forward_step(stub, batch)
            np.testing.assert_allclose(captured["output_center"], [[50., 50.]])
            np.testing.assert_allclose(captured["output_principal"], [[50.25, 44.5]])
            np.testing.assert_allclose(captured["conditioning_offset"], [[.25 if flipped else -.25, 5.5]])
            expected_x = (-.1 if flipped else .1) + 2 * (50 - 50.25) / 70
            np.testing.assert_allclose(output["out"]["trans_full"][0, 0],
                                       [expected_x, .2 + 2 * 5.5 / 70, 160 / 70], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

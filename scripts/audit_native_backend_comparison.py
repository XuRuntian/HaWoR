"""Independent archive/mask/provenance and sampled video-alignment checks."""

import argparse
from pathlib import Path

import cv2
import numpy as np

import run_native_backend_comparison as backend

base = backend.base


def frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,image = cap.read()
    cap.release()
    assert ok,(str(path),index)
    return image


def audit(out):
    result = {"passed":True,"branches":{},"sampled_video_alignment":[]}
    for branch in ("native","optimized"):
        source = backend.load_predictions(branch)
        destination = out/"predictions"/branch
        completed = np.load(destination/"world_completed.npz")
        observed = np.load(destination/"world_observed.npz")
        mask = completed["observed"]
        assert mask.dtype == bool and np.array_equal(~mask,completed["generated"])
        ids = completed["source_prediction_index"][mask]
        assert (completed["source_prediction_index"][~mask] == -1).all()
        assert len(np.unique(ids)) == len(ids)
        for key in ("frame_idx","side","slot"):
            assert np.array_equal(completed[key][mask],source[key][ids]),(branch,key)
        assert len(observed["frame_idx"]) == len(source["frame_idx"])
        np.testing.assert_array_equal(observed["source_prediction_index"],np.arange(len(source["frame_idx"])))
        if branch == "optimized":
            np.testing.assert_array_equal(np.sort(ids),np.arange(len(source["frame_idx"])))
        error = float(abs(completed["vertices_camera"][mask]-source["vertices"][ids]).max())
        assert error < .0002,(branch,error)
        obs,gen = np.zeros((1800,2),bool),np.zeros((1800,2),bool)
        obs[completed["frame_idx"][mask],completed["slot"][mask]] = True
        gen[completed["frame_idx"][~mask],completed["slot"][~mask]] = True
        assert not (obs & gen).any()
        np.testing.assert_array_equal(obs,completed["observed_mask"])
        np.testing.assert_array_equal(gen,completed["generated_mask"])
        np.testing.assert_array_equal(~(obs|gen),completed["missing_mask"])
        np.testing.assert_array_equal(obs,source["raw_observed_mask"])
        generated_owners_checked = 0
        if branch == "optimized":
            mapping = base.read(destination/"infiller_audit.json")
            parts = {x["ownership_interval"]:x for x in mapping["completion_parts"]}
            for fi,side,slot,part in zip(completed["frame_idx"][~mask],completed["side"][~mask],
                                        completed["slot"][~mask],completed["ownership_interval"][~mask]):
                owner = parts[int(part)]
                assert owner["first"]<=fi<=owner["last"]
                assert fi not in mapping["ambiguities"]["side_conflict_frames"]
                assert owner["owners"][side][0] == slot and owner["owners"][side][2] == side
                generated_owners_checked += 1
        joints = completed["joints_camera"]
        for key in ("vertices","joints_world","vertices_camera","joints_camera"):
            assert np.isfinite(completed[key]).all(),(branch,key)
        nonpositive = (joints[...,2] <= 0).any(axis=-1)
        result["branches"][branch] = {"observed_source_geometry_max_error":error,
            "observed_instances":int(mask.sum()),"generated_instances":int((~mask).sum()),
            "missing_slots":int((~(obs|gen)).sum()),"generated_physical_owners_checked":generated_owners_checked,
            "frames_with_nonpositive_joint_depth":np.unique(completed["frame_idx"][nonpositive]).tolist(),
            "nonpositive_depth_generated_instances":int((nonpositive & ~mask).sum()),
            "interpretation":"Behind-camera predictions are retained and reported, not filtered or counted as observations"}
    for fi in (0,88,114,196,599,974,1258,1545,1780,1799):
        rgb = frame(base.VIDEO,fi)
        comparison = frame(out/"comparison.mp4",fi)
        errors = {"rgb":float(abs(rgb[100:].astype(float)-comparison[100:,:base.WIDTH].astype(float)).mean())}
        for column,branch in enumerate(("native","optimized"),1):
            image = frame(out/f"{branch}_overlay.mp4",fi)
            panel = comparison[:,column*base.WIDTH:(column+1)*base.WIDTH]
            errors[branch] = float(abs(image[100:].astype(float)-panel[100:].astype(float)).mean())
        assert max(errors.values()) < 8,(fi,errors)
        result["sampled_video_alignment"].append({"frame":fi,"mean_absolute_rgb_errors":errors})
    base.dump(out/"independent_audit.json",result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=backend.OUTPUT)
    audit(parser.parse_args().output)

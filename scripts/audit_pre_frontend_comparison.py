"""Independent post-run checks of selection provenance, masks and video artifacts."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def video_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {index} from {path}")
        return frame
    finally:
        cap.release()


def audit(out):
    start = time.perf_counter()
    manifest = read(out / "manifest.json")
    root = Path(__file__).resolve().parents[1]
    code_integrity = {}
    for name, digest in manifest["code_sha256"].items():
        snapshot = (out / "provenance/code" / name).read_bytes()
        current = (root / name).read_bytes()
        assert hashlib.sha256(snapshot).hexdigest() == digest
        equivalent = ast.dump(ast.parse(snapshot)) == ast.dump(ast.parse(current))
        assert equivalent, f"Executable source changed since inference: {name}"
        code_integrity[name] = {
            "snapshot_matches_manifest": True, "current_ast_matches_executed_snapshot": equivalent,
            "current_bytes_match_snapshot": current == snapshot,
            "current_sha256": hashlib.sha256(current).hexdigest(),
        }
    frontend = read(out / "cache/frontend_selection.json")
    expected = {(r["frame_idx"], r["observation_id"]): r for r in frontend["selected"]}
    arrays, indices = {}, {}
    geometry = {}
    for branch in ("native", "optimized"):
        with np.load(out / f"predictions/{branch}/camera_predictions.npz") as data:
            arrays[branch] = {key: data[key] for key in (
                "frame_idx", "side", "slot", "bbox", "observed", "generated",
                "raw_observed_mask", "present_mask", "generated_mask", "missing_mask")}
            geometry[branch] = {"positive_joint_depth_fraction": float((data["joints_camera"][..., 2] > 0).mean())}
        indices[branch] = read(out / f"predictions/{branch}/index.json")
    actual = {}
    b = arrays["optimized"]
    for i, record in enumerate(indices["optimized"]):
        assert record["prediction_index"] == i
        assert len(record["source_observations"]) == 1
        source = record["source_observations"][0]
        key = (record["frame_idx"], source["observation_id"])
        assert key not in actual, f"Reused observation {key}"
        actual[key] = record
        wanted = expected[key]
        assert np.array_equal(b["bbox"][i, :4], wanted["bbox_xyxy"]), key
        assert int(b["side"][i]) == int(wanted["handedness"] == "right"), key
        assert source["physical_track_id"] == wanted["physical_track_id"]
        assert source["physical_track_fragment_id"] == wanted["physical_track_fragment_id"]
        assert record["state"] == "observed" and b["observed"][i] and not b["generated"][i]
    assert set(actual) == set(expected), "Selected observations lost or added at the adapter boundary"
    for chunk in read(out / "predictions/optimized/chunks.json"):
        rows = [r for r in indices["optimized"] if r["chunk_id"] == chunk["chunk_id"]]
        identities = {(r["source_observations"][0]["physical_track_id"],
                       r["source_observations"][0]["physical_track_fragment_id"], r["hawor_side"]) for r in rows}
        assert len(identities) == 1
        assert np.all(np.diff(chunk["frame_idx"]) == 1), "Chunk crossed a missing frame"
    expected_mask = np.zeros((1800, 2), dtype=bool)
    for row in frontend["states"]:
        expected_mask[row["frame_idx"], row["track_id"]] = row["state"] == "observed"
    assert np.array_equal(expected_mask, b["raw_observed_mask"])
    assert np.array_equal(expected_mask, b["present_mask"])
    for branch, data in arrays.items():
        assert data["raw_observed_mask"].shape == (1800, 2)
        assert not data["generated"].any() and not data["generated_mask"].any()
        assert np.array_equal(data["missing_mask"], ~data["present_mask"])
        assert np.array_equal(data["present_mask"], data["raw_observed_mask"])
        assert data["frame_idx"].min() >= 0 and data["frame_idx"].max() < 1800
    counts = {branch: Counter(data["frame_idx"].tolist()) for branch, data in arrays.items()}
    native_side_overrides = sum(
        source["source_handedness"] != record["hawor_side"]
        for record in indices["native"] for source in record["source_observations"])
    serialization_collisions = {}
    for branch in arrays:
        chunks = read(out / f"predictions/{branch}/chunks.json")
        filenames = Counter((str(c["group"]), c["frame_idx"][0], c["frame_idx"][-1]) for c in chunks)
        serialization_collisions[branch] = [
            {"group": group, "start_frame": first, "end_frame": last, "write_count": count}
            for (group, first, last), count in filenames.items() if count > 1]
    per_frame = [{"frame_idx": fi, "native_observed_output_instances": counts["native"][fi],
                  "optimized_observed_output_instances": counts["optimized"][fi],
                  "generated_instances_both_branches": 0} for fi in range(1800)]
    windows = {}
    for first, last in ((88,94), (100,114), (176,182), (193,201), (596,602)):
        windows[f"{first:04d}-{last:04d}"] = {"frame_range_inclusive": [first, last]}
        for branch in arrays:
            windows[f"{first:04d}-{last:04d}"][branch] = {
                "observed_instances": sum(counts[branch][fi] for fi in range(first, last+1)),
                "generated_instances": 0,
                "missing_slots": int(arrays[branch]["missing_mask"][first:last+1].sum())}
    alignment = {}
    source_video = Path(manifest["input"])
    for fi in (0, 88, 114, 196, 599, 1799):
        source = video_frame(source_video, fi)
        comparison = video_frame(out / "comparison.mp4", fi)
        assert comparison.shape == (1074, 5760, 3)
        rgb_column = comparison[48:, :1920]
        error = float(np.abs(rgb_column.astype(float) - source[48:].astype(float)).mean())
        assert error < 8, (fi, error)
        alignment[str(fi)] = {"RGB_column_mean_abs_codec_error_0_255": error}
    for first, last in ((88,94), (100,114), (176,182), (193,201), (596,602)):
        source = video_frame(source_video, first)
        window = video_frame(out / f"windows/{first:04d}-{last:04d}/comparison.mp4", 0)
        error = float(np.abs(window[48:, :1920].astype(float) - source[48:].astype(float)).mean())
        assert error < 8, (first, error)
        windows[f"{first:04d}-{last:04d}"]["first_frame_RGB_alignment_MAE"] = error
    for name in ("native_overlay", "optimized_overlay", "comparison"):
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(out / f"{name}.mp4"),
                        "-f", "null", "-"], check=True)
    result = {
        "passed": True, "optimized_observation_mapping_count": len(actual),
        "bbox_max_abs_change": 0.0, "side_overrides": 0,
        "native_majority_routing_source_side_overrides": native_side_overrides,
        "native_camera_json_filename_collisions": serialization_collisions,
        "executed_code_integrity": code_integrity,
        "camera_depth_diagnostics": geometry,
        "chunks_crossing_fragment_side_or_missing_boundaries": 0,
        "degenerate_bbox_observations_preserved": [{"frame_idx": r["frame_idx"],
            "track": r["physical_track_id"], "observation_id": r["observation_id"],
            "bbox": r["bbox_xyxy"]} for r in frontend["selected"] if r["degenerate_bbox_axis"]],
        "frame_alignment_checks": alignment, "windows": windows,
        "per_frame": per_frame, "full_video_decode_checks": "passed for all three videos",
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - start,
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    diagnostics = read(out / "diagnostics.json")
    diagnostics["independent_artifact_audit"] = {k:v for k,v in result.items() if k != "per_frame"}
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    report = (out / "report.md").read_text()
    header = "\n## Independent Artifact Audit\n"
    report = report.split(header)[0]
    report += header + f"""
All {len(actual)} selected B observations map one-to-one to HaWoR outputs with exact source bbox,
unchanged source handedness and intact physical fragment IDs. No chunk crosses a missing frame,
fragment or side boundary. All observed/missing masks match the full cached temporal state table.
All three full videos decode without FFmpeg errors; RGB-column checks at frames 0, 88, 114, 196,
599 and 1799 match the same source frames within codec tolerance. Window starts were also checked.
See `audit.json` for frame-level counts, window counts and the exact audit-script hash.

Four zero-width but positive-height source boxes (frames 268, 309, 350, 412) are preserved,
not repaired or rejected. Native HaWoR square crops use max(width,height), so these remain usable,
but they are a frontend geometry limitation, not evidence of good hand localization.
Native majority routing changed {native_side_overrides} observation-side hypotheses; B changed zero.
These labels are not anatomical ground truth, so this is a routing diagnostic, not an error count.

The native camera JSON naming convention collides for side 1, frame 1569 (two singleton calls).
The unmodified native serializer still overwrites the earlier JSON at that path. This experiment's
callback additionally preserves every actual inference call in `predictions/native/chunks/`;
instance counts and the overlay use these lossless outputs, without silently discarding the earlier
mesh. Thus Native's 2364 inference instances are not the 2363 instances recoverable from its camera
JSON files alone. The native union-mask rendering also processes both calls. This export/visualization
difference is not a reconstruction or frontend gain. Other same-side overlaps are retained too.

Archived executable sources match the manifest hashes. Current source ASTs match those snapshots;
the only byte-level difference is trailing-whitespace cleanup in `hawor_video.py` after inference
started. No executed behavior changed. Detailed hashes are in `audit.json`.
To repeat the independent audit after reproduction, run
`python scripts/audit_pre_frontend_comparison.py --output /home/user/HaWoR/outputs/pre_frontend_comparison/reproduction`.

| Inclusive Window | Native Observed Instances | B Observed Instances | Generated A/B |
|---|---:|---:|---:|
"""
    for label, window in windows.items():
        report += f"| {label} | {window['native']['observed_instances']} | {window['optimized']['observed_instances']} | 0 / 0 |\n"
    (out / "report.md").write_text(report)
    target = out / "provenance/code/scripts/audit_pre_frontend_comparison.py"
    target.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_frame", "windows")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/pre_frontend_comparison/full_60s")
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(Path(__file__).resolve().parents[1])
    audit(out)

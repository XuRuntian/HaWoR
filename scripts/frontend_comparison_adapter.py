"""Metadata-only adapter. Cached reconstruction/crop fields never cross this boundary."""

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np


def observation_id(observation):
    for key in ("original_candidate_id", "official_candidate_id", "candidate_id"):
        if key in observation:
            return int(observation[key])
    raise ValueError("Observation has no candidate ID")


def read_frontend(payload, video_path, image_size, frame_count):
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported frontend cache schema")
    if payload["sources"]["video"] != str(video_path):
        raise ValueError("Cache references a different input video")
    info = payload["video_info"]
    if (info["width"], info["height"]) != tuple(image_size):
        raise ValueError("Cache coordinates do not match video dimensions")
    frames = payload["frames"]
    if [f["frame_idx"] for f in frames] != list(range(frame_count)):
        raise ValueError("Cache must include every frame without reindexing")
    selected, states = [], []
    conflicts = []
    fragments = defaultdict(list)
    for frame in frames:
        fi = frame["frame_idx"]
        by_id = {}
        for obs in frame["observations"]:
            oid = observation_id(obs)
            if oid in by_id:
                raise ValueError(f"Duplicate observation ID at frame {fi}: {oid}")
            by_id[oid] = obs
        tracks = frame["temporal_tracks"]
        if sorted(t["track_id"] for t in tracks) != [0, 1]:
            raise ValueError(f"Expected two exclusive track states at {fi}")
        used, frame_sides = set(), []
        for track in tracks:
            if track["frame_idx"] != fi or track["state"] not in ("observed", "missing"):
                raise ValueError(f"Invalid track state at {fi}")
            states.append(deepcopy(track))
            if track["state"] == "missing":
                if track["selected_candidate_id"] is not None:
                    raise ValueError("Missing state has an observation")
                continue
            oid = int(track["selected_candidate_id"])
            if oid in used or oid not in by_id:
                raise ValueError(f"Non-exclusive or unresolved observation at {fi}: {oid}")
            used.add(oid)
            obs = by_id[oid]
            side = obs["handedness"]
            if side not in ("left", "right") or side != track["vitpose_side"]:
                raise ValueError(f"Unresolved input handedness at {fi}: {oid}")
            if "is_right" in obs and int(obs["is_right"]) != int(side == "right"):
                raise ValueError(f"Conflicting handedness fields at {fi}: {oid}")
            if not obs.get("official_gate_passed", False):
                raise ValueError("Selected observation did not pass the original gate")
            bbox = np.asarray(obs["bbox_xyxy"], dtype=np.float64)
            # Native crop uses max(width, height), so a zero-width keypoint bbox
            # is still usable when its height is positive. Do not repair/filter it.
            if (bbox.shape != (4,) or not np.isfinite(bbox).all()
                    or np.any(bbox[2:] < bbox[:2]) or np.max(bbox[2:] - bbox[:2]) <= 0):
                raise ValueError(f"Invalid bbox at {fi}: {oid}")
            fragment = track["track_fragment_id"]
            if fragment is None:
                raise ValueError("Observed state lacks a fragment ID")
            # Explicit allowlist: no HaMeR pose, camera, vertices, crop or ego_score.
            row = {
                "frame_idx": fi, "time_s": frame["time_s"],
                "observation_id": oid,
                "baseline_candidate_id": obs.get("baseline_candidate_id"),
                "physical_track_id": int(track["track_id"]),
                "physical_track_fragment_id": int(fragment),
                "handedness": side, "hawor_side": int(side == "right"),
                "bbox_xyxy": bbox.tolist(), "observed": True,
                "degenerate_bbox_axis": bool(np.any(bbox[2:] == bbox[:2])),
                "person_score": obs["person_score"],
                "vitpose_keypoints_2d": deepcopy(obs["vitpose_keypoints_2d"]),
                "dominant_track_handedness_diagnostic_only": track["dominant_track_handedness"],
                "handedness_flip": track["handedness_flip"],
                "identity_switch_suspected": track["identity_switch_suspected"],
            }
            selected.append(row)
            frame_sides.append(side)
            fragments[(row["physical_track_id"], int(fragment))].append(row)
        if len(frame_sides) == 2 and len(set(frame_sides)) == 1:
            conflicts.append(fi)
    mapping = []
    for (track, fragment), rows in sorted(fragments.items()):
        counts = Counter(r["handedness"] for r in rows)
        mapping.append({
            "physical_track_id": track, "fragment_id": fragment,
            "first_frame": min(r["frame_idx"] for r in rows),
            "last_frame": max(r["frame_idx"] for r in rows),
            "observed_count": len(rows), "source_handedness_counts": dict(counts),
            "unique_anatomical_identity_established": False,
            "mixed_source_handedness": len(counts) > 1,
            "routing_rule": "per-observation handedness; fragment and contiguous side runs stay separate",
        })
    return {"selected": selected, "states": states, "fragment_mapping": mapping,
            "same_side_conflict_frames": conflicts,
            "cache_association_summary": deepcopy(payload["association_summary"])}


def optimized_groups(selected, allowed_frames=None):
    groups = defaultdict(list)
    for obs in selected:
        fi = obs["frame_idx"]
        if allowed_frames is not None and fi not in allowed_frames:
            continue
        key = f"t{obs['physical_track_id']:02d}_f{obs['physical_track_fragment_id']:03d}_{obs['handedness']}"
        groups[key].append({
            "frame": fi, "det": True,
            "det_box": np.asarray([obs["bbox_xyxy"] + [obs["person_score"]]], dtype=np.float64),
            "det_handedness": np.asarray([obs["hawor_side"]]),
            "provenance": deepcopy(obs),
        })
    for rows in groups.values():
        rows.sort(key=lambda r: r["frame"])
        if len({r["frame"] for r in rows}) != len(rows):
            raise ValueError("Duplicate frame inside a physical fragment")
    return dict(groups)


def native_groups_for_audit(tracks):
    """Mirror native *metadata routing* only; inference still calls upstream HaWoR."""
    groups = {0: [], 1: []}
    mapping = []
    for track_id, track in tracks.items():
        valid = np.asarray([r["det"] for r in track])
        labels = np.concatenate([r["det_handedness"] for r in track])[valid]
        side = int(float(labels.mean()) >= 0.5)
        mapping.append({"native_track_id": int(track_id), "routed_hawor_side": side,
                        "source_left": int((labels == 0).sum()),
                        "source_right": int((labels == 1).sum()),
                        "first_frame": min(r["frame"] for r in track),
                        "last_frame": max(r["frame"] for r in track)})
        for index, row in enumerate(track):
            item = deepcopy(row)
            item["provenance"] = {
                "frame_idx": int(row["frame"]),
                "observation_id": f"native:{int(track_id)}:{index}",
                "native_track_id": int(track_id), "observed": bool(row["det"]),
                "source_handedness": int(row["det_handedness"][0]),
                "hawor_side": side,
            }
            groups[side].append(item)
    for rows in groups.values():
        rows.sort(key=lambda r: r["frame"])
    return groups, mapping

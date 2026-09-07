"""Identity-aware serialization around the unchanged native HaWoR infiller."""

from collections import defaultdict
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import run_native_backend_comparison as backend

base = backend.base


def ownership_intervals(metadata, first, last):
    """Never infer anatomical side from slot number or bridge fragment/side changes."""
    fragments = defaultdict(list)
    frame_sides = defaultdict(list)
    for row in metadata:
        fi = row["frame_idx"]
        if not first <= fi <= last:
            continue
        source, = row["source_observations"]
        key = (source["physical_track_id"], source["physical_track_fragment_id"])
        fragments[key].append((fi, row["hawor_side"], row["prediction_index"]))
        frame_sides[fi, row["hawor_side"]].append(key)
    owners = [[None, None] for _ in range(last-first+1)]
    collisions = set()
    runs = []
    for key, observations in sorted(fragments.items()):
        observations.sort()
        split = [0] + [i for i in range(1,len(observations)) if observations[i][1] != observations[i-1][1]] + [len(observations)]
        for run, (a, b) in enumerate(zip(split, split[1:])):
            points = observations[a:b]
            side = points[0][1]
            owner = (*key, side, run)
            runs.append({"owner": list(owner), "first": points[0][0], "last": points[-1][0],
                         "observed_frames": [x[0] for x in points]})
            for fi in range(points[0][0], points[-1][0]+1):
                if owners[fi-first][side] is not None:
                    collisions.add(fi)
                owners[fi-first][side] = owner
    collisions.update(fi for (fi, _), keys in frame_sides.items() if len(keys) > 1)
    for fi in collisions:
        owners[fi-first] = [None, None]
    boundaries = [0] + [i for i in range(1,len(owners)) if owners[i] != owners[i-1]] + [len(owners)]
    intervals = []
    for a, b in zip(boundaries, boundaries[1:]):
        own = owners[a]
        eligible = all(x is not None for x in own)
        intervals.append({"first": first+a, "last": first+b-1,
                          "owners": [list(x) if x else None for x in own], "eligible": eligible})
    return intervals, {"side_conflict_frames": sorted(collisions), "side_consistent_fragment_runs": runs}


def serialize_native_input(out, phase, branch, data, ids, start, end, tag, native_chunks=None):
    root = backend.folder(out, phase, branch) / "infiller_inputs" / tag
    root.mkdir(parents=True, exist_ok=True)
    sequence = root / "sequence"
    images = sequence / "extracted_images"
    images.mkdir(parents=True, exist_ok=True)
    for fi in range(start, end+1):
        path = images / f"{fi-start:04d}.jpg"
        if not path.exists():
            path.symlink_to(out / f"cache/frames/{fi:04d}.jpg")
    source_slam = np.load(backend.folder(out, phase, branch) / "slam_scaled.npz")
    selected = np.arange(start,end+1) - backend.frame_ids(phase)[0]
    slam_dir = sequence / "SLAM"
    slam_dir.mkdir(exist_ok=True)
    np.savez(slam_dir / f"hawor_slam_w_scale_0_{end-start+1}.npz",
             traj=source_slam["traj"][selected], scale=source_slam["scale"])
    chunks = defaultdict(list)
    native_paths = {}
    if native_chunks is None:
        groups = []
        for side in (0,1):
            chosen = ids[data["side"][ids] == side]
            groups.append((side, chosen))
    else:
        groups = []
        for chunk in native_chunks:
            chosen = ids[data["chunk_id"][ids] == chunk["chunk_id"]]
            if len(chosen):
                groups.append((chunk["hawor_side"], chosen))
    for side, chosen in groups:
        if not len(chosen):
            continue
        local_frames = data["frame_idx"][chosen] - start
        chunks[side].append(local_frames)
        path = sequence / "cam_space" / str(side) / f"{local_frames[0]}_{local_frames[-1]}.json"
        base.dump(path, {key: data[key][chosen][None].tolist() for key in backend.PARAMETERS})
        native_paths[str(path)] = chosen
    # Match the native JSON overwrite/read order, including any filename collision.
    dense_source = np.full((2,end-start+1), -1, np.int32)
    for side in (0,1):
        for frames in chunks[side]:
            path = sequence / "cam_space" / str(side) / f"{frames[0]}_{frames[-1]}.json"
            chosen = native_paths[str(path)]
            for fi, index in zip(frames, chosen):
                dense_source[side, fi] = index
    args = SimpleNamespace(video_path=str(root / "sequence.mp4"),
                           infiller_weight=str(backend.ROOT / "weights/hawor/checkpoints/infiller.pt"))
    return args, chunks, dense_source, sequence


@torch.no_grad()
def run_original(out, phase, branch, data, ids, start, end, tag, native_chunks=None):
    args, chunks, source, sequence = serialize_native_input(out, phase, branch, data, ids, start, end, tag, native_chunks)
    begin = time.perf_counter()
    # An incomplete adapter attempt must not disguise cache reads as inference time.
    base.seed()
    result = base.hv.hawor_infiller(args, 0, end-start+1, chunks)
    trans, rot, pose, betas, valid = [x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x) for x in result]
    valid = valid.astype(bool)
    assert not ((source >= 0) & ~valid).any()
    for array in (trans,rot,pose,betas):
        assert np.isfinite(array[valid]).all(), tag
    return {"init_trans": trans, "init_root_orient": rot, "init_hand_pose": pose.reshape(2,end-start+1,15,3),
            "init_betas": betas, "valid": valid, "source": source}, time.perf_counter()-begin


def physical_slots(sides, owners):
    values = sides if owners is None else [owners[side][0] for side in sides]
    return np.asarray(values,dtype=np.int64)


@torch.no_grad()
def fill(out, phase, branch, k):
    frames = backend.frame_ids(phase)
    first, last = int(frames[0]), int(frames[-1])
    destination = out / ("predictions" if phase == "full" else "pilot/predictions") / branch
    if (destination / "infiller_audit.json").exists():
        return
    data = backend.load_predictions(branch)
    ids = np.flatnonzero(np.isin(data["frame_idx"],frames))
    with np.load(destination / "world_observed.npz") as archive:
        observed = {key: archive[key] for key in archive.files}
    parts, mapping, runtime = [], [], 0.
    if branch == "native":
        chunks = base.read(backend.SOURCE / "predictions/native/chunks.json")
        result, seconds = run_original(out,phase,branch,data,ids,first,last,"native_full",chunks)
        runtime += seconds
        parts.append((first,last,result,None))
        policy = "Unmodified native whole-sequence two-side infiller, including native JSON/slot overwrite behavior"
        conflicts = {}
    else:
        metadata = base.read(backend.SOURCE / "predictions/optimized/index.json")
        intervals, conflicts = ownership_intervals(metadata, first, last)
        for number, interval in enumerate(intervals):
            start, end = interval["first"], interval["last"]
            chosen = ids[(data["frame_idx"][ids]>=start) & (data["frame_idx"][ids]<=end)]
            # Both anatomical inputs need a known physical owner and observed anchors.
            counts = [int((data["side"][chosen]==side).sum()) for side in (0,1)]
            missing = 2*(end-start+1)-len(chosen)
            interval.update({"observations_by_side": counts, "missing_slots": missing})
            if interval["eligible"] and min(counts)>=2 and missing>0:
                result, seconds = run_original(out,phase,branch,data,chosen,start,end,f"segment_{number:03d}")
                runtime += seconds
                parts.append((start,end,result,interval["owners"]))
                interval["ran_infiller"] = True
            else:
                interval["ran_infiller"] = False
            mapping.append(interval)
        policy = "Original infiller within known two-hand ownership intervals; no fragment/side-boundary crossing or guesses in conflicts"
    output = {key: [] for key in backend.PARAMETERS}
    output.update({key: [] for key in ("frame_idx","side","slot","observed","generated","source_prediction_index","ownership_interval")})
    if branch == "optimized":
        for key in backend.PARAMETERS:
            output[key].append(observed[key])
        for key in ("frame_idx","side","slot","observed","generated","source_prediction_index"):
            output[key].append(observed[key])
        output["ownership_interval"].append(np.full(len(observed["frame_idx"]),-1,np.int32))
    for part, (start,end,result,owners) in enumerate(parts):
        select = result["valid"] if branch == "native" else result["valid"] & (result["source"]<0)
        sides, local = np.nonzero(select)
        source = result["source"][sides,local]
        for key in backend.PARAMETERS:
            output[key].append(result[key][sides,local])
        output["frame_idx"].append(local+start)
        output["side"].append(sides)
        output["slot"].append(physical_slots(sides,owners))
        output["observed"].append(source>=0)
        output["generated"].append(source<0)
        output["source_prediction_index"].append(source)
        output["ownership_interval"].append(np.full(len(local),part,np.int32))
    output = {key: np.concatenate(values) for key,values in output.items()}
    order = np.lexsort((output["side"],output["frame_idx"]))
    output = {key: value[order] for key,value in output.items()}
    vertices = np.empty((len(order),778,3),np.float32)
    joints = np.empty((len(order),21,3),np.float32)
    for side in (0,1):
        chosen = np.flatnonzero(output["side"]==side)
        for start in range(0,len(chosen),128):
            take = chosen[start:start+128]
            params = {key: torch.from_numpy(output[key][take])[None].float() for key in backend.PARAMETERS}
            result = (base.run_mano if side else base.run_mano_left)(params["init_trans"],params["init_root_orient"],
                params["init_hand_pose"],betas=params["init_betas"])
            vertices[take] = result["vertices"][0].cpu().numpy()
            joints[take] = result["joints"][0].cpu().numpy()
    output.update(vertices=vertices,joints_world=joints)
    from lib.eval_utils.custom_utils import load_slam_cam
    rwc,twc,_,_ = load_slam_cam(backend.folder(out,phase,branch)/"slam_scaled.npz")
    local = output["frame_idx"]-first
    output["vertices_camera"] = (np.einsum("nij,nvj->nvi",rwc[local].numpy(),vertices)+twc[local].numpy()[:,None]).astype(np.float32)
    output["joints_camera"] = (np.einsum("nij,nvj->nvi",rwc[local].numpy(),joints)+twc[local].numpy()[:,None]).astype(np.float32)
    valid = np.zeros((base.N,2),bool)
    observed_mask, generated_mask = valid.copy(), valid.copy()
    valid[output["frame_idx"],output["slot"]] = True
    obs = output["observed"]
    observed_mask[output["frame_idx"][obs],output["slot"][obs]] = True
    generated_mask[output["frame_idx"][~obs],output["slot"][~obs]] = True
    output.update(observed_mask=observed_mask,generated_mask=generated_mask,missing_mask=~valid)
    assert np.isfinite(vertices).all()
    lookup = {int(source):i for i,source in enumerate(observed["source_prediction_index"])}
    restored_error = max(float(abs(vertices[i]-observed["vertices"][lookup[int(source)]]).max())
        for i,source in enumerate(output["source_prediction_index"]) if source>=0)
    assert restored_error < .0002, restored_error
    np.savez_compressed(destination/"world_completed.npz",**output)
    base.dump(destination/"infiller_audit.json", {"policy": policy, "seconds": runtime,
        "observed_instances":int(obs.sum()),"generated_instances":int((~obs).sum()),
        "missing_slots":int((~valid)[frames].sum()),"observed_geometry_preservation_error":restored_error,
        "raw_observed_instances":len(observed["frame_idx"]),"intervals":mapping,"ambiguities":conflicts,
        "completion_parts":[{"ownership_interval":i,"first":start,"last":end,"owners":owners}
                            for i,(start,end,_,owners) in enumerate(parts)],
        "observed_geometry_separate_export":"world_observed.npz", "native_function_unchanged":True,
        "native_last_frame_exclusive_end_retained":True})


def camera_relative_world(data, rcw, tcw):
    rotation = rcw[0].T
    return np.einsum("ij,nvj->nvi", rotation, data - tcw[0][None,None])


def virtual_camera(points, width=960, height=720):
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low+high)/2
    radius = max(float(np.linalg.norm(high-low)/2), .3)
    direction = np.array([1.,-.6,-1.4])
    eye = center + direction / np.linalg.norm(direction) * radius * 3.8
    z = (center-eye) / np.linalg.norm(center-eye)
    x = np.cross(z, np.array([0.,-1.,0.]))
    x /= np.linalg.norm(x)
    y = np.cross(z,x)
    rotation = np.stack([x,y,z])
    translation = -rotation @ eye
    intrinsics = np.array([[800.,0,width/2],[0,800.,height/2],[0,0,1]])
    return rotation,translation,intrinsics,{"bounds": [low.tolist(),high.tolist()],"eye":eye.tolist(),
        "target":center.tolist(),"rotation_w2view":rotation.tolist(),"translation_w2view":translation.tolist(),
        "camera_intrinsics":intrinsics.tolist(),"world_alignment":"Each branch's first camera -> identity, SE(3) only; no scale alignment"}


def draw_path(frame, points, rotation, translation, k, current):
    xyz = points[:current+1] @ rotation.T + translation
    good = xyz[:,2] > 0
    uv = xyz[:,:2]/np.maximum(xyz[:,2:],1e-6) @ k[:2,:2].T + k[:2,2]
    uv = np.clip(uv,-100000,100000).astype(np.int32)
    for i in range(1,len(uv)):
        if good[i-1] and good[i]:
            cv2.line(frame,tuple(uv[i-1]),tuple(uv[i]),(74,140,45),2,cv2.LINE_AA)
    if len(uv) and good[-1]:
        cv2.circle(frame,tuple(uv[-1]),5,(50,100,25),-1,cv2.LINE_AA)


def reference_grid(view, rotation, translation, k):
    image = np.full((720,960,3),242,np.uint8)
    low,high = np.asarray(view["bounds"])
    step = max(.1, 10 ** np.floor(np.log10(max(high-low)/8)))
    y = high[1] + .1
    ranges = [np.arange(np.floor(low[a]/step)*step,np.ceil(high[a]/step)*step+step*.5,step) for a in (0,2)]

    def line(a,b,color,thickness=1):
        xyz = np.asarray([a,b]) @ rotation.T + translation
        if (xyz[:,2] <= 0).any():
            return
        uv = xyz[:,:2]/xyz[:,2:] @ k[:2,:2].T + k[:2,2]
        uv = np.clip(uv,-100000,100000).astype(np.int32)
        cv2.line(image,tuple(uv[0]),tuple(uv[1]),color,thickness,cv2.LINE_AA)

    for x in ranges[0]:
        line([x,y,ranges[1][0]],[x,y,ranges[1][-1]],(218,218,218))
    for z in ranges[1]:
        line([ranges[0][0],y,z],[ranges[0][-1],y,z],(218,218,218))
    for axis,color in enumerate(((70,70,190),(70,160,70),(190,90,70))):
        endpoint = np.zeros(3)
        endpoint[axis] = .2
        line([0,0,0],endpoint,color,2)
    view["reference_grid"] = {"spacing_model_units":float(step),"y_model_units":float(y),"not_estimated_ground":True}
    return image


@torch.no_grad()
def mesh_image(frame, vertices, sides, generated, renderer, camera, lights, faces):
    if not len(vertices):
        return frame,0
    colors = []
    for side,gen in zip(sides,generated):
        color = np.asarray(base.COLORS[int(side)])
        if gen:
            color = .35*color + .65
        colors.append(torch.as_tensor(color,dtype=torch.float32,device="cuda").expand(778,3))
    mesh = base.create_meshes([torch.as_tensor(v,dtype=torch.float32,device="cuda") for v in vertices],
                             [faces[int(s)] for s in sides],colors)
    result = renderer.renderer(mesh,cameras=camera,lights=lights,materials=base.Materials(device="cuda",shininess=0))[0].cpu().numpy()
    mask = result[...,3]>0
    rgb = np.clip(result[...,:3]*255,0,255).astype(np.uint8)[...,::-1]
    frame[mask] = np.clip(.35*frame[mask]+.65*rgb[mask],0,255).astype(np.uint8)
    return frame,int(mask.sum())


def title(frame, text, fi, observed=None, generated=None):
    frame = base.annotate(frame,text,fi)
    if observed is not None:
        cv2.putText(frame,f"OBS {observed}  GEN {generated} (pale)",(18,78),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),3,cv2.LINE_AA)
        cv2.putText(frame,f"OBS {observed}  GEN {generated} (pale)",(18,78),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    return frame


def render(out, phase, branch, k):
    from lib.eval_utils.custom_utils import load_slam_cam
    frames = backend.frame_ids(phase)
    dest = out if phase == "full" else out/"pilot"
    sources = out/"predictions" if phase == "full" else out/"pilot/predictions"
    if (dest/"render_audit.json").exists():
        return
    archives, tables, cameras_world, bounds = {},{}, {},[]
    for name in ("native","optimized"):
        _,_,rcw,tcw = load_slam_cam(backend.folder(out,phase,name)/"slam_scaled.npz")
        rcw,tcw = rcw.numpy(),tcw.numpy()
        cameras_world[name] = (tcw-tcw[0]) @ rcw[0]
        bounds.append(cameras_world[name])
        for mode in ("observed","completed"):
            with np.load(sources/name/f"world_{mode}.npz") as data:
                arrays = {key:data[key] for key in ("vertices","frame_idx","side","generated","observed")}
                if mode == "completed":
                    arrays["vertices_camera"] = data["vertices_camera"]
                wrists = data["joints_world"][:,0]
            arrays["aligned_vertices"] = camera_relative_world(arrays["vertices"],rcw,tcw)
            archives[name,mode] = arrays
            table = defaultdict(list)
            for i,fi in enumerate(arrays["frame_idx"]):
                table[int(fi)].append(i)
            tables[name,mode] = table
            if mode == "observed":
                bounds.append((wrists-tcw[0]) @ rcw[0])
    vr,vt,vk,view_info = virtual_camera(np.concatenate(bounds))
    background = reference_grid(view_info,vr,vt,vk)
    world_renderer,_,world_lights,world_faces = backend.render_setup(vk,960,720)
    world_camera,_ = world_renderer.create_camera_from_cv(torch.tensor(vr,dtype=torch.float32,device="cuda")[None],
        torch.tensor(vt,dtype=torch.float32,device="cuda")[None],K=torch.tensor(vk,dtype=torch.float32,device="cuda")[None])
    renderer,camera,lights,faces = backend.render_setup(k)
    writers = {"native":base.VideoWriter(dest/"native_overlay.mp4",base.WIDTH,base.HEIGHT),
               "optimized":base.VideoWriter(dest/"optimized_overlay.mp4",base.WIDTH,base.HEIGHT),
               "comparison":base.VideoWriter(dest/"comparison.mp4",base.WIDTH*3,base.HEIGHT),
               "observed":base.VideoWriter(dest/"world_observed_comparison.mp4",1920,720),
               "completed":base.VideoWriter(dest/"world_completed_comparison.mp4",1920,720)}
    audit = {"seconds":None,"view":view_info,"frame_count":len(frames),"mesh_pixels":{}}
    for name in ("native","optimized"):
        for mode in ("camera","observed","completed"):
            audit["mesh_pixels"][f"{name}_{mode}"] = []
    cap = cv2.VideoCapture(str(base.VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0]))
    start = time.perf_counter()
    try:
        for local,fi in enumerate(frames):
            ok,rgb = cap.read()
            assert ok and rgb.shape[:2] == (base.HEIGHT,base.WIDTH), int(fi)
            overlays,world_views = {},{"observed":[],"completed":[]}
            for name in ("native","optimized"):
                data = archives[name,"completed"]
                ids = tables[name,"completed"][int(fi)]
                image,count = mesh_image(rgb.copy(),data["vertices_camera"][ids],data["side"][ids],data["generated"][ids],
                                       renderer,camera,lights,faces)
                audit["mesh_pixels"][f"{name}_camera"].append(count)
                text = "Native frontend + native backend" if name == "native" else "Optimized + native backend (identity-bounded fill)"
                overlays[name] = title(image,text,int(fi),int(data["observed"][ids].sum()),int(data["generated"][ids].sum()))
                writers[name].write(overlays[name])
                for mode in ("observed","completed"):
                    data = archives[name,mode]
                    ids = tables[name,mode][int(fi)]
                    image = background.copy()
                    image,count = mesh_image(image,data["aligned_vertices"][ids],data["side"][ids],data["generated"][ids],
                        world_renderer,world_camera,world_lights,world_faces)
                    draw_path(image,cameras_world[name],vr,vt,vk,local)
                    audit["mesh_pixels"][f"{name}_{mode}"].append(count)
                    text = f"{name.title()} | world {mode}"
                    if name == "optimized" and mode == "completed":
                        text = "Optimized | identity-bounded fill"
                    image = title(image,text,int(fi),int(data["observed"][ids].sum()),int(data["generated"][ids].sum()))
                    cv2.putText(image,"Camera path: green | shared fixed view | SE(3) origin only",(16,701),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(40,40,40),1,cv2.LINE_AA)
                    cv2.putText(image,f"Reference grid: {view_info['reference_grid']['spacing_model_units']:g} model units (not ground)",(16,677),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(80,80,80),1,cv2.LINE_AA)
                    world_views[mode].append(image)
            comparison = np.concatenate([title(rgb,"RGB",int(fi)),overlays["native"],overlays["optimized"]],axis=1)
            writers["comparison"].write(comparison)
            for mode in world_views:
                writers[mode].write(np.concatenate(world_views[mode],axis=1))
            if fi in (88,91,100,107,114,179,196,599,974,1258,1545,1780):
                screenshots = dest/"screenshots"
                screenshots.mkdir(exist_ok=True)
                cv2.imwrite(str(screenshots/f"comparison_{fi:06d}.jpg"),comparison)
                cv2.imwrite(str(screenshots/f"world_{fi:06d}.jpg"),np.concatenate(world_views["completed"],axis=1))
            if local % 100 == 0:
                print(f"RENDER {phase} {fi}/{frames[-1]}",flush=True)
    finally:
        cap.release()
        for writer in writers.values():
            writer.close()
    audit["seconds"] = time.perf_counter()-start
    files = ["native_overlay.mp4","optimized_overlay.mp4","comparison.mp4","world_observed_comparison.mp4","world_completed_comparison.mp4"]
    audit["videos"] = {name:base.probe(dest/name) for name in files}
    for name,probe in audit["videos"].items():
        assert int(probe["streams"][0]["nb_frames"]) == len(frames),name
        assert probe["streams"][0]["avg_frame_rate"] == "30/1",name
        subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(dest/name),"-f","null","-"],check=True)
    if phase == "full":
        for first,last in base.WINDOWS:
            window = dest/f"windows/{first:04d}-{last:04d}"
            window.mkdir(parents=True,exist_ok=True)
            for name in files:
                subprocess.run(["ffmpeg","-v","error","-n","-i",str(dest/name),"-vf",
                    f"trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS","-an","-c:v","libx264",
                    "-threads","4","-preset","fast","-crf","20",str(window/name)],check=True)
                assert int(base.probe(window/name)["streams"][0]["nb_frames"]) == last-first+1
    base.dump(dest/"render_audit.json",audit)


def report(out, phase, branch, k):
    from audit_native_backend_comparison import audit
    independent = audit(out)
    environment = out/"provenance/environment"
    environment.mkdir(parents=True,exist_ok=True)
    (environment/"pip-freeze.txt").write_text(subprocess.check_output([backend.sys.executable,"-m","pip","freeze"],text=True))
    (environment/"git-status.txt").write_text(subprocess.check_output(["git","status","--short"],text=True))
    (environment/"tracked-interface.patch").write_text(subprocess.check_output(["git","diff"],text=True))
    (environment/"submodules.txt").write_text(subprocess.check_output(["git","submodule","status"],text=True))
    test_command = [backend.sys.executable,"-m","unittest","discover","-s","tests","-v"]
    tests = subprocess.run(test_command,cwd=backend.ROOT,capture_output=True,text=True)
    (environment/"unittest.log").write_text(tests.stdout+tests.stderr)
    assert tests.returncode == 0, "Focused unit tests failed; see provenance/environment/unittest.log"
    manifest = base.read(out/"manifest.json")
    protected = manifest["protected_sha256"]
    for path,digest in protected.items():
        assert base.sha(path) == digest,path
    branches = {}
    for name in ("native","optimized"):
        predictions = out/"predictions"/name
        cache = backend.folder(out,"full",name)
        branches[name] = {"droid":base.read(cache/"droid_audit.json"),"metric3d":base.read(cache/"scale_audit.json"),
            "world":base.read(predictions/"world_validation.json"),"infiller":base.read(predictions/"infiller_audit.json"),
            "mask":base.read(cache/"mask_audit.json")}
    from scipy.spatial.transform import Rotation
    aligned = {}
    for name in ("native","optimized"):
        data = np.load(backend.folder(out,"full",name)/"slam_scaled.npz")
        xyz = data["traj"][:,:3]*data["scale"]
        rotation = Rotation.from_quat(data["traj"][:,3:]).as_matrix()
        aligned[name] = ((xyz-xyz[0]) @ rotation[0],rotation[0].T @ rotation)
        branches[name]["trajectory"] = {"path_length_model_m":float(np.linalg.norm(np.diff(xyz,axis=0),axis=-1).sum()),
            "net_displacement_model_m":float(np.linalg.norm(xyz[-1]-xyz[0])),
            "interpretation":"Metric3D-scaled trajectory diagnostics, not ground-truth accuracy"}
    position_change = np.linalg.norm(aligned["native"][0]-aligned["optimized"][0],axis=-1)
    relative_rotation = aligned["native"][1].transpose(0,2,1) @ aligned["optimized"][1]
    rotation_change = np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude())
    trajectory_comparison = {"interpretation":"Between-branch differences after first-pose SE(3) alignment, not errors",
        "position_change_model_m":{"median":float(np.median(position_change)),"p95":float(np.percentile(position_change,95)),"max":float(position_change.max())},
        "rotation_change_degrees":{"median":float(np.median(rotation_change)),"p95":float(np.percentile(rotation_change,95)),"max":float(rotation_change.max())}}
    rendering = base.read(out/"render_audit.json")
    base.dump(out/"diagnostics.json",{"passed":True,"branches":branches,"rendering":rendering,
              "protected_files_unchanged":len(protected),"frames":1800,"fps":30,
              "trajectory_comparison":trajectory_comparison,
              "independent_audit":independent,
              "unit_tests":{"command":test_command,"exit_code":tests.returncode,"log":str(environment/"unittest.log")},
              "interpretation":"Frontend plus necessary fragment-aware infiller adaptation; not a pure detector ablation or an accuracy claim"})
    a,b = branches["native"],branches["optimized"]
    rows = []
    for label,key in [("Lossless pre-infiller observed instances","raw_observed_instances"),
                      ("Post-interface observed output instances","observed_instances"),("Generated output instances","generated_instances"),
                      ("Missing slots","missing_slots"),("Infiller seconds","seconds")]:
        rows.append(f"| {label} | {a['infiller'][key]} | {b['infiller'][key]} |")
    for label,stage,key in [("DROID keyframes","droid","keyframes"),("DROID seconds","droid","seconds"),
                            ("Median metric scale","metric3d","median_scale"),("Metric3D + scale seconds","metric3d","seconds")]:
        rows.append(f"| {label} | {a[stage][key]} | {b[stage][key]} |")
    for label,key in [("Scaled camera path length","path_length_model_m"),("Scaled camera net displacement","net_displacement_model_m")]:
        rows.append(f"| {label} | {a['trajectory'][key]} | {b['trajectory'][key]} |")
    text = f"""# Calibrated Native Backend Frontend Comparison

Completed the same rectified frames 0..1799, 1920x1074, 30 FPS. Both branches use K={k.tolist()}.
The original frontend and optimized frontend camera predictions are reused exactly from `{backend.SOURCE}`.
Native masked DROID, Metric3D scale recovery, world conversion and the native infiller were actually executed.
No external VIO, FoundationStereo, hand-depth correction, extra pose filter, gate, network or weight modification.

## What Is Compared

A: native frontend -> calibrated HaWoR camera predictions -> native per-hand union masks -> native
masked DROID -> Metric3D and native scale fit -> original whole-sequence two-side infiller.

B: optimized physical-fragment frontend -> calibrated HaWoR predictions -> the same mask procedure,
DROID and Metric3D algorithms -> the ORIGINAL infiller within explicitly known ownership intervals.
All observed B outputs are preserved. Same-side conflicts, fragment boundaries and side changes are
not silently merged. Generated predictions carry separate masks and their physical owner.

This is a full-backend experiment WITH AN EXPLICIT B INFILLER INTERFACE ADAPTATION, not an unmodified
two-anatomical-track B demo. Native A keeps original slot/JSON overwrite semantics. Lossless pre-infiller
observations remain in `world_observed.npz`; do not confuse their count with dense anatomical slots.
A's slots are anatomical-side hypotheses; B's are anonymous physical slots. Neither is a common
ground-truth identity set, and availability differences are not detection recall.
`world_observed_comparison.mp4` excludes completion effects; `world_completed_comparison.mp4` includes them.

## Native Behavior and Necessary Interfaces

- Calibrated fx/fy/cx/cy enter both DROID and Metric3D, not only hand inference.
- DROID uses each branch's own native hand masks. Therefore frontend effects also propagate through
  dynamic-region masking, keyframe selection, camera trajectory and scale estimation.
- Native DROID RGB resize/crop, mask resizing, Metric3D direct depth resizing and scale hyperparameters
  are preserved. Their existing small resize-versus-crop discrepancy is not silently fixed here.
- Scale estimation retains native threshold expansion and median aggregation; a 100-retry finite-positive
  guard fails explicitly instead of hanging forever. Per-keyframe retries and thresholds are recorded.
- No hand observations are promoted from missing to observed by the infiller. Native's exclusive end-index
  behavior is retained and generated availability is not detection recall.
- B completion has the same weights, 120-frame horizon, native preprocessing and inference. Its available
  context is restricted by ownership, so completion differences cannot be attributed to detection alone.
- B's four same-side observed conflicts are frames 1258, 1346, 1348 and 1545. The ownership-conflict
  exclusion also includes frame 1347 between observations; that is not a fifth observed conflict.

## Results

| Quantity | Native A | Optimized B |
|---|---:|---:|
{chr(10).join(rows)}

All world transforms passed camera->world->camera and native MANO-parameter reconstruction checks.
Observed geometry is preserved through completion; errors are in `diagnostics.json`. These are numerical
consistency checks, not pose/depth/trajectory ground-truth accuracy. DROID/Metric3D scale remains estimated.
Raw median scale factors are attached to different DROID reconstructions and should not be compared as
standalone depth-model accuracy. Trajectory differences after first-pose rigid alignment are recorded
separately in diagnostics; no fitted scale is used to hide their differences.

## Known Generated-Pose Issues

Native completed outputs have nonpositive camera-space joint depth at frames
`{independent['branches']['native']['frames_with_nonpositive_joint_depth']}`; optimized outputs at
`{independent['branches']['optimized']['frames_with_nonpositive_joint_depth']}`. These generated outputs
are retained and flagged, not filtered, treated as detections, or claimed to be accurate. Finite world
coordinates and valid camera transformations do not establish plausible hand completion. Existing
frontend side ambiguity and degenerate/off-image observations also remain unchanged.

## Viewing

`comparison.mp4`: RGB / Native full backend / Optimized backend with the documented identity adapter.
`native_overlay.mp4` and `optimized_overlay.mp4`: original video dimensions, common shading and opacity.
Observed hands use pink/cyan; generated hands use pale variants. Headers report separate OBS/GEN counts.
`world_observed_comparison.mp4` and `world_completed_comparison.mp4`: Native left, Optimized right.
Both use a shared fixed virtual camera; each branch's first camera defines its origin using SE(3) only.
No scale alignment, dynamic view tracking or external world registration is applied. Green is camera path.
Virtual bounds derive from all observed wrists and camera poses, not from generated outliers.
The displayed reference grid is a coordinate aid, not an estimated physical floor.
Five inclusive windows 88-94, 100-114, 176-182, 193-201, 596-602 contain all five videos.

All full videos passed frame-count/FPS checks and complete FFmpeg decoding. Rendering took
{rendering['seconds']:.3f} seconds before validation/window exports. All {len(protected)} protected
old-result, source, foreign-cache and model files retained their pre-run SHA256.
Source snapshots for each executed stage are under `provenance/`; process logs and commands under `logs/`.
The focused unit suite passed; its complete output is `provenance/environment/unittest.log`.
The first full B export encountered an empty generated-index dtype error in the new adapter.
The adapter was corrected and its empty-output case regression-tested; B's original infiller was
actually rerun, not replaced with cached timing. Failed and successful attempt logs are retained.

## Reproduce

Use the configured hawor environment and the unchanged calibrated camera archives; use a fresh output.

```bash
conda activate hawor
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage prepare
python scripts/run_native_backend_stages.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --phase pilot
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage render --phase pilot
# Inspect both pilot world validations, infiller audits and videos before full inference.
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage validate
python scripts/run_native_backend_stages.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --phase full
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage render
python scripts/run_native_backend_comparison.py --output "$PWD/outputs/pre_frontend_comparison/backend_reproduction" --stage report
mpv --speed=0.5 '{out}/comparison.mp4'
```
"""
    (out/"report.md").write_text(text)

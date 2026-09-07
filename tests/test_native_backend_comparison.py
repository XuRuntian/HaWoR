import sys
from pathlib import Path
import unittest
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from native_backend_infilling import ownership_intervals, serialize_native_input, physical_slots


def row(frame,slot,fragment,side,index=0):
    return {"frame_idx":frame,"hawor_side":side,"prediction_index":index,
            "source_observations":[{"physical_track_id":slot,"physical_track_fragment_id":fragment}]}


class OwnershipTest(unittest.TestCase):
    def test_no_generated_rows_keep_integer_slot_indices(self):
        empty = physical_slots(np.array([],dtype=np.int64),[[1,0,0,0],[0,0,1,0]])
        self.assertEqual(empty.dtype,np.int64)
        slots = np.concatenate([np.array([0,1],dtype=np.int8),empty])
        mask = np.zeros((1,2),bool)
        mask[np.array([0,0]),slots] = True
        self.assertTrue(mask.all())

    def test_slot_number_does_not_imply_side(self):
        rows = [row(f,1,10,0) for f in (0,2)] + [row(f,0,20,1) for f in (0,2)]
        intervals,_ = ownership_intervals(rows,0,2)
        self.assertEqual(len(intervals),1)
        self.assertTrue(intervals[0]["eligible"])
        self.assertEqual(intervals[0]["owners"][0][0],1)
        self.assertEqual(intervals[0]["owners"][1][0],0)

    def test_fragment_gap_is_not_bridged(self):
        rows = [row(f,0,0,0) for f in (0,1)] + [row(f,0,1,0) for f in (3,4)] + [row(f,1,0,1) for f in (0,4)]
        intervals,_ = ownership_intervals(rows,0,4)
        middle = [x for x in intervals if x["first"]<=2<=x["last"]][0]
        self.assertFalse(middle["eligible"])

    def test_side_flip_splits_even_within_fragment(self):
        rows = [row(0,0,0,0),row(1,0,0,0),row(2,0,0,1),row(3,0,0,1)]
        intervals,audit = ownership_intervals(rows,0,3)
        self.assertEqual(len(audit["side_consistent_fragment_runs"]),2)
        self.assertEqual(len(intervals),2)

    def test_same_side_conflict_disables_both_inputs(self):
        rows = [row(f,0,0,0) for f in (0,2)] + [row(1,1,0,0)]
        intervals,audit = ownership_intervals(rows,0,2)
        self.assertEqual(audit["side_conflict_frames"],[1])
        middle = [x for x in intervals if x["first"]==1][0]
        self.assertEqual(middle["owners"],[None,None])

    def test_native_duplicate_json_uses_last_prediction_without_losing_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            cache = out/"cache/full/native"
            cache.mkdir(parents=True)
            traj = np.zeros((1800,7),np.float32)
            traj[:,6] = 1
            np.savez(cache/"slam_scaled.npz",traj=traj,scale=1.)
            data = {"frame_idx":np.array([1,1]),"side":np.array([0,0]),"chunk_id":np.array([0,1]),
                "init_root_orient":np.tile(np.eye(3),(2,1,1)),"init_hand_pose":np.tile(np.eye(3),(2,15,1,1)),
                "init_trans":np.array([[1.,0.,1.],[2.,0.,1.]]),"init_betas":np.zeros((2,10))}
            chunks = [{"chunk_id":0,"hawor_side":0},{"chunk_id":1,"hawor_side":0}]
            _,groups,source,_ = serialize_native_input(out,"full","native",data,np.array([0,1]),0,2,"test",chunks)
            self.assertEqual(len(groups[0]),2)
            self.assertEqual(source[0,1],1)
            np.testing.assert_array_equal(data["init_trans"][:,0],[1.,2.])


if __name__ == "__main__":
    unittest.main()

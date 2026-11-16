"""Device-v7 output equality plus actual candidate-mapping branch audits."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch=None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'requires CUDA')
class GsplatCandidateCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gsplat_mixed_ext import load
        from tests.test_gsplat_tensor_v7_cuda import LegacyV7
        cls.ca=load('tensor_v7_ca');cls.base=load('tensor_v7');cls.legacy=LegacyV7()

    def inputs(self,rows=139,counts=(3,7,7,0,9,4)):
        from tests import test_gsplat_tensor_v7_cuda as sample
        r,d=sample.GsplatTensorV7Test.inputs(self,n=73,rows=rows)
        offsets=[];ids=[]
        for count in counts:
            offsets.append(len(ids));ids.extend(range(count))
        r[4]=torch.tensor(offsets,device='cuda',dtype=torch.int32).reshape(1,2,3)
        r[5]=torch.tensor(ids,device='cuda',dtype=torch.int32)
        return r,d

    def compare(self,rows=139,counts=(3,7,7,0,9,4),blocks=6,enabled=True):
        from gaussian_renderer.gsplat_pairing import candidate_pairing_plan
        r,d=self.inputs(rows,counts)
        expected=self.base.forward(r,d,37,29,blocks)
        result=self.ca.forward(r,d,37,29,blocks,2,True,enabled)
        for index in (0,1,2,3,4,5,6,7): self.assertTrue(torch.equal(result[index],expected[index]))
        for a,b in zip(result[3:8],self.legacy(d)): self.assertTrue(torch.equal(a,b))
        physical=blocks or 6
        reference=candidate_pairing_plan(counts,(rows+15)//16,physical,enabled)
        audit=result[9].cpu().tolist()
        self.assertEqual([row[0] for row in audit],[len(t) for t in reference['tiles']])
        self.assertEqual([row[1] for row in audit],reference['extra_ranks'])
        self.assertEqual([bool(row[2]) for row in audit],[reference['active']]*physical)

    def test_actual_candidate_branch_and_group_winners(self):
        with torch.no_grad(): self.compare()

    def test_ties_and_empty_candidate_ranges(self):
        with torch.no_grad():
            self.compare(counts=(7,7,0,0,9,9))
            self.compare(counts=(0,0,0,0,0,0))

    def test_defaults_partial_rows_and_fallback_grids(self):
        with torch.no_grad():
            for rows in (0,1,96,97,139,176,192,209):
                for blocks in (0,1,4,6): self.compare(rows=rows,blocks=blocks)

    def test_same_binary_pairing_disabled(self):
        with torch.no_grad(): self.compare(enabled=False)

    def test_graph_replay_preserves_mapping(self):
        with torch.no_grad():
            r,d=self.inputs()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            owner=torch.cuda.CUDAGraph()
            with torch.cuda.graph(owner,stream=stream):
                outputs=self.ca.forward(r,d,37,29,6,2,True,True)
            with torch.cuda.stream(stream): owner.replay()
            stream.synchronize()
            self.assertEqual(outputs[9][:,1].cpu().tolist(),[-1,0,1,-1,2,-1])
            reference=self.base.forward(r,d,37,29,6)
            for a,b in zip(outputs[3:8],reference[3:8]): self.assertTrue(torch.equal(a,b))


if __name__=='__main__': unittest.main()

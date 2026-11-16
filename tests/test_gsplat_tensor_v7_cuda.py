"""Verify gsplat's device-v7 against the independent device-v7 MLP reference."""
import ctypes
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch=None


class LegacyV7:
    def __init__(self,path=None):
        path=(path or os.environ.get('GSPLAT_V7_REFERENCE_LIBRARY')
              or os.environ.get('GSPLAT_LEGACY_V7_LIBRARY')
              or Path(__file__).resolve().parent / 'cuda/libdeform_v7_reference.so')
        if not Path(path).is_file():
            raise RuntimeError('Build the device-v7 reference library as described in tests/README.md: '+str(path))
        self.library=ctypes.CDLL(str(path))
        self.function=self.library.deform_matched_launch
        self.function.argtypes=[ctypes.c_void_p]*12+[ctypes.c_int,ctypes.c_size_t]
        self.function.restype=ctypes.c_int

    def __call__(self,inputs):
        rows=inputs[0].shape[0]
        outputs=[torch.empty((rows,width),device=inputs[0].device) for width in (3,3,4,1,48)]
        if rows:
            status=self.function(*[x.data_ptr() for x in [*inputs,*outputs]],rows,
                                 torch.cuda.current_stream().cuda_stream)
            if status: raise RuntimeError('legacy MLP launch failed: '+str(status))
        return outputs


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'requires CUDA')
class GsplatTensorV7Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gsplat_mixed_ext import load
        cls.ext=load('tensor_v7')
        cls.legacy=LegacyV7()

    def inputs(self,**kwargs):
        from tests import test_gsplat_mixed_cuda as samples
        from tacker_4dgs_head import pack_deform_tail_weights_v3
        r,d=samples.GsplatMixedCudaTest.inputs(self,**kwargs)
        d[5]=pack_deform_tail_weights_v3(d[5])
        return r,d

    def compare(self,**kwargs):
        from gsplat.cuda._wrapper import rasterize_to_pixels
        r,d=self.inputs(**kwargs)
        image,alpha=rasterize_to_pixels(*r[:4],37,29,16,r[4],r[5],backgrounds=r[6])
        expected=self.legacy(d)
        for actual,reference in zip(self.ext.mlp_forward(d),expected):
            self.assertTrue(torch.equal(actual,reference))
        for blocks in (0,1,4):
            result=self.ext.forward(r,d,37,29,blocks,2,True)
            self.assertTrue(torch.equal(result[0],image))
            self.assertTrue(torch.equal(result[1],alpha))
            for actual,reference in zip(result[3:8],expected):
                self.assertTrue(torch.equal(actual,reference))

    def test_odd_shapes_grid_stride_and_legacy_equality(self):
        with torch.no_grad(): self.compare()

    def test_early_termination(self):
        with torch.no_grad(): self.compare(opaque=True)

    def test_empty_gaussians_and_zero_rows(self):
        with torch.no_grad():
            for n,rows in ((0,0),(0,17),(73,0)):
                self.compare(n=n,rows=rows)

    def test_partial_mlp_tiles(self):
        with torch.no_grad():
            for rows in (1,16,17,129):
                _,d=self.inputs(n=0,rows=rows)
                reference=self.legacy(d)
                for blocks in (0,1,4):
                    for a,b in zip(self.ext.mlp_forward(d,blocks),reference):
                        self.assertTrue(torch.equal(a,b))

    def test_graph_replay_and_resource_controls(self):
        with torch.no_grad():
            r,d=self.inputs()
            reference=self.ext.forward(r,d,37,29)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            for mode in (0,1,2,3,4,5):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    result=self.ext.forward(r,d,37,29,1,mode)
                with torch.cuda.stream(stream): graph.replay()
                stream.synchronize()
                if mode not in (1,5):
                    self.assertTrue(torch.equal(result[0],reference[0]))
                if mode not in (0,4):
                    for a,b in zip(result[3:8],reference[3:8]): self.assertTrue(torch.equal(a,b))

    def test_dtype_and_inference_guards(self):
        with torch.no_grad():
            r,d=self.inputs()
            bad=list(d); bad[5]=bad[5].float()
            with self.assertRaisesRegex(RuntimeError,'dtype/layout'): self.ext.mlp_forward(bad)
            with self.assertRaisesRegex(RuntimeError,'dtype/layout'): self.ext.forward(r,bad,37,29)
        with torch.enable_grad():
            with self.assertRaisesRegex(RuntimeError,'inference-only'): self.ext.mlp_forward(d)


if __name__=='__main__': unittest.main()

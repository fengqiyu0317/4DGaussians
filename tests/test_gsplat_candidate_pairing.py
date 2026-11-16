import importlib.util
from pathlib import Path
import unittest

source=Path(__file__).resolve().parents[1]/'gaussian_renderer/gsplat_pairing.py'
spec=importlib.util.spec_from_file_location('candidate_pairing_reference',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plan=module.candidate_pairing_plan


class CandidatePairingTest(unittest.TestCase):
    def test_group_maximum_and_complete_coverage(self):
        result=plan([3,7,7,0,9,4],9)
        self.assertTrue(result['active'])
        self.assertEqual(result['tiles'],[[0],[1,6],[2,7],[3],[4,8],[5]])
        self.assertEqual(result['extra_ranks'],[-1,0,1,-1,2,-1])

    def test_ties_choose_first_tile(self):
        result=plan([5]*6,9)
        self.assertEqual(result['extra_ranks'],[0,-1,1,-1,2,-1])

    def test_fallbacks_cover_every_tile_exactly_once(self):
        for deform in (0,1,6,7,9,12,19):
            for blocks in (1,3,6,10):
                for enabled in (False,True):
                    result=plan([0,5,1,4,3,2],deform,blocks,enabled)
                    assigned=[tile for worker in result['tiles'] for tile in worker]
                    self.assertEqual(sorted(assigned),list(range(deform)))
                    self.assertEqual(len(assigned),len(set(assigned)))
                    self.assertEqual(result['active'],enabled and blocks==6 and 6<deform<12)

    def test_invalid_inputs(self):
        for args in (([],0),([-1],1),([1],-1),([1],1,0)):
            with self.assertRaises(ValueError): plan(*args)


if __name__=='__main__': unittest.main()

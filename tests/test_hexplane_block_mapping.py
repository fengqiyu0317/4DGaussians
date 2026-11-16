"""Execute the actual host/device mapper without needing CUDA or PyTorch."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class BlockMappingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "C++ compiler required")
    def test_task_ownership_is_complete_disjoint_and_ordered_within_each_domain(self):
        include = Path(__file__).resolve().parents[1] / "tacker_ext/include"
        source = r'''
#include "hexplane_sample_args.h"
#include <cassert>
#include <climits>
int main() {
    for (int samples : {0, 1, 2, 12, 436, 5232}) {
        for (int raster : {0, 1, 3, 1954, 5440}) {
            int next_sample = 0, next_raster = 0;
            for (int block = 0; block < samples + raster; ++block) {
                auto task = tacker_4dgs::hexplane_block_task(block, samples, raster);
                assert(task.index == (task.sampling ? next_sample++ : next_raster++));
            }
            assert(next_sample == samples && next_raster == raster);
        }
    }
    auto task = tacker_4dgs::hexplane_block_task(INT_MAX - 1, INT_MAX - 10, 10);
    assert(task.sampling && task.index == INT_MAX - 11);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "mapping.cpp").write_text("#include <initializer_list>\n" + source)
            subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(include),
                            str(path / "mapping.cpp"), "-o", str(path / "mapping")], check=True)
            subprocess.run([str(path / "mapping")], check=True)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Build a setup-only gsplat projector that also exports off-screen geometry."""
import difflib
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys


def main(build, output):
    obj = 'ProjectionEWA3DGSFused.cuda.o'
    compile_command = shlex.split(subprocess.check_output(
        ['ninja', '-t', 'commands', obj], cwd=build, text=True))
    original_path = Path(compile_command[compile_command.index('-c') + 1])
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == (
        'b8537134770e4e098d6d8616164a20d3bfe34dbf0c4c1dffee42f35cc3c3710e')
    original = original_path.read_text().split(
        'template <typename scalar_t>\n__global__ void projection_ewa_3dgs_fused_bwd_kernel', 1)[0]
    start = original.index('    // mask out gaussians outside the image region')
    end = original.index('    // write to outputs', start)
    edited = (original[:start] + original[end:]).replace(
        'projection_ewa_3dgs_fused_fwd_kernel', 'projection_tile_ablation_fwd_kernel')
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'project.cu'
    (output / 'project.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), edited.splitlines(keepends=True),
        fromfile=original_path.name, tofile=source.name)))
    source.write_text(edited + '''} // namespace gsplat

#include <torch/library.h>

static void project(const std::vector<at::Tensor>& inputs, int64_t width, int64_t height,
                    at::Tensor radii, at::Tensor means2d, at::Tensor depths, at::Tensor conics) {
    gsplat::launch_projection_tile_ablation_fwd_kernel(
        inputs[0], c10::nullopt, inputs[1], inputs[2], c10::nullopt, inputs[3], inputs[4],
        width, height, 0.3f, 0.2f, 1e10f, 0.0f, gsplat::CameraModelType::PINHOLE,
        radii, means2d, depths, conics, c10::nullopt);
}

TORCH_LIBRARY(gsplat_tile_ablation, m) {
    m.def("project(Tensor[] inputs, int width, int height, Tensor(a!) radii, "
          "Tensor(b!) means2d, Tensor(c!) depths, Tensor(d!) conics) -> ()", &project);
}
''')
    compile_command = [str(source) if arg == str(original_path) else
                       arg.replace(obj, 'project.o') for arg in compile_command]
    compile_command.extend(['-I', str(original_path.parent)])
    link = shlex.split(subprocess.check_output(
        ['ninja', '-t', 'commands', 'gsplat_cuda.so'], cwd=build, text=True).splitlines()[-1])
    link = [link[0], 'project.o', *link[link.index('-shared'):-1], 'project.so']
    (output / 'commands.txt').write_text(shlex.join(compile_command) + '\n' + shlex.join(link) + '\n')
    for command in (compile_command, link):
        subprocess.run(command, cwd=output, check=True)


if __name__ == '__main__':
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))

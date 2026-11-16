#!/usr/bin/env python3
"""Build two isolated RGBD loading variants from the sealed gsplat CUDA source."""
import difflib
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main(build, destination):
    obj = 'RasterizeToPixels3DGSFwd.cuda.o'
    compile_command = shlex.split(subprocess.check_output(
        ['ninja', '-t', 'commands', obj], cwd=build, text=True))
    original_path = Path(compile_command[compile_command.index('-c') + 1])
    original = original_path.read_text()
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == (
        '9bebdfcb347798d49d329aa3d295431c53b41308b81dd5d8f64721b1426ec7de')
    link = shlex.split(subprocess.check_output(
        ['ninja', '-t', 'commands', 'gsplat_cuda.so'], cwd=build, text=True).splitlines()[-1])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / original_path.name).write_text(original)
    for arm in ('prefetch', 'padded'):
        source = original.split('// Explicit Instantiation:', 1)[0]
        source = source.replace('rasterize_to_pixels_3dgs_fwd_kernel',
                                f'rasterize_rgbd_{arm}_kernel')
        source = replace_once(source, 'sizeof(int32_t) + sizeof(vec3) + sizeof(vec3)',
                              'sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float4)')
        if arm == 'prefetch':
            source = replace_once(source, 'extern __shared__ int s[];',
                                  'extern __shared__ __align__(16) int s[];')
            source = replace_once(source, '    // current visibility left to render',
                '    float4 *rgbd_batch = reinterpret_cast<float4 *>(&conic_batch[block_size]);\n'
                '    // current visibility left to render')
            source = replace_once(source, '            conic_batch[tr] = conics[g];',
                '            conic_batch[tr] = conics[g];\n'
                '            rgbd_batch[tr] = reinterpret_cast<const float4 *>(colors)[g];')
            source = replace_once(source,
                '            int32_t g = id_batch[t];\n'
                '            const float vis = alpha * T;\n'
                '            const float *c_ptr = colors + g * CDIM;',
                '            const float vis = alpha * T;\n'
                '            const float4 rgbd = rgbd_batch[t];\n'
                '            const float *c_ptr = reinterpret_cast<const float *>(&rgbd);')
        body = source + '}  // namespace gsplat\n'
        path = destination / f'{arm}.cu'
        (destination / f'{arm}.diff').write_text(''.join(difflib.unified_diff(
            original.split('// Explicit Instantiation:', 1)[0].splitlines(keepends=True),
            source.splitlines(keepends=True), fromfile=original_path.name, tofile=path.name)))
        path.write_text(body + f'''
#include <torch/library.h>

static void launch_{arm}(const std::vector<at::Tensor>& inputs,
                        int64_t width, int64_t height,
                        at::Tensor image, at::Tensor alpha, at::Tensor last_ids) {{
    gsplat::launch_rasterize_rgbd_{arm}_kernel<4>(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[6], c10::nullopt,
        width, height, 16, inputs[4], inputs[5], image, alpha, last_ids);
}}

TORCH_LIBRARY_FRAGMENT(gsplat_rgbd_ablation, m) {{
    m.def("{arm}(Tensor[] inputs, int width, int height, Tensor(a!) image, "
          "Tensor(b!) alpha, Tensor(c!) last_ids) -> ()", &launch_{arm});
}}
''')
        object_file = f'{arm}.o'
        command = [str(path) if arg == str(original_path) else
                   arg.replace(obj, object_file) for arg in compile_command]
        command.extend(['-I', str(original_path.parent), '-Xptxas=-v'])
        shared_object = [link[0], object_file, *link[link.index('-shared'):-1], f'{arm}.so']
        (destination / f'{arm}.commands.txt').write_text(
            shlex.join(command) + '\n' + shlex.join(shared_object) + '\n')
        subprocess.run(command, cwd=destination, check=True)
        subprocess.run(shared_object, cwd=destination, check=True)


if __name__ == '__main__':
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))

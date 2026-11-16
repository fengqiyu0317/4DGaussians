#!/usr/bin/env python3
"""Build the wide geometry layout from the sealed RGBD prefetch source."""
import difflib
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys


def main(reference, output):
    source = reference / 'prefetch.cu'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        '0fa41f5bc4ae59982cc6b4176ea0a09cb600b93d19c135ef353f828d226da515')
    original = source.read_text()
    wide = original.replace('prefetch', 'wide')
    substitutions = (
        ('''    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]

    float4 *rgbd_batch = reinterpret_cast<float4 *>(&conic_batch[block_size]);''',
         '''    float2 *xy_batch = reinterpret_cast<float2 *>(&id_batch[block_size]);
    float4 *conic_opacity_batch = reinterpret_cast<float4 *>(&xy_batch[block_size]);

    float4 *rgbd_batch = &conic_opacity_batch[block_size];'''),
        ('''            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];''',
         '''            const vec3 conic = conics[g];
            xy_batch[tr] = make_float2(xy.x, xy.y);
            conic_opacity_batch[tr] = make_float4(conic.x, conic.y, conic.z, opac);'''),
        ('''            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};''',
         '''            const float4 conic = conic_opacity_batch[t];
            const float2 xy = xy_batch[t];
            const float opac = conic.w;
            const vec2 delta = {xy.x - px, xy.y - py};'''),
        ('sizeof(vec3) + sizeof(vec3)', 'sizeof(float2) + sizeof(float4)'),
    )
    for old, new in substitutions:
        assert wide.count(old) == 1, old
        wide = wide.replace(old, new)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'wide.cu').write_text(wide)
    (output / 'wide.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), wide.splitlines(keepends=True),
        fromfile='prefetch.cu', tofile='wide.cu')))
    commands = []
    for line in (reference / 'prefetch.commands.txt').read_text().splitlines():
        command = shlex.split(line)
        command = ['wide.cu' if arg == str(source) else
                   arg.replace('prefetch.o', 'wide.o').replace('prefetch.so', 'wide.so')
                   for arg in command]
        commands.append(command)
    (output / 'commands.txt').write_text(''.join(shlex.join(command) + '\n' for command in commands))
    for command in commands:
        subprocess.run(command, cwd=output, check=True)
    sass = subprocess.check_output(
        ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', 'wide.so'], cwd=output, text=True)
    (output / 'wide.sass.txt').write_text(sass)
    assert 'LDS.64 ' in sass and sass.count('LDS.128 ') >= 2


if __name__ == '__main__':
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))

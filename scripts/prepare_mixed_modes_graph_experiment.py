"""Create an isolated Jetson runtime with runtime-selected mixed-tail controls.

Run remotely with BASE and a fresh EXPERIMENT directory. Existing runtimes and
binaries are read-only. Production kernels remain present and unchanged.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base, experiment = map(lambda p: Path(p).resolve(), sys.argv[1:])
    assert str(experiment).startswith("/data/qyfeng/")
    experiment.mkdir(exist_ok=False)
    root = experiment / "runtime"
    shutil.copytree(base, root, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "build", "results*", "nsight_reports", "outputs"))
    relative = Path("submodules/depth-diff-gaussian-rasterization")
    source = root / relative / "cuda_rasterizer/tacker_mixed.cu"
    original = source.read_text()
    start = original.index("template <bool FAST_RASTER_MATH, bool OPTIMIZED_DEFORM_SHARED,\n\tbool CANDIDATE_AWARE_PAIRING>\n__device__ __forceinline__ void tackerMixedBalancedRasterOptimizedDevice(")
    end = original.index("// ABI-v10 preserves", start)
    device = original[start:end].replace("tackerMixedBalancedRasterOptimizedDevice(",
                                         "tackerMixedModesBenchmarkDevice(", 1)
    device = device.replace("\tint deform_tiles)\n", "\tint deform_tiles,\n\tint benchmark_mode)\n", 1)
    device = device.replace("\tCudaRasterizer::Tacker::general_ptb_render_raster_optimized<",
                            "\tif (benchmark_mode != 2)\n\tCudaRasterizer::Tacker::general_ptb_render_raster_optimized<", 1)
    device = device.replace("\tconst int local_thread = static_cast<int>(threadIdx.x) -",
                            "\tif (benchmark_mode == 1)\n\t\treturn;\n\tconst int local_thread = static_cast<int>(threadIdx.x) -", 1)
    wrappers = []
    for symbol in ("tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11",
                   "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12"):
        name = original.index(symbol + "(")
        begin = original.rfind('extern "C" __global__', 0, name)
        finish = original.index("\n}\n", name) + 3
        wrapper = original[begin:finish].replace(symbol, symbol + "_modes_benchmark", 1)
        wrapper = wrapper.replace("\tint deform_tiles)\n", "\tint deform_tiles,\n\tint benchmark_mode)\n", 1)
        wrapper = wrapper.replace("tackerMixedBalancedRasterOptimizedDevice<",
                                  "tackerMixedModesBenchmarkDevice<", 1)
        wrapper = wrapper.replace("\t\tdeform_tiles);", "\t\tdeform_tiles, benchmark_mode);")
        wrappers.append(wrapper)
    assert all("benchmark_mode);" in text for text in wrappers)
    addition = '\n// Isolated experiment only: 0=both, 1=Raster, 2=Deform.\n' + device + "\n".join(wrappers)
    patched = original[:end] + addition + original[end:]
    for launcher, symbol in (
        ("launchMixedRenderDeformMlpTensorTailRasterFast", "tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11"),
        ("launchMixedRenderDeformMlpTensorTailSplitRaster", "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12"),
    ):
        begin = patched.index("void CudaRasterizer::Tacker::" + launcher + "(")
        call = patched.index("\t" + symbol + "<<<", begin)
        finish = patched.index(";\n", call) + 2
        invocation = patched[call:finish]
        benchmark_call = invocation.replace(symbol, symbol + "_modes_benchmark", 1)
        benchmark_call = benchmark_call.replace("deform_tiles);", "deform_tiles, benchmark_mode);")
        alternate = ('\tconst char* benchmark_env = std::getenv("TACKER_BENCHMARK_MIXED_MODE");\n'
                     '\tif (benchmark_env != nullptr)\n\t{\n'
                     '\t\tconst int benchmark_mode = std::atoi(benchmark_env);\n'
                     '\t\tif (benchmark_mode < 0 || benchmark_mode > 2)\n'
                     '\t\t\tthrow std::invalid_argument("invalid benchmark mode");\n' +
                     benchmark_call + '\t\tthrowOnLaunchError();\n\t\treturn;\n\t}\n')
        patched = patched[:call] + alternate + patched[call:]
    source.write_text(patched)
    (experiment / "original_tacker_mixed.cu").write_text(original)
    (experiment / "patched_tacker_mixed.cu").write_text(patched)
    objects = sorted((base / relative / "build").rglob("*.o"))
    assert len(objects) == 7, objects
    out_object = experiment / "tacker_mixed.o"
    compile_command = ["/usr/local/cuda/bin/nvcc", "-O3", "-lineinfo", "-Xptxas=-v",
                       "-std=c++17", "--compiler-options", "-fPIC",
                       "-gencode=arch=compute_87,code=sm_87",
                       "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                       "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
                       "--expt-relaxed-constexpr", "-I" + str(root / "tacker_ext/include"),
                       "-I" + str(root / relative / "third_party/glm"),
                       "-c", str(source), "-o", str(out_object)]
    original_binary = base / relative / "diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so"
    original_hash = digest(original_binary)
    with (experiment / "compile.log").open("w") as log:
        subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    target = root / relative / "diff_gaussian_rasterization" / original_binary.name
    link_command = ["aarch64-linux-gnu-g++", "-shared"] + [
        str(out_object if obj.name == "tacker_mixed.o" else obj) for obj in objects
    ] + ["-L/home/jetson/.local/lib/python3.10/site-packages/torch/lib", "-L/usr/local/cuda/lib64",
         "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lcudart", "-lc10_cuda", "-ltorch_cuda",
         "-o", str(target)]
    with (experiment / "link.log").open("w") as log:
        subprocess.run(link_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert digest(original_binary) == original_hash
    (experiment / "build.json").write_text(json.dumps({
        "base": str(base), "runtime": str(root), "compile_command": compile_command,
        "link_command": link_command, "base_binary_sha256": original_hash,
        "base_binary_unchanged": True, "experiment_binary_sha256": digest(target),
        "original_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "patched_source_sha256": digest(source), "production_kernels_unchanged": True,
        "diagnostic": "runtime mode only; identical production device work, mapping and launch bounds; no clock64 instrumentation",
    }, indent=2) + "\n")
    print("BUILD_DONE", root, flush=True)


if __name__ == "__main__":
    main()

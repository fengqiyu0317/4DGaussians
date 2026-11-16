"""Run independent processes of the synthetic v11 standalone benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--deps-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--rows", type=int, default=111525)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    root = Path(args.experiment_dir).resolve()
    selected_env = {
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": args.deps_dir + ":" + str(root),
        "TORCHINDUCTOR_CACHE_DIR": str(root.parent / "inductor_cache_v11_standalone"),
        "TRITON_CACHE_DIR": str(root.parent / "triton_cache"),
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
    }
    environment = dict(os.environ, **selected_env)
    commands = []
    for index in range(args.repeats):
        output = root / f"run_{index+1:02d}.json"
        assert not output.exists(), f"refusing to replace {output}"
        commands.append([sys.executable, "-u", str(root / "benchmark_deform_v11_standalone.py"),
                         "--library", str(root / "libdeform_v11_standalone.so"),
                         "--header", str(root / "include/deform_mlp_device.cuh"),
                         "--output", str(output), "--rows", str(args.rows), "--seed", str(args.seed),
                         "--rounds", str(args.rounds), "--calls", str(args.calls)])
    build = ["/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-lineinfo", "-Xptxas=-v",
             "-gencode=arch=compute_87,code=sm_87", "-shared", "-Xcompiler", "-fPIC",
             "-I", str(root / "include"), str(root / "deform_v11_standalone_adapter.cu"),
             "-o", str(root / "libdeform_v11_standalone.so")]
    manifest = {"environment": selected_env, "commands": commands, "build_command": build,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "independent_processes": args.repeats, "runs": []}
    manifest_path = root / "reproduce_commands.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    for index, command in enumerate(commands):
        log_path = root / f"run_{index+1:02d}.log"
        print("START", index + 1, flush=True)
        with log_path.open("w") as log:
            run = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        manifest["runs"].append({"index": index+1, "exit_code": run.returncode,
                                 "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save()
        if run.returncode:
            print(log_path.read_text()[-7000:], flush=True)
            raise SystemExit(run.returncode)
        report = json.loads(Path(command[command.index("--output")+1]).read_text())
        assert report["passed"]
        print("DONE", index+1, {name: round(value["median_cuda_ms"], 6)
                               for name, value in report["measurements"].items()}, flush=True)
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()


if __name__ == "__main__":
    main()

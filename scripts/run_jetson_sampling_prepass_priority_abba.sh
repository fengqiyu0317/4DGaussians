#!/usr/bin/env bash
# Predeclared outer ABBA: A=priority 0, B=sampling priority -3 (read back).
# Identical native kernels and Graph instantiation/launch backend in both arms.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: RUNTIME_ROOT SCRIPT_DIRECTORY OUTPUT_ROOT'; exit 2; }
runtime_root=$(realpath "$1")
script_root=$(realpath "$2")
output_root=$(realpath -m "$3")
[[ "$script_root" == /data/qyfeng/* && "$output_root" == /data/qyfeng/* ]] || exit 2
[[ ! -e "$output_root" ]] || { echo 'refusing to overwrite results'; exit 2; }
mkdir -p "$output_root"
sha256sum "$0" "$script_root/benchmark_sampling_prepass_graph.py" \
    "$script_root/sampling_prepass_priority.py" "$script_root/run_jetson_sampling_prepass_graph.sh" \
    > "$output_root/batch_sources.sha256"
date --iso-8601=seconds > "$output_root/started.txt"
policies=(default-node high high default-node)
labels=(A B B A)
for index in 0 1 2 3; do
    policy=${policies[$index]}
    printf 'RUN %s %s %s\n' "$index" "${labels[$index]}" "$policy"
    bash "$script_root/run_jetson_sampling_prepass_graph.sh" "$runtime_root" \
        "$script_root/benchmark_sampling_prepass_graph.py" "$output_root/run${index}_${labels[$index]}" \
        "$policy" 4 40
    sha256sum --check "$output_root/batch_sources.sha256" > "$output_root/run${index}.batch_source_check.txt"
done
date --iso-8601=seconds > "$output_root/completed.txt"
echo "ABBA_DONE $output_root"

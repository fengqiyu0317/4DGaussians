#!/usr/bin/env bash
# Prepare a COLMAP point cloud from existing camera poses. Run from any cwd.
set -euo pipefail
if [[ $# != 2 || ! $2 =~ ^(blender|hypernerf|llff)$ ]]; then
    echo 'Usage: bash colmap.sh DATASET blender|hypernerf|llff' >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workdir=$(cd -- "$1" && pwd)
datatype=$2
python_bin=${PYTHON:-python3}
command -v colmap >/dev/null
# Keep user data and previous reconstructions intact; use a fresh dataset copy to rerun.
for generated in sparse_ image_colmap colmap; do
    if [[ -e "$workdir/$generated" ]]; then
        echo "Refusing to overwrite existing output: $workdir/$generated" >&2
        exit 1
    fi
done
"$python_bin" "$root/scripts/${datatype}2colmap.py" "$workdir"
mkdir "$workdir/colmap"
cp -r "$workdir/image_colmap" "$workdir/colmap/images"
cp -r "$workdir/sparse_" "$workdir/colmap/sparse_custom"
colmap feature_extractor --database_path "$workdir/colmap/database.db" --image_path "$workdir/colmap/images" --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
"$python_bin" "$root/database.py" --database_path "$workdir/colmap/database.db" --txt_path "$workdir/colmap/sparse_custom/cameras.txt"
colmap exhaustive_matcher --database_path "$workdir/colmap/database.db"
mkdir -p "$workdir/colmap/sparse/0"
colmap point_triangulator --database_path "$workdir/colmap/database.db" --image_path "$workdir/colmap/images" --input_path "$workdir/colmap/sparse_custom" --output_path "$workdir/colmap/sparse/0" --clear_points 1
mkdir -p "$workdir/colmap/dense/workspace"
colmap image_undistorter --image_path "$workdir/colmap/images" --input_path "$workdir/colmap/sparse/0" --output_path "$workdir/colmap/dense/workspace"
colmap patch_match_stereo --workspace_path "$workdir/colmap/dense/workspace"
colmap stereo_fusion --workspace_path "$workdir/colmap/dense/workspace" --output_path "$workdir/colmap/dense/workspace/fused.ply"

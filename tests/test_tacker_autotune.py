"""CPU contracts for the Phase-3 Tacker autotune coordinator."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests.test_tacker_pipeline import _load_module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "tacker_autotune.py"
SPEC = importlib.util.spec_from_file_location("tacker_autotune", SCRIPT_PATH)
AUTOTUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOTUNE)


def _resource_facts(worker_groups=1):
    return {
        "physical_threads": 256 + 128 * worker_groups,
        "registers_per_thread": 32 + worker_groups,
        "static_shared_bytes": 128,
        "kernel_max_threads_per_block": 896,
        "active_blocks_per_multiprocessor": 1,
        "occupancy": 0.5,
        "launch_supported": True,
    }


def _physical_matrix(values=(80,)):
    return AUTOTUNE.build_base_matrix(
        list(values),
        sm_count=84,
        raster_tile_count=5440,
        backend_logical_blocks=13942,
    )


class ImportAndGeometryTest(unittest.TestCase):
    def test_top_level_has_no_torch_cuda_or_gaussian_renderer_import(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(
            any(
                name == "torch"
                or name.startswith("torch.")
                or name == "gaussian_renderer"
                or name.startswith("gaussian_renderer.")
                or name.startswith("diff_gaussian_rasterization")
                for name in imported
            )
        )

    def test_delayed_runtime_import_bootstraps_project_root(self):
        sentinel = object()
        with mock.patch.object(AUTOTUNE.sys, "path", ["/isolated"]), mock.patch.object(
            AUTOTUNE.importlib, "import_module", return_value=sentinel
        ) as import_module:
            self.assertIs(AUTOTUNE._load_tacker_pipeline(), sentinel)
            self.assertEqual(AUTOTUNE.sys.path[0], str(ROOT))
        import_module.assert_called_once_with("gaussian_renderer.tacker_pipeline")

    def test_launch_formula_matches_mixed_kernel(self):
        self.assertEqual(AUTOTUNE.raster_logical_blocks(1352, 1014), 5440)
        self.assertEqual(AUTOTUNE.first_linear_logical_blocks(111525), 13942)
        self.assertEqual(
            AUTOTUNE.effective_persistent_blocks(0, 84, 5440, 13942), 84
        )
        self.assertEqual(
            AUTOTUNE.effective_persistent_blocks(7000, 84, 5440, 13942),
            7000,
        )
        self.assertEqual(
            AUTOTUNE.effective_persistent_blocks(20000, 84, 5440, 13942),
            13942,
        )

    def test_pb_scan_covers_required_shape_facts(self):
        values = AUTOTUNE.derive_persistent_blocks(
            84, 5440, 13942, current_persistent_blocks=7000, extra_values=[0]
        )
        self.assertEqual(values, (0, 84, 168, 336, 5440, 7000, 13942))


class CandidateMatrixTest(unittest.TestCase):
    def test_base_enumeration_is_deterministic_and_complete(self):
        first = _physical_matrix((80, 160))
        second = _physical_matrix((160, 80, 80))
        self.assertEqual(first, second)
        self.assertIs(AUTOTUNE.validate_matrix(first), first)

        c0 = [item for item in first["candidates"] if item["search_level"] == "c0"]
        c1 = [item for item in first["candidates"] if item["search_level"] == "c1"]
        c2 = [item for item in first["candidates"] if item["search_level"] == "c2"]
        self.assertEqual(len(c0), 2)
        self.assertEqual(len(c1), 5 * 2)
        self.assertEqual(len(c2), 10 * 2 * 2)

        observed_pairs = {
            (tuple(item["selected_heads"]), item["worker_groups"])
            for item in c2
        }
        expected_pairs = set()
        for left_index, left in enumerate(AUTOTUNE.HEAD_ORDER):
            for right in AUTOTUNE.HEAD_ORDER[left_index + 1 :]:
                expected_pairs.add(((left, right), 1))
                expected_pairs.add(((left, right), 2))
        self.assertEqual(observed_pairs, expected_pairs)

    def test_effective_launch_identity_merges_zero_and_clamped_aliases(self):
        matrix = _physical_matrix((0, 84, 13942, 20000))
        c0 = [item for item in matrix["candidates"] if item["search_level"] == "c0"]
        self.assertEqual(len(c0), 2)
        self.assertEqual(
            [
                (
                    item["persistent_blocks"],
                    item["effective_persistent_blocks"],
                    item["requested_persistent_blocks"],
                )
                for item in c0
            ],
            [(84, 84, [0, 84]), (13942, 13942, [13942, 20000])],
        )

        explicit = AUTOTUNE.make_candidate(
            AUTOTUNE.FIRST_LINEAR_ABI_FAMILY,
            ["pos"],
            1,
            84,
            effective_block_count=84,
        )
        automatic = AUTOTUNE.make_candidate(
            AUTOTUNE.FIRST_LINEAR_ABI_FAMILY,
            ["pos"],
            1,
            0,
            effective_block_count=84,
        )
        self.assertEqual(
            explicit["candidate_sha256"], automatic["candidate_sha256"]
        )
        merged = AUTOTUNE.deduplicate_candidates([explicit, automatic])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["requested_persistent_blocks"], [0, 84])
        self.assertEqual(merged[0]["persistent_blocks"], 84)

    def test_abi_family_is_part_of_canonical_identity(self):
        legacy = AUTOTUNE.make_candidate(
            AUTOTUNE.LEGACY_ABI_FAMILY, ["pos"], 1, 80
        )
        v2 = AUTOTUNE.make_candidate(
            AUTOTUNE.FIRST_LINEAR_ABI_FAMILY, ["pos"], 1, 80
        )
        self.assertNotEqual(legacy["candidate_sha256"], v2["candidate_sha256"])

    def test_matrix_hash_and_candidate_hash_fail_closed(self):
        matrix = _physical_matrix()
        changed = deepcopy(matrix)
        changed["candidates"][0]["effective_persistent_blocks"] += 1
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "candidate_sha256"):
            AUTOTUNE.validate_matrix(changed)

        changed = deepcopy(matrix)
        changed["launch_geometry"]["sm_count"] += 1
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "geometry|matrix_sha256"):
            AUTOTUNE.validate_matrix(changed)

    def test_seal_requires_complete_base_cartesian_product_and_aliases(self):
        matrix = _physical_matrix((0, 84, 160))
        missing_c1 = list(matrix["candidates"])
        del missing_c1[
            next(
                index
                for index, item in enumerate(missing_c1)
                if item["search_level"] == "c1"
            )
        ]
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "Cartesian"):
            AUTOTUNE.seal_matrix(
                matrix["head_order"],
                matrix["persistent_blocks"],
                missing_c1,
                launch_geometry=matrix["launch_geometry"],
            )

        wrong_aliases = deepcopy(matrix["candidates"])
        for item in wrong_aliases:
            if item["effective_persistent_blocks"] == 84:
                item["requested_persistent_blocks"] = [84]
        with self.assertRaisesRegex(
            AUTOTUNE.AutotuneError, "PB set|Cartesian"
        ):
            AUTOTUNE.seal_matrix(
                matrix["head_order"],
                matrix["persistent_blocks"],
                wrong_aliases,
                launch_geometry=matrix["launch_geometry"],
            )

    def test_head_order_is_fixed_to_runtime_contract(self):
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "runtime HEAD_ORDER"):
            AUTOTUNE.build_base_matrix(
                [80], head_order=tuple(reversed(AUTOTUNE.HEAD_ORDER))
            )


class BeamSearchTest(unittest.TestCase):
    def _scores_for_size(self, matrix, size):
        candidates = [
            item
            for item in matrix["candidates"]
            if len(item["selected_heads"]) == size
            and item["abi_family"] == AUTOTUNE.FIRST_LINEAR_ABI_FAMILY
        ]
        return {
            item["candidate_sha256"]: float(len(candidates) - index)
            for index, item in enumerate(candidates)
        }

    def test_top_k_tie_break_and_three_head_coverage_are_stable(self):
        matrix = _physical_matrix()
        pairs = [
            item
            for item in matrix["candidates"]
            if len(item["selected_heads"]) == 2
        ]
        tied = {item["candidate_sha256"]: 1.0 for item in reversed(pairs)}
        first = AUTOTUNE.expand_beam_candidates(
            matrix["candidates"],
            tied,
            beam_width=1,
            target_head_count=3,
            persistent_blocks=matrix["persistent_blocks"],
        )
        second = AUTOTUNE.expand_beam_candidates(
            matrix["candidates"],
            list(
                reversed(
                    [
                        {"candidate_sha256": key, "score": value}
                        for key, value in tied.items()
                    ]
                )
            ),
            beam_width=1,
            target_head_count=3,
            persistent_blocks=matrix["persistent_blocks"],
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3 * 3)
        self.assertEqual({item["worker_groups"] for item in first}, {1, 2, 3})
        self.assertTrue(all(len(item["selected_heads"]) == 3 for item in first))

    def test_beam_can_expand_successively_through_five_heads(self):
        matrix = _physical_matrix()
        matrix = AUTOTUNE.extend_matrix_with_beam(
            matrix, self._scores_for_size(matrix, 2), 2, 3
        )
        matrix = AUTOTUNE.extend_matrix_with_beam(
            matrix, self._scores_for_size(matrix, 3), 2, 4
        )
        matrix = AUTOTUNE.extend_matrix_with_beam(
            matrix, self._scores_for_size(matrix, 4), 2, 5
        )
        AUTOTUNE.validate_matrix(matrix)
        five = [
            item for item in matrix["candidates"] if len(item["selected_heads"]) == 5
        ]
        self.assertEqual({item["worker_groups"] for item in five}, {1, 2, 3, 4, 5})
        self.assertEqual(
            {tuple(item["selected_heads"]) for item in five},
            {AUTOTUNE.HEAD_ORDER},
        )

    def test_failed_or_nonfinite_screening_cannot_enter_beam(self):
        matrix = _physical_matrix()
        pair = next(
            item for item in matrix["candidates"] if len(item["selected_heads"]) == 2
        )
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "finite"):
            AUTOTUNE.expand_beam_candidates(
                matrix["candidates"],
                {pair["candidate_sha256"]: float("nan")},
                1,
                3,
            )


class DescriptorAndProfileTest(unittest.TestCase):
    def test_v2_descriptor_is_runtime_contract_output(self):
        candidate = AUTOTUNE.make_candidate(
            AUTOTUNE.FIRST_LINEAR_ABI_FAMILY,
            ["pos", "scales"],
            2,
            80,
        )
        calls = []

        def contract(variant_id, heads, **kwargs):
            calls.append((variant_id, list(heads), kwargs))
            return {
                "variant_id": variant_id,
                "persistent_blocks": kwargs["persistent_blocks"],
                "partition": {
                    "selected_heads": list(heads),
                    "worker_groups": kwargs["worker_groups"],
                },
                "sentinel_from_runtime": True,
            }

        module = types.SimpleNamespace(
            HEAD_ORDER=AUTOTUNE.HEAD_ORDER,
            first_linear_candidate_contract=contract,
        )
        descriptor = AUTOTUNE.materialize_candidate_descriptor(
            candidate, module=module, resources={"registers_per_thread": 1}
        )
        self.assertTrue(descriptor["sentinel_from_runtime"])
        self.assertEqual(calls[0][1], ["pos", "scales"])
        self.assertEqual(calls[0][2]["worker_groups"], 2)

    def test_all_candidates_materialize_as_disabled_valid_profiles(self):
        module = _load_module()
        matrix = _physical_matrix()
        calls = []

        def provider(abi_family, worker_groups):
            calls.append((abi_family, worker_groups))
            return _resource_facts(worker_groups)

        profiles = AUTOTUNE.build_qualification_profiles(
            matrix,
            provider,
            module=module,
            template=module.load_tacker_profile(),
        )
        self.assertEqual(len(profiles), len(matrix["candidates"]))
        self.assertEqual(
            calls,
            [
                (AUTOTUNE.LEGACY_ABI_FAMILY, 1),
                (AUTOTUNE.FIRST_LINEAR_ABI_FAMILY, 1),
                (AUTOTUNE.FIRST_LINEAR_ABI_FAMILY, 2),
            ],
        )
        for candidate in matrix["candidates"]:
            profile = profiles[candidate["variant_id"]]
            self.assertEqual(profile["deployment"], {"enabled": False, "valid": False})
            self.assertEqual(
                profile["provenance"]["candidate_sha256"],
                candidate["candidate_sha256"],
            )
            self.assertEqual(
                profile["manifest"]["persistent_blocks"],
                candidate["effective_persistent_blocks"],
            )
            self.assertIs(module.validate_tacker_profile(profile), profile)

        with tempfile.TemporaryDirectory() as directory:
            manifest = AUTOTUNE.publish_qualification_profiles(
                matrix, profiles, directory
            )
            self.assertEqual(len(manifest["profiles"]), len(matrix["candidates"]))
            self.assertTrue(
                (Path(directory) / "qualification_profiles.json").is_file()
            )


class ProfileDatabaseTest(unittest.TestCase):
    def _setup(self, directory):
        matrix = _physical_matrix()
        candidate = matrix["candidates"][0]
        database = AUTOTUNE.ProfileDB(Path(directory) / "profile.sqlite")
        database.register_matrix(matrix)
        return database, matrix, candidate

    def test_success_is_skipped_after_reopen_but_changed_inputs_run(self):
        with tempfile.TemporaryDirectory() as directory:
            database, matrix, candidate = self._setup(directory)
            artifact = Path(directory) / "screening.json"
            artifact.write_text('{"fps":90}\n', encoding="utf-8")
            key = (matrix["matrix_sha256"], candidate["candidate_sha256"])
            claim = database.claim_stage(*key, "screening", {"frames": 5})
            self.assertEqual(claim["action"], "run")
            completed_record = database.complete_stage(
                *key,
                "screening",
                artifact,
                claim_token=claim["claim_token"],
                inputs={"frames": 5},
                result={"score": 90.0},
            )
            database.close()

            with AUTOTUNE.ProfileDB(Path(directory) / "profile.sqlite") as reopened:
                skip = reopened.claim_stage(*key, "screening", {"frames": 5})
                self.assertEqual(skip["action"], "skip")
                self.assertEqual(
                    skip["record"]["input_sha256"],
                    AUTOTUNE.input_sha256({"frames": 5}),
                )
                changed = reopened.claim_stage(*key, "screening", {"frames": 6})
                self.assertEqual(changed["action"], "run")
                self.assertNotEqual(
                    changed["record"]["input_sha256"],
                    skip["record"]["input_sha256"],
                )

    def test_failed_and_abandoned_stages_have_explicit_resume_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            database, matrix, candidate = self._setup(directory)
            key = (matrix["matrix_sha256"], candidate["candidate_sha256"])
            original = database.claim_stage(*key, "correctness", {"seed": 0})
            busy = database.claim_stage(*key, "correctness", {"seed": 0})
            self.assertEqual(busy["action"], "busy")
            reclaimed = database.claim_stage(
                *key,
                "correctness",
                {"seed": 0},
                reclaim_running=True,
            )
            self.assertEqual(reclaimed["action"], "run")
            self.assertEqual(reclaimed["record"]["attempt"], 2)
            self.assertNotEqual(original["claim_token"], reclaimed["claim_token"])
            stale_artifact = Path(directory) / "stale.json"
            stale_artifact.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                AUTOTUNE.ProfileDBConflictError, "claim token"
            ):
                database.complete_stage(
                    *key,
                    "correctness",
                    stale_artifact,
                    claim_token=original["claim_token"],
                    inputs={"seed": 0},
                    result={"valid": True},
                )
            with self.assertRaisesRegex(
                AUTOTUNE.ProfileDBConflictError, "claim token"
            ):
                database.fail_stage(
                    *key,
                    "correctness",
                    "stale worker",
                    claim_token=original["claim_token"],
                    inputs={"seed": 0}
                )
            database.fail_stage(
                *key,
                "correctness",
                "numeric mismatch",
                claim_token=reclaimed["claim_token"],
                inputs={"seed": 0}
            )
            retry = database.claim_stage(*key, "correctness", {"seed": 0})
            self.assertEqual(retry["action"], "run")
            self.assertEqual(retry["record"]["attempt"], 3)
            database.close()

    def test_artifact_damage_and_success_conflict_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            database, matrix, candidate = self._setup(directory)
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            key = (matrix["matrix_sha256"], candidate["candidate_sha256"])
            claim = database.claim_stage(*key, "screening", {"frames": 5})
            completed_record = database.complete_stage(
                *key,
                "screening",
                first,
                claim_token=claim["claim_token"],
                inputs={"frames": 5},
                result={"score": 1}
            )
            with self.assertRaisesRegex(AUTOTUNE.ProfileDBConflictError, "running"):
                database.complete_stage(
                    *key,
                    "screening",
                    second,
                    claim_token=claim["claim_token"],
                    inputs={"frames": 5},
                    result={"score": 1},
                )
            first.write_text("source changed", encoding="utf-8")
            skip = database.claim_stage(*key, "screening", {"frames": 5})
            self.assertEqual(skip["action"], "skip")
            self.assertEqual(
                completed_record["source_artifact_path"], str(first.resolve())
            )
            self.assertNotEqual(
                completed_record["artifact_path"], str(first.resolve())
            )
            snapshot = Path(completed_record["artifact_path"])
            os.chmod(str(snapshot), 0o644)
            snapshot.write_text("tampered snapshot", encoding="utf-8")
            with self.assertRaisesRegex(
                AUTOTUNE.ProfileDBCorruptionError, "artifact"
            ):
                database.claim_stage(*key, "screening", {"frames": 5})
            database.close()

    def test_logical_database_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            database, matrix, unused = self._setup(directory)
            path = database.path
            database.close()
            connection = sqlite3.connect(str(path))
            connection.execute(
                "UPDATE matrices SET matrix_json = ? WHERE matrix_sha256 = ?",
                ("{}", matrix["matrix_sha256"]),
            )
            connection.commit()
            connection.close()
            with AUTOTUNE.ProfileDB(path) as corrupted:
                with self.assertRaisesRegex(
                    AUTOTUNE.ProfileDBCorruptionError, "stored matrix"
                ):
                    corrupted.validate()

    def test_foreign_key_orphans_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            database, matrix, candidate = self._setup(directory)
            path = database.path
            database.close()
            connection = sqlite3.connect(str(path))
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute(
                "INSERT INTO candidates(matrix_sha256, candidate_sha256, "
                "candidate_json) VALUES (?, ?, ?)",
                ("f" * 64, "e" * 64, "{}"),
            )
            connection.commit()
            connection.close()
            with AUTOTUNE.ProfileDB(path) as corrupted:
                with self.assertRaisesRegex(
                    AUTOTUNE.ProfileDBCorruptionError, "foreign_key|orphan"
                ):
                    corrupted.validate()

    def test_source_change_after_final_check_does_not_change_sealed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            database, matrix, candidate = self._setup(directory)
            artifact = Path(directory) / "racing.json"
            artifact.write_text("before", encoding="utf-8")
            key = (matrix["matrix_sha256"], candidate["candidate_sha256"])
            claim = database.claim_stage(*key, "screening", {"frames": 5})
            original_hash = AUTOTUNE.hash_artifact_file
            calls = []

            def source_changing_hash(path):
                facts = original_hash(path)
                calls.append(facts["artifact_sha256"])
                if len(calls) == 4:
                    artifact.write_text("after", encoding="utf-8")
                return facts

            with mock.patch.object(
                AUTOTUNE, "hash_artifact_file", side_effect=source_changing_hash
            ):
                record = database.complete_stage(
                    *key,
                    "screening",
                    artifact,
                    claim_token=claim["claim_token"],
                    inputs={"frames": 5},
                    result={"score": 90.0},
                )
            self.assertEqual(record["status"], AUTOTUNE.STAGE_SUCCEEDED)
            self.assertIsNone(record["claim_token"])
            self.assertEqual(artifact.read_text(encoding="utf-8"), "after")
            snapshot = Path(record["artifact_path"])
            self.assertEqual(snapshot.read_text(encoding="utf-8"), "before")
            self.assertEqual(
                snapshot.parent, Path(str(database.path) + ".artifacts")
            )
            self.assertEqual(snapshot.name, record["artifact_sha256"])
            self.assertTrue(database.validate())
            skip = database.claim_stage(*key, "screening", {"frames": 5})
            self.assertEqual(skip["action"], "skip")
            self.assertEqual(skip["record"]["artifact_path"], str(snapshot))
            database.close()


class FormalPlanTest(unittest.TestCase):
    def test_serial_and_two_stream_baselines_must_be_correctness_valid(self):
        baselines = {
            "serial": {"valid": False},
            "two_stream": {"valid": True},
            "current_tacker": {"valid": True},
        }
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "serial.*valid=true"):
            AUTOTUNE._validate_baseline_correctness(baselines)

    def test_exact_db_results_generate_stable_top_k_benchmark_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = _physical_matrix()
            selected = matrix["candidates"][:2]
            correctness_inputs = {"reference": "serial", "seed": 0}
            screening_inputs = {"frames": 5, "seed": 0}
            correctness_digest = AUTOTUNE.input_sha256(correctness_inputs)
            screening_digest = AUTOTUNE.input_sha256(screening_inputs)
            database = AUTOTUNE.ProfileDB(root / "profile.sqlite")
            database.register_matrix(matrix)
            for index, candidate in enumerate(selected):
                correctness_artifact = root / "correctness-{}.json".format(index)
                screening_artifact = root / "screening-{}.json".format(index)
                correctness_artifact.write_text("{}", encoding="utf-8")
                screening_artifact.write_text("{}", encoding="utf-8")
                key = (matrix["matrix_sha256"], candidate["candidate_sha256"])
                correctness_claim = database.claim_stage(
                    *key, "correctness", correctness_inputs
                )
                database.complete_stage(
                    *key,
                    "correctness",
                    correctness_artifact,
                    claim_token=correctness_claim["claim_token"],
                    inputs=correctness_inputs,
                    result={"valid": True, "candidate_index": index},
                )
                screening_claim = database.claim_stage(
                    *key, "screening", screening_inputs
                )
                database.complete_stage(
                    *key,
                    "screening",
                    screening_artifact,
                    claim_token=screening_claim["claim_token"],
                    inputs=screening_inputs,
                    result={"score": 80.0 + index},
                )

            profiles = root / "profiles"
            profiles.mkdir()
            for candidate in selected:
                document = {
                    "selected_variant_id": candidate["variant_id"],
                    "deployment": {"enabled": False, "valid": False},
                    "provenance": {
                        "matrix_sha256": matrix["matrix_sha256"],
                        "candidate_sha256": candidate["candidate_sha256"],
                    },
                }
                (profiles / "{}.json".format(candidate["variant_id"])).write_text(
                    json.dumps(document), encoding="utf-8"
                )
            current = root / "current.json"
            current.write_text("{}", encoding="utf-8")
            baselines = {
                "serial": {"valid": True},
                "two_stream": {"valid": True},
                "current_tacker": {"valid": True},
            }
            correctness_output = root / "formal-correctness.json"
            first = AUTOTUNE.build_formal_benchmark_plan(
                database,
                matrix["matrix_sha256"],
                correctness_digest,
                screening_digest,
                1,
                profiles,
                current,
                baselines,
                correctness_output,
            )
            second = AUTOTUNE.build_formal_benchmark_plan(
                database,
                matrix["matrix_sha256"],
                correctness_digest,
                screening_digest,
                1,
                profiles,
                current,
                baselines,
                correctness_output,
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first["candidates"]), 1)
            self.assertEqual(first["candidates"][0]["screening_score"], 81.0)
            self.assertIn("--candidate", first["benchmark_argv_fragment"])
            self.assertEqual(
                set(first["correctness_qualifications"]),
                {
                    "serial",
                    "two_stream",
                    "current_tacker",
                    selected[1]["variant_id"],
                },
            )
            database.close()


class CLITest(unittest.TestCase):
    def test_matrix_and_db_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "matrix",
                    "--sm-count",
                    "84",
                    "--raster-tile-count",
                    "5440",
                    "--backend-logical-blocks",
                    "13942",
                    "--persistent-block",
                    "0",
                    "--persistent-block",
                    "20000",
                    "--output",
                    str(matrix_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            AUTOTUNE.validate_matrix(matrix)
            self.assertEqual(matrix["launch_geometry"]["sm_count"], 84)

            database_path = root / "profile.sqlite"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "db-register",
                    "--db",
                    str(database_path),
                    "--matrix",
                    str(matrix_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["matrix_sha256"], matrix["matrix_sha256"])

            inputs_path = root / "inputs.json"
            inputs_path.write_text('{"frames":5}', encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "db-claim",
                    "--db",
                    str(database_path),
                    "--matrix-sha256",
                    matrix["matrix_sha256"],
                    "--candidate-sha256",
                    matrix["candidates"][0]["candidate_sha256"],
                    "--stage",
                    "screening",
                    "--inputs",
                    str(inputs_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            claim = json.loads(completed.stdout)
            self.assertEqual(claim["action"], "run")
            self.assertRegex(claim["claim_token"], r"^[0-9a-f]{32}$")

            artifact_path = root / "screening.json"
            result_path = root / "screening-result.json"
            artifact_path.write_text("{}", encoding="utf-8")
            result_path.write_text('{"score":90.0}', encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "db-complete",
                    "--db",
                    str(database_path),
                    "--matrix-sha256",
                    matrix["matrix_sha256"],
                    "--candidate-sha256",
                    matrix["candidates"][0]["candidate_sha256"],
                    "--stage",
                    "screening",
                    "--inputs",
                    str(inputs_path),
                    "--artifact",
                    str(artifact_path),
                    "--claim-token",
                    claim["claim_token"],
                    "--result",
                    str(result_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "succeeded")

    def test_main_stdout_is_stable_json(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = AUTOTUNE.main(
                [
                    "matrix",
                    "--sm-count",
                    "84",
                    "--raster-tile-count",
                    "5440",
                    "--backend-logical-blocks",
                    "13942",
                    "--output",
                    "-",
                ]
            )
        self.assertEqual(result, 0)
        document = json.loads(stream.getvalue())
        AUTOTUNE.validate_matrix(document)
        self.assertEqual(stream.getvalue(), AUTOTUNE.pretty_json_text(document))


if __name__ == "__main__":
    unittest.main()

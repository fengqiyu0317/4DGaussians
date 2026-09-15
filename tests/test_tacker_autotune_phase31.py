"""CPU-only contracts for the Phase-3.1 Tacker candidate coordinator."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "tacker_autotune.py"
SPEC = importlib.util.spec_from_file_location(
    "tacker_autotune_phase31_contract", SCRIPT_PATH
)
AUTOTUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOTUNE)

SM_COUNT = 84
RASTER_BLOCKS = 5440
FIRST_LINEAR_BLOCKS = 13942
WHOLE_HEAD_BLOCKS = 111525
BASE_BLOCKS = (84, 168, 336, 5440, 7000, 13942)


def build_base(blocks=BASE_BLOCKS, packed=None, whole=None):
    return AUTOTUNE.build_phase31_base_matrix(
        blocks,
        sm_count=SM_COUNT,
        raster_tile_count=RASTER_BLOCKS,
        backend_logical_blocks=FIRST_LINEAR_BLOCKS,
        whole_head_logical_blocks=WHOLE_HEAD_BLOCKS,
        packed_persistent_blocks=packed,
        whole_head_persistent_blocks=whole,
    )


def successes(matrix, families=None, score_offset=0.0):
    result = []
    for index, candidate in enumerate(matrix["candidates"]):
        if families is not None and candidate["search_family"] not in families:
            continue
        result.append(
            {
                "candidate_sha256": candidate["candidate_sha256"],
                "status": AUTOTUNE.STAGE_SUCCEEDED,
                "score": score_offset + float(index),
            }
        )
    return result


def record_for(candidate, score):
    return {
        "candidate_sha256": candidate["candidate_sha256"],
        "status": AUTOTUNE.STAGE_SUCCEEDED,
        "score": float(score),
    }


class ExhaustiveMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = build_base()

    def test_current_geometry_is_486_with_exact_c2_450(self):
        matrix = self.matrix
        self.assertEqual(matrix["schema_version"], 2)
        counts = {
            family: sum(
                item["search_family"] == family
                for item in matrix["candidates"]
            )
            for family in AUTOTUNE.SEARCH_FAMILIES
        }
        self.assertEqual(
            counts, {"c0": 6, "c1": 30, "c2": 450, "c3": 0, "c4": 0}
        )
        c2_by_heads = {
            head_count: sum(
                item["search_family"] == "c2"
                and len(item["selected_heads"]) == head_count
                for item in matrix["candidates"]
            )
            for head_count in range(2, 6)
        }
        self.assertEqual(c2_by_heads, {2: 120, 3: 180, 4: 120, 5: 30})
        self.assertEqual(len(matrix["candidates"]), 486)
        c2 = [item for item in matrix["candidates"] if item["search_family"] == "c2"]
        self.assertEqual(
            AUTOTUNE.validate_exhaustive_c2_candidates(
                c2,
                BASE_BLOCKS,
                sm_count=SM_COUNT,
                raster_tile_count=RASTER_BLOCKS,
                backend_logical_blocks=FIRST_LINEAR_BLOCKS,
            ),
            c2,
        )

    def test_every_c2_head_set_has_all_legal_worker_groups_and_pb(self):
        by_head_set = {}
        for candidate in self.matrix["candidates"]:
            if candidate["search_family"] != "c2":
                continue
            key = tuple(candidate["selected_heads"])
            by_head_set.setdefault(key, set()).add(
                (
                    candidate["worker_groups"],
                    candidate["effective_persistent_blocks"],
                )
            )
        self.assertEqual(len(by_head_set), 26)
        for head_set, launches in by_head_set.items():
            expected = {
                (worker_groups, block_count)
                for worker_groups in range(1, len(head_set) + 1)
                for block_count in BASE_BLOCKS
            }
            self.assertEqual(launches, expected)

    def test_family_pb_grids_keep_whole_row_count_out_of_c2(self):
        matrix = self.matrix
        requested = matrix["persistent_blocks_by_family"]
        effective = matrix["effective_persistent_blocks_by_family"]
        for family in ("c0", "c1", "c2", "c3"):
            self.assertEqual(requested[family], list(BASE_BLOCKS))
            self.assertEqual(effective[family], list(BASE_BLOCKS))
        self.assertEqual(
            requested["c4"], list(BASE_BLOCKS) + [WHOLE_HEAD_BLOCKS]
        )
        self.assertEqual(
            effective["c4"], list(BASE_BLOCKS) + [WHOLE_HEAD_BLOCKS]
        )
        self.assertFalse(
            any(
                item["effective_persistent_blocks"] == WHOLE_HEAD_BLOCKS
                for item in matrix["candidates"]
            )
        )

    def test_effective_pb_aliases_dedupe_without_expanding_c2(self):
        matrix = build_base((0, 84, 13942, 20000))
        self.assertEqual(
            matrix["effective_persistent_blocks_by_family"]["c2"],
            [84, 13942],
        )
        self.assertEqual(
            sum(item["search_family"] == "c2" for item in matrix["candidates"]),
            150,
        )
        pos_pair = next(
            item
            for item in matrix["candidates"]
            if item["search_family"] == "c2"
            and item["selected_heads"] == ["pos", "scales"]
            and item["worker_groups"] == 1
            and item["effective_persistent_blocks"] == 84
        )
        self.assertEqual(pos_pair["requested_persistent_blocks"], [0, 84])
        self.assertEqual(
            matrix["persistent_blocks_by_family"]["c4"],
            [0, 84, 13942, 20000, WHOLE_HEAD_BLOCKS],
        )

    def test_matrix_and_candidate_hashes_are_deterministic_and_tamper_evident(self):
        reversed_matrix = build_base(tuple(reversed(BASE_BLOCKS)))
        self.assertEqual(self.matrix, reversed_matrix)
        self.assertIs(AUTOTUNE.validate_matrix(self.matrix), self.matrix)

        changed = deepcopy(self.matrix)
        changed["candidates"][0]["search_family"] = "c4"
        with self.assertRaisesRegex(
            AUTOTUNE.AutotuneError, "candidate_sha256|non-canonical"
        ):
            AUTOTUNE.validate_matrix(changed)

        changed = deepcopy(self.matrix)
        changed["effective_persistent_blocks_by_family"]["c4"][-1] -= 1
        with self.assertRaisesRegex(
            AUTOTUNE.AutotuneError, "matrix_sha256|not canonical"
        ):
            AUTOTUNE.validate_matrix(changed)

        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "C0--C2"):
            AUTOTUNE.seal_phase31_matrix(
                self.matrix["head_order"],
                self.matrix["persistent_blocks"],
                self.matrix["candidates"][:-1],
                self.matrix["launch_geometry"],
                self.matrix["persistent_blocks_by_family"],
            )

    def test_schema_v1_stays_byte_stable_and_valid(self):
        matrix = AUTOTUNE.build_base_matrix(
            [80],
            sm_count=SM_COUNT,
            raster_tile_count=RASTER_BLOCKS,
            backend_logical_blocks=FIRST_LINEAR_BLOCKS,
        )
        self.assertEqual(matrix["schema_version"], 1)
        self.assertEqual(
            matrix["matrix_sha256"],
            "cf2da673623010debd6063c4896d0d24a399bea9fab7a74f63b5c82464cfb1e6",
        )
        self.assertEqual(
            hashlib.sha256(AUTOTUNE.canonical_json_bytes(matrix)).hexdigest(),
            "61466e9b897190e8ee1e40425bcdcf9e82a823921080bc025214e5865d03b8e2",
        )
        self.assertIs(AUTOTUNE.validate_matrix(matrix), matrix)


class StagedGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = build_base()

    def test_c3_uses_top_unique_c2_head_sets_and_complete_family_grid(self):
        pair = next(
            item
            for item in self.base["candidates"]
            if item["search_family"] == "c2"
            and item["selected_heads"] == ["pos", "scales"]
            and item["worker_groups"] == 2
        )
        triple = next(
            item
            for item in self.base["candidates"]
            if item["search_family"] == "c2"
            and item["selected_heads"] == ["pos", "scales", "rotations"]
            and item["worker_groups"] == 3
        )
        records = [record_for(pair, 100.0), record_for(triple, 90.0)]
        matrix = AUTOTUNE.extend_phase31_with_c3(self.base, records, top_k=2)
        c3 = [item for item in matrix["candidates"] if item["search_family"] == "c3"]
        self.assertEqual(
            {tuple(item["selected_heads"]) for item in c3},
            {("pos", "scales"), ("pos", "scales", "rotations")},
        )
        self.assertEqual(len(c3), (2 + 3) * len(BASE_BLOCKS))
        self.assertEqual(
            {item["abi_family"] for item in c3},
            {AUTOTUNE.PACKED_FIRST_LINEAR_ABI_FAMILY},
        )
        self.assertIs(AUTOTUNE.validate_matrix(matrix), matrix)

    def test_c3_accepts_sealed_terminal_screening_evidence(self):
        candidate = next(
            item
            for item in self.base["candidates"]
            if item["search_family"] == "c2"
        )
        record = record_for(candidate, 100.0)
        record.update(
            {
                "error": None,
                "artifact_sha256": "a" * 64,
                "attempt": 1,
            }
        )
        matrix = AUTOTUNE.extend_phase31_with_c3(
            self.base, [record], top_k=1
        )
        self.assertTrue(
            any(item["search_family"] == "c3" for item in matrix["candidates"])
        )

        bad = deepcopy(record)
        bad["unsealed_note"] = "must fail closed"
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "unknown fields"):
            AUTOTUNE.extend_phase31_with_c3(self.base, [bad], top_k=1)

    def test_c4_preserves_successful_single_and_multi_sources(self):
        c1 = next(
            item
            for item in self.base["candidates"]
            if item["search_family"] == "c1" and item["selected_heads"] == ["shs"]
        )
        c2 = next(
            item
            for item in self.base["candidates"]
            if item["search_family"] == "c2"
            and item["selected_heads"] == ["pos", "opacity"]
        )
        c3_matrix = AUTOTUNE.extend_phase31_with_c3(
            self.base, [record_for(c2, 10.0)], top_k=1
        )
        c3 = next(
            item
            for item in c3_matrix["candidates"]
            if item["search_family"] == "c3"
        )
        records = [
            record_for(c1, 100.0),
            record_for(c2, 90.0),
            record_for(c3, 110.0),
        ]
        matrix = AUTOTUNE.extend_phase31_with_c4(
            c3_matrix, records, top_k_per_family=1
        )
        c4 = [item for item in matrix["candidates"] if item["search_family"] == "c4"]
        head_sets = {tuple(item["selected_heads"]) for item in c4}
        self.assertIn(("shs",), head_sets)
        self.assertIn(("pos", "opacity"), head_sets)
        self.assertTrue(any(len(head_set) == 1 for head_set in head_sets))
        self.assertTrue(any(len(head_set) > 1 for head_set in head_sets))
        self.assertEqual(
            {item["abi_family"] for item in c4},
            {AUTOTUNE.WHOLE_HEAD_ABI_FAMILY},
        )
        for head_set in head_sets:
            launches = {
                (item["worker_groups"], item["effective_persistent_blocks"])
                for item in c4
                if tuple(item["selected_heads"]) == head_set
            }
            expected = {
                (worker_groups, block_count)
                for worker_groups in range(1, len(head_set) + 1)
                for block_count in list(BASE_BLOCKS) + [WHOLE_HEAD_BLOCKS]
            }
            self.assertEqual(launches, expected)

    def test_staged_generation_fails_without_successful_parent(self):
        failed = {
            item["candidate_sha256"]: {"status": "failed", "score": None}
            for item in self.base["candidates"]
            if item["search_family"] == "c2"
        }
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "successful C2"):
            AUTOTUNE.extend_phase31_with_c3(self.base, failed, 1)
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "successful C1/C2/C3"):
            AUTOTUNE.extend_phase31_with_c4(self.base, failed, 1)


class RankingAndFormalSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = build_base()
        base_records = successes(base)
        c3 = AUTOTUNE.extend_phase31_with_c3(base, base_records, 1)
        parent_records = base_records + successes(c3, {"c3"}, 1000.0)
        cls.matrix = AUTOTUNE.extend_phase31_with_c4(c3, parent_records, 1)
        cls.records = parent_records + successes(cls.matrix, {"c4"}, 2000.0)

    def test_full_ranking_records_success_failure_and_tie_order(self):
        records = deepcopy(self.records)
        failed_digest = records[0]["candidate_sha256"]
        records[0] = {
            "candidate_sha256": failed_digest,
            "status": "failed",
            "score": None,
            "error": "qualification failed",
        }
        records[1]["score"] = records[2]["score"]
        ranking = AUTOTUNE.build_screening_ranking(
            self.matrix,
            list(reversed(records)),
            screening_input_sha256="a" * 64,
        )
        self.assertTrue(ranking["complete"])
        self.assertEqual(ranking["terminal_count"], len(self.matrix["candidates"]))
        self.assertEqual(ranking["failed_count"], 1)
        self.assertNotIn(
            failed_digest,
            {item["candidate_sha256"] for item in ranking["ranking"]},
        )
        self.assertIs(AUTOTUNE.validate_screening_ranking(ranking, self.matrix), ranking)
        repeated = AUTOTUNE.build_screening_ranking(
            self.matrix,
            records,
            screening_input_sha256="a" * 64,
        )
        self.assertEqual(ranking, repeated)

    def test_missing_or_nonterminal_results_fail_closed(self):
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "all candidates"):
            AUTOTUNE.build_screening_ranking(self.matrix, self.records[:-1])
        incomplete = AUTOTUNE.build_screening_ranking(
            self.matrix, self.records[:-1], require_terminal=False
        )
        self.assertFalse(incomplete["complete"])
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "not terminal"):
            AUTOTUNE.validate_screening_ranking(incomplete)
        running = deepcopy(self.records)
        running[0]["status"] = "running"
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "terminal"):
            AUTOTUNE.build_screening_ranking(self.matrix, running)

    def test_ranking_hash_and_unknown_candidates_fail_closed(self):
        ranking = AUTOTUNE.build_screening_ranking(self.matrix, self.records)
        changed = deepcopy(ranking)
        changed["ranking"][0]["score"] += 1.0
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "ranking_sha256"):
            AUTOTUNE.validate_screening_ranking(changed, self.matrix)
        records = deepcopy(self.records)
        records[0]["candidate_sha256"] = "f" * 64
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "unknown candidate"):
            AUTOTUNE.build_screening_ranking(self.matrix, records)

    def test_formal_set_is_exact_global_top_k_union_family_best(self):
        ranking = AUTOTUNE.build_screening_ranking(self.matrix, self.records)
        formal = AUTOTUNE.build_formal_candidate_set(
            self.matrix, ranking, global_top_k=3
        )
        global_digests = {
            item["candidate_sha256"] for item in ranking["ranking"][:3]
        }
        family_best = {
            next(
                item["candidate_sha256"]
                for item in ranking["ranking"]
                if item["search_family"] == family
            )
            for family in AUTOTUNE.SEARCH_FAMILIES
        }
        selected = {
            item["candidate_sha256"] for item in formal["candidates"]
        }
        self.assertEqual(selected, global_digests | family_best)
        self.assertEqual(
            {item["search_family"] for item in formal["candidates"]},
            set(AUTOTUNE.SEARCH_FAMILIES),
        )
        self.assertEqual(formal["families_without_screening_success"], [])
        self.assertIs(
            AUTOTUNE.validate_formal_candidate_set(self.matrix, ranking, formal),
            formal,
        )
        self.assertEqual(
            formal,
            AUTOTUNE.build_formal_candidate_set(self.matrix, ranking, 3),
        )

    def test_family_local_backfill_respects_exclusions_and_zero_count(self):
        ranking = AUTOTUNE.build_screening_ranking(self.matrix, self.records)
        family = "c3"
        ranked = AUTOTUNE.ranked_family_candidates(self.matrix, ranking, family)
        self.assertGreaterEqual(len(ranked), 2)
        excluded = {ranked[0]["candidate"]["candidate_sha256"]}
        one = AUTOTUNE.family_local_backfill_candidates(
            self.matrix, ranking, family, count=1,
            excluded_candidate_sha256s=excluded,
        )
        self.assertEqual(one, [ranked[1]])
        self.assertEqual(
            AUTOTUNE.family_local_backfill_candidates(
                self.matrix, ranking, family, count=0
            ),
            [],
        )
        self.assertTrue(
            all(item["candidate"]["search_family"] == family for item in one)
        )


class DescriptorAndBaselineTest(unittest.TestCase):
    def test_c3_c4_descriptors_are_owned_by_runtime_contracts(self):
        calls = []

        def contract_for(family, backend):
            def contract(variant_id, heads, **kwargs):
                calls.append((family, variant_id, list(heads), deepcopy(kwargs)))
                return {
                    "variant_id": variant_id,
                    "abi_family": family,
                    "persistent_blocks": kwargs["persistent_blocks"],
                    "partition": {
                        "selected_heads": list(heads),
                        "worker_groups": kwargs["worker_groups"],
                        "backend": backend,
                    },
                    "runtime_owned": True,
                }
            return contract

        module = types.SimpleNamespace(
            HEAD_ORDER=AUTOTUNE.HEAD_ORDER,
            packed_first_linear_candidate_contract=contract_for(
                AUTOTUNE.PACKED_FIRST_LINEAR_ABI_FAMILY,
                "packed_first_linear",
            ),
            whole_head_candidate_contract=contract_for(
                AUTOTUNE.WHOLE_HEAD_ABI_FAMILY, "whole_head"
            ),
        )
        for family, heads in (
            (AUTOTUNE.PACKED_FIRST_LINEAR_ABI_FAMILY, ["pos", "scales"]),
            (AUTOTUNE.WHOLE_HEAD_ABI_FAMILY, ["opacity"]),
        ):
            candidate = AUTOTUNE.make_phase31_candidate(
                family, heads, len(heads), 84
            )
            descriptor = AUTOTUNE.materialize_candidate_descriptor(
                candidate,
                module=module,
                resources={"registers_per_thread": 48},
            )
            self.assertTrue(descriptor["runtime_owned"])
        self.assertEqual(
            [item[0] for item in calls],
            [
                AUTOTUNE.PACKED_FIRST_LINEAR_ABI_FAMILY,
                AUTOTUNE.WHOLE_HEAD_ABI_FAMILY,
            ],
        )
        self.assertEqual(calls[0][3]["resources"], {"registers_per_thread": 48})

    def test_descriptor_contract_mismatch_fails_closed(self):
        candidate = AUTOTUNE.make_phase31_candidate(
            AUTOTUNE.PACKED_FIRST_LINEAR_ABI_FAMILY,
            ["pos", "scales"],
            1,
            84,
        )
        module = types.SimpleNamespace(
            HEAD_ORDER=AUTOTUNE.HEAD_ORDER,
            packed_first_linear_candidate_contract=lambda *args, **kwargs: {
                "variant_id": candidate["variant_id"],
                "abi_family": AUTOTUNE.WHOLE_HEAD_ABI_FAMILY,
                "persistent_blocks": 84,
                "partition": {
                    "selected_heads": ["pos", "scales"],
                    "worker_groups": 1,
                    "backend": "packed_first_linear",
                },
            },
        )
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "ABI family"):
            AUTOTUNE.materialize_candidate_descriptor(candidate, module=module)

    def test_phase31_baselines_preserve_explicit_invalid_status(self):
        baselines = {
            "serial": {"valid": False, "reason": "reference failed"},
            "two_stream": {"valid": True},
            "current_tacker": {"valid": False, "reason": "stale profile"},
        }
        with self.assertRaisesRegex(AUTOTUNE.AutotuneError, "serial.*valid=true"):
            AUTOTUNE._validate_baseline_correctness(baselines)
        self.assertEqual(
            AUTOTUNE._validate_baseline_correctness(
                baselines, require_safe_baselines=False
            )["serial"],
            baselines["serial"],
        )

    def test_phase31_formal_plan_seals_invalid_baseline_status(self):
        matrix = build_base()
        candidate = matrix["candidates"][0]
        correctness_digest = "c" * 64
        screening_digest = "d" * 64

        class FakeDatabase(AUTOTUNE.ProfileDB):
            def __init__(self):
                pass

            def get_matrix(self, matrix_sha256):
                if matrix_sha256 != matrix["matrix_sha256"]:
                    raise AssertionError("unexpected matrix")
                return matrix

            def get_stage(
                self,
                matrix_sha256,
                candidate_sha256,
                stage_name,
                inputs=AUTOTUNE._MISSING,
                input_sha256=None,
                verify_artifact=True,
            ):
                if candidate_sha256 != candidate["candidate_sha256"]:
                    return None
                result = (
                    {"valid": True}
                    if stage_name == "correctness"
                    else {"score": 99.0}
                )
                return {
                    "status": AUTOTUNE.STAGE_SUCCEEDED,
                    "result": result,
                    "artifact_sha256": (
                        "e" * 64 if stage_name == "correctness" else "f" * 64
                    ),
                }

        baselines = {
            "serial": {"valid": False, "reason": "reference failed"},
            "two_stream": {"valid": True},
            "current_tacker": {"valid": False, "reason": "stale profile"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = root / "profiles"
            profiles.mkdir()
            AUTOTUNE.atomic_write_json(
                profiles / "{}.json".format(candidate["variant_id"]),
                {
                    "selected_variant_id": candidate["variant_id"],
                    "deployment": {"enabled": False, "valid": False},
                    "provenance": {
                        "matrix_sha256": matrix["matrix_sha256"],
                        "candidate_sha256": candidate["candidate_sha256"],
                    },
                },
            )
            current = root / "current.json"
            AUTOTUNE.atomic_write_json(current, {})
            plan = AUTOTUNE.build_formal_benchmark_plan(
                FakeDatabase(),
                matrix["matrix_sha256"],
                correctness_digest,
                screening_digest,
                1,
                profiles,
                current,
                baselines,
                root / "correctness.json",
            )
        self.assertEqual(plan["schema_version"], 2)
        self.assertEqual(plan["invalid_baselines"], ["serial", "current_tacker"])
        self.assertEqual(
            plan["baseline_terminal_status"],
            {
                "serial": "correctness_invalid",
                "two_stream": "correctness_valid",
                "current_tacker": "correctness_invalid",
            },
        )


class Phase31CLITest(unittest.TestCase):
    def run_cli(self, *arguments, expected=0):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)] + list(arguments),
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=20,
        )
        self.assertEqual(completed.returncode, expected, completed.stderr)
        return completed

    def test_phase31_cli_roundtrip_matrix_c3_c4_ranking_and_formal_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            c3_path = root / "c3.json"
            c4_path = root / "c4.json"
            records_path = root / "records.json"
            parent_path = root / "parents.json"
            all_path = root / "all.json"
            ranking_path = root / "ranking.json"
            formal_path = root / "formal.json"
            self.run_cli(
                "phase31-matrix",
                "--sm-count", str(SM_COUNT),
                "--raster-tile-count", str(RASTER_BLOCKS),
                "--backend-logical-blocks", str(FIRST_LINEAR_BLOCKS),
                "--whole-head-logical-blocks", str(WHOLE_HEAD_BLOCKS),
                "--output", str(matrix_path),
            )
            matrix = AUTOTUNE.load_json_file(matrix_path)
            self.assertEqual(len(matrix["candidates"]), 486)
            AUTOTUNE.atomic_write_json(records_path, successes(matrix))
            self.run_cli(
                "phase31-c3", "--matrix", str(matrix_path),
                "--screening", str(records_path), "--top-k", "1",
                "--output", str(c3_path),
            )
            c3 = AUTOTUNE.load_json_file(c3_path)
            parent_records = successes(matrix) + successes(c3, {"c3"}, 1000.0)
            AUTOTUNE.atomic_write_json(parent_path, parent_records)
            self.run_cli(
                "phase31-c4", "--matrix", str(c3_path),
                "--screening", str(parent_path), "--top-k-per-family", "1",
                "--output", str(c4_path),
            )
            c4 = AUTOTUNE.load_json_file(c4_path)
            all_records = parent_records + successes(c4, {"c4"}, 2000.0)
            AUTOTUNE.atomic_write_json(all_path, all_records)
            self.run_cli(
                "screening-ranking", "--matrix", str(c4_path),
                "--screening", str(all_path), "--output", str(ranking_path),
            )
            self.run_cli(
                "formal-set", "--matrix", str(c4_path),
                "--screening-ranking", str(ranking_path), "--top-k", "2",
                "--output", str(formal_path),
            )
            ranking = AUTOTUNE.load_json_file(ranking_path)
            formal = AUTOTUNE.load_json_file(formal_path)
            AUTOTUNE.validate_screening_ranking(ranking, c4)
            AUTOTUNE.validate_formal_candidate_set(c4, ranking, formal)
            self.assertEqual(
                {item["search_family"] for item in formal["candidates"]},
                set(AUTOTUNE.SEARCH_FAMILIES),
            )

    def test_cli_errors_fail_closed_and_do_not_hang(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = build_base()
            matrix_path = root / "matrix.json"
            failed_path = root / "failed.json"
            missing_path = root / "missing.json"
            AUTOTUNE.atomic_write_json(matrix_path, matrix)
            AUTOTUNE.atomic_write_json(
                failed_path,
                [
                    {
                        "candidate_sha256": item["candidate_sha256"],
                        "status": "failed",
                        "score": None,
                    }
                    for item in matrix["candidates"]
                    if item["search_family"] == "c2"
                ],
            )
            AUTOTUNE.atomic_write_json(missing_path, [])
            completed = self.run_cli(
                "phase31-c3", "--matrix", str(matrix_path),
                "--screening", str(failed_path), "--top-k", "1",
                expected=1,
            )
            self.assertIn("successful C2", completed.stderr)
            completed = self.run_cli(
                "screening-ranking", "--matrix", str(matrix_path),
                "--screening", str(missing_path), expected=1,
            )
            self.assertIn("all candidates", completed.stderr)


if __name__ == "__main__":
    unittest.main()

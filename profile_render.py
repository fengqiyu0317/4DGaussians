"""Render-only benchmark entry point for Nsight Systems profiling.

The executable portion starts with standard-library imports only.  The CLI
snapshots every executable/config/profile input plus the Raster, deformation
head, and simple-knn wrappers/binaries before importing PyTorch or any project
module.  Runtime imports and config/profile evaluation are then served from
those exact bytes.
"""

from argparse import ArgumentParser, Namespace
import ast
from copy import copy
import csv
import hashlib
import io
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter


# ``scripts/run_profile_render_sealed.py`` injects this object while executing
# the exact ``profile_render.py`` bytes that it read through one stable file
# descriptor.  Merely executing this file by pathname cannot establish that
# property, so the production CLI below requires the injected bootstrap seal.
_PROFILE_RENDER_BOOTSTRAP = globals().get("_PROFILE_RENDER_BOOTSTRAP")


# Populated only by ``_activate_runtime_imports`` after the pre-import seal.
# Globals preserve the historical helper/main APIs while making a CPU-only
# import of this module safe.
torch = None
_rasterizer_module = None
ModelHiddenParams = None
ModelParams = None
PipelineParams = None
GaussianModel = None
TackerRenderer = None
TwoStreamRenderer = None
deform_for_render = None
prepare_render_context = None
rasterize_state = None
render = None
Scene = None
safe_state = None
nvtx_range = None

_BINARY_SNAPSHOT_DIRECTORIES = []


class ProvenanceError(RuntimeError):
    """An executable input could not be sealed or changed during the run."""


def _stat_identity(stat_result):
    """Return the stable, JSON-safe stat fields used by the byte seal."""

    return {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "mode": int(stat_result.st_mode),
        "mtime_ns": int(
            getattr(
                stat_result,
                "st_mtime_ns",
                int(stat_result.st_mtime * 1000000000),
            )
        ),
        "ctime_ns": int(
            getattr(
                stat_result,
                "st_ctime_ns",
                int(stat_result.st_ctime * 1000000000),
            )
        ),
    }


def _same_stat_identity(first, second):
    return _stat_identity(first) == _stat_identity(second) and int(
        first.st_size
    ) == int(second.st_size)


def _snapshot_file_bytes(path, role, required=True):
    """Read one stable file image and return its public identity plus bytes."""

    input_path = str(Path(os.path.abspath(str(Path(path).expanduser()))))
    target = Path(input_path)
    resolved = target.resolve()
    if str(resolved) != input_path:
        raise ProvenanceError(
            "pre-import input path contains a symbolic link: {} ({})".format(
                role, input_path
            )
        )
    try:
        initial_path_stat = os.lstat(input_path)
    except OSError as error:
        if error.errno != getattr(os, "ENOENT", 2):
            raise ProvenanceError(
                "cannot inspect pre-import input {} ({}): {}".format(
                    role, input_path, error
                )
            )
        initial_path_stat = None
    path_exists = initial_path_stat is not None
    if path_exists and stat.S_ISLNK(initial_path_stat.st_mode):
        raise ProvenanceError(
            "pre-import input path is a symbolic link: {} ({})".format(
                role, input_path
            )
        )
    is_file = bool(
        path_exists and stat.S_ISREG(initial_path_stat.st_mode)
    )
    if path_exists and not is_file:
        raise ProvenanceError(
            "pre-import input is not a regular file: {} ({})".format(
                role, resolved
            )
        )
    record = {
        "role": role,
        "input_path": input_path,
        "path": input_path,
        "required": bool(required),
        "exists": is_file,
        "stat": None,
        "size_bytes": None,
        "sha256": None,
    }
    if not record["exists"]:
        if required:
            raise ProvenanceError(
                "required pre-import input {} is missing: {}".format(
                    role, input_path
                )
            )
        return record, None

    descriptor = None
    try:
        path_stat_before = os.lstat(input_path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(input_path, flags)
        descriptor_stat_before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        descriptor_stat_after = os.fstat(descriptor)
        path_stat_after = os.lstat(input_path)
    except OSError as error:
        raise ProvenanceError(
            "cannot snapshot pre-import input {} ({}): {}".format(
            role, input_path, error
            )
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)

    raw = b"".join(chunks)
    if not (
        _same_stat_identity(path_stat_before, descriptor_stat_before)
        and _same_stat_identity(
            descriptor_stat_before, descriptor_stat_after
        )
        and _same_stat_identity(descriptor_stat_after, path_stat_after)
        and len(raw) == int(descriptor_stat_after.st_size)
        and not stat.S_ISLNK(path_stat_after.st_mode)
        and str(Path(input_path).resolve()) == input_path
    ):
        raise ProvenanceError(
            "pre-import input changed while being snapshotted: {}".format(
                input_path
            )
        )

    record.update(
        {
            "stat": _stat_identity(descriptor_stat_after),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    return record, raw


def _python_config_bases(path, raw):
    """Find literal ``_base_`` paths without executing Python config bytes."""

    try:
        source = raw.decode("utf-8")
        tree = ast.parse(source, str(path))
    except (UnicodeError, SyntaxError) as error:
        raise ProvenanceError(
            "cannot parse configuration {} for _base_: {}".format(path, error)
        )
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_base_"
            for target in node.targets
        ):
            assignments.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_base_"
        ):
            assignments.append(node.value)
    if len(assignments) > 1:
        raise ProvenanceError(
            "configuration assigns _base_ more than once: {}".format(path)
        )
    if not assignments:
        return []
    try:
        base_value = ast.literal_eval(assignments[0])
    except (TypeError, ValueError) as error:
        raise ProvenanceError(
            "configuration {} has non-literal _base_: {}".format(path, error)
        )
    if isinstance(base_value, str):
        values = [base_value] if base_value else []
    elif isinstance(base_value, (list, tuple)):
        values = list(base_value)
    else:
        values = None
    if values is None or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ProvenanceError(
            "configuration {} _base_ must be a string or string list".format(
                path
            )
        )
    return [
        Path(os.path.abspath(str(Path(path).parent / value)))
        for value in values
    ]


def _locate_extension_package_files(
    package_name, fallback_directory, wrapper_required=True
):
    """Locate one package wrapper and its import-compatible ``_C`` safely.

    ``find_spec(package_name)`` does not execute a top-level package.  Looking
    up ``package_name + '._C'`` would execute the parent and is intentionally
    avoided here.
    """

    package_directories = []
    try:
        package_spec = importlib.util.find_spec(package_name)
    except (ImportError, AttributeError, ValueError):
        package_spec = None
    if package_spec is not None:
        if package_spec.origin:
            origin = Path(package_spec.origin).expanduser().absolute()
            package_directories.append(origin.parent)
        for location in package_spec.submodule_search_locations or ():
            package_directories.append(Path(location).expanduser().absolute())
    fallback = Path(fallback_directory).expanduser().absolute()
    package_directories.append(fallback)
    package_directories = list(
        dict.fromkeys(str(path) for path in package_directories)
    )

    inspected = []
    for directory_text in package_directories:
        directory = Path(directory_text)
        wrapper = directory / "__init__.py"
        compatible = []
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = directory / ("_C" + suffix)
            if candidate.is_file():
                compatible.append(candidate.absolute())
        compatible = list(dict.fromkeys(compatible))
        if not compatible:
            compatible = sorted(
                path.absolute()
                for path in directory.glob("_C*.so")
                if path.is_file()
            )
        inspected.append(str(directory))
        if len(compatible) > 1:
            raise ProvenanceError(
                "multiple import-compatible {} extensions found in {}: {}"
                .format(
                    package_name,
                    directory,
                    ", ".join(str(path) for path in compatible),
                )
            )
        if len(compatible) == 1 and (
            wrapper.is_file() or not wrapper_required
        ):
            return (
                wrapper.absolute(),
                compatible[0],
                bool(wrapper.is_file()),
            )
    raise ProvenanceError(
        "cannot pre-locate {} wrapper/_C extension in {}".format(
            package_name, ", ".join(inspected)
        )
    )


def _locate_rasterizer_files():
    """Locate the Raster wrapper and extension without importing either."""

    root = Path(__file__).absolute().parent
    wrapper, binary, _wrapper_exists = _locate_extension_package_files(
        "diff_gaussian_rasterization",
        root
        / "submodules"
        / "depth-diff-gaussian-rasterization"
        / "diff_gaussian_rasterization",
        wrapper_required=True,
    )
    return wrapper, binary


def _locate_binary_packages(root, rasterizer_paths=None):
    if rasterizer_paths is None:
        rasterizer_paths = _locate_rasterizer_files()
    head_wrapper, head_binary, head_wrapper_exists = (
        _locate_extension_package_files(
            "tacker_4dgs_head",
            root / "tacker_ext" / "tacker_4dgs_head",
            wrapper_required=True,
        )
    )
    simple_wrapper, simple_binary, simple_wrapper_exists = (
        _locate_extension_package_files(
            "simple_knn",
            root / "submodules" / "simple-knn" / "simple_knn",
            wrapper_required=False,
        )
    )
    return {
        "rasterizer": {
            "package": "diff_gaussian_rasterization",
            "wrapper": Path(rasterizer_paths[0]).absolute(),
            "wrapper_exists": True,
            "binary": Path(rasterizer_paths[1]).absolute(),
            "wrapper_role": "rasterizer.wrapper",
            "binary_role": "rasterizer.binary",
        },
        "head": {
            "package": "tacker_4dgs_head",
            "wrapper": head_wrapper,
            "wrapper_exists": head_wrapper_exists,
            "binary": head_binary,
            "wrapper_role": "head.wrapper",
            "binary_role": "head.binary",
        },
        "simple_knn": {
            "package": "simple_knn",
            "wrapper": simple_wrapper,
            "wrapper_exists": simple_wrapper_exists,
            "binary": simple_binary,
            "wrapper_role": "simple_knn.wrapper",
            "binary_role": "simple_knn.binary",
        },
    }


def _source_role(module_name, is_package):
    if module_name == "profile_render":
        return "source.profile_render"
    if is_package:
        return "source.{}.__init__".format(module_name)
    return "source.{}".format(module_name)


def _python_modules_under(package_name, directory):
    modules = {}
    directory = Path(directory).absolute()
    for source in sorted(directory.rglob("*.py")):
        relative = source.relative_to(directory)
        parts = list(relative.parts)
        is_package = parts[-1] == "__init__.py"
        if is_package:
            module_parts = parts[:-1]
        else:
            module_parts = parts[:-1] + [Path(parts[-1]).stem]
        if not module_parts:
            module_name = package_name
        else:
            module_name = package_name + "." + ".".join(module_parts)
        modules[module_name] = (source.absolute(), is_package)
    return modules


def _consume_profile_render_bootstrap():
    """Validate and consume the exact source image executed by bootstrap."""

    value = _PROFILE_RENDER_BOOTSTRAP
    if not isinstance(value, dict) or value.get("protocol") != 1:
        raise ProvenanceError(
            "profile_render CLI requires scripts/run_profile_render_sealed.py"
        )
    record = value.get("record")
    raw = value.get("bytes")
    execution = value.get("execution")
    if not isinstance(record, dict) or not isinstance(raw, bytes):
        raise ProvenanceError("profile_render bootstrap seal is malformed")
    digest = hashlib.sha256(raw).hexdigest()
    if (
        not isinstance(execution, dict)
        or execution.get("compiled_sha256") != digest
        or record.get("sha256") != digest
        or record.get("size_bytes") != len(raw)
        or not record.get("exists")
    ):
        raise ProvenanceError(
            "profile_render bootstrap execution/source binding is invalid"
        )
    expected_path = str(Path(__file__).expanduser().absolute())
    if record.get("path") != expected_path:
        raise ProvenanceError(
            "profile_render bootstrap path does not match __file__"
        )
    current, current_raw = _snapshot_file_bytes(
        expected_path, "source.profile_render", required=True
    )
    for key in ("path", "exists", "stat", "size_bytes", "sha256"):
        if current.get(key) != record.get(key):
            raise ProvenanceError(
                "profile_render changed after bootstrap snapshot ({})".format(
                    key
                )
            )
    if current_raw != raw:
        raise ProvenanceError(
            "profile_render bytes differ from bootstrap execution bytes"
        )
    consumed = dict(record)
    consumed["role"] = "source.profile_render"
    return consumed, raw, dict(execution)


def _capture_pre_import_snapshot(
    args,
    source_paths=None,
    rasterizer_paths=None,
    binary_package_paths=None,
):
    """Seal all first-party sources, configs, profiles, and CUDA binaries."""

    root = Path(__file__).absolute().parent
    production_capture = source_paths is None
    bootstrap_record = None
    bootstrap_raw = None
    bootstrap_execution = None
    python_modules = {}
    if production_capture:
        (
            bootstrap_record,
            bootstrap_raw,
            bootstrap_execution,
        ) = _consume_profile_render_bootstrap()
        source_paths = {}
        for package_name in ("arguments", "gaussian_renderer", "scene", "utils"):
            python_modules.update(
                _python_modules_under(package_name, root / package_name)
            )
        for module_name, description in sorted(python_modules.items()):
            role = _source_role(module_name, description[1])
            source_paths[role] = description[0]
    if binary_package_paths is None:
        if production_capture:
            binary_package_paths = _locate_binary_packages(
                root, rasterizer_paths=rasterizer_paths
            )
        else:
            if rasterizer_paths is None:
                rasterizer_paths = _locate_rasterizer_files()
            binary_package_paths = {
                "rasterizer": {
                    "package": "diff_gaussian_rasterization",
                    "wrapper": Path(rasterizer_paths[0]).absolute(),
                    "wrapper_exists": True,
                    "binary": Path(rasterizer_paths[1]).absolute(),
                    "wrapper_role": "rasterizer.wrapper",
                    "binary_role": "rasterizer.binary",
                }
            }

    public_files = []
    sealed_bytes = {}
    identity_by_path = {}
    config_bases = {}

    def capture(path, role, required=True):
        record, raw = _snapshot_file_bytes(path, role, required=required)
        previous = identity_by_path.get(record["path"])
        if previous is not None:
            for key in ("exists", "stat", "size_bytes", "sha256"):
                if previous.get(key) != record.get(key):
                    raise ProvenanceError(
                        "one pre-import path produced multiple snapshots: {}"
                        .format(record["path"])
                    )
            if raw is not None and raw != sealed_bytes[record["path"]]:
                raise ProvenanceError(
                    "one pre-import path produced multiple byte images: {}"
                    .format(record["path"])
                )
        else:
            identity_by_path[record["path"]] = record
        public_files.append(record)
        if raw is not None and record["path"] not in sealed_bytes:
            sealed_bytes[record["path"]] = raw
        return record, raw

    def capture_presealed(record, raw):
        normalized = dict(record)
        role = normalized["role"]
        previous = identity_by_path.get(normalized["path"])
        if previous is not None and previous != normalized:
            raise ProvenanceError(
                "one pre-import path produced multiple snapshots: {}".format(
                    normalized["path"]
                )
            )
        identity_by_path[normalized["path"]] = normalized
        public_files.append(normalized)
        sealed_bytes[normalized["path"]] = raw
        return normalized

    if bootstrap_record is not None:
        capture_presealed(bootstrap_record, bootstrap_raw)
    for role, path in source_paths.items():
        capture(path, role)
    namespace_roles = {}
    if production_capture and "utils" not in python_modules:
        role = "source.namespace.utils"
        record, _raw = capture(
            root / "utils" / "__init__.py", role, required=False
        )
        if record["exists"]:
            raise ProvenanceError(
                "utils package source escaped Python module enumeration"
            )
        namespace_roles["utils"] = role

    binary_packages = {}
    for component, description in sorted(binary_package_paths.items()):
        wrapper_record, wrapper_raw = capture(
            description["wrapper"],
            description["wrapper_role"],
            required=bool(description.get("wrapper_exists", True)),
        )
        binary_record, _binary_raw = capture(
            description["binary"], description["binary_role"], required=True
        )
        package_name = description["package"]
        binary_packages[component] = {
            "package": package_name,
            "wrapper_role": description["wrapper_role"],
            "binary_role": description["binary_role"],
            "wrapper_exists": bool(wrapper_record["exists"]),
        }
        if wrapper_record["exists"]:
            module_description = (
                Path(wrapper_record["path"]),
                True,
            )
            existing = python_modules.get(package_name)
            if existing is not None and existing != module_description:
                raise ProvenanceError(
                    "multiple sealed sources claim module {}".format(
                        package_name
                    )
                )
            python_modules[package_name] = module_description
        if wrapper_record["exists"]:
            extension_package_modules = _python_modules_under(
                package_name, Path(wrapper_record["path"]).parent
            )
            for module_name, module_description in sorted(
                extension_package_modules.items()
            ):
                if module_name == package_name:
                    continue
                role = _source_role(
                    module_name, module_description[1]
                )
                capture(module_description[0], role)
                python_modules[module_name] = module_description

    tacker_profile = getattr(args, "tacker_profile", None)
    if tacker_profile is not None:
        capture(tacker_profile, "profile.tacker", required=False)
    qualification_profile = getattr(args, "qualification_profile", None)
    if qualification_profile is not None:
        capture(
            qualification_profile,
            "profile.qualification",
            required=True,
        )

    active = []
    visited = set()

    def capture_config(path):
        resolved = Path(
            os.path.abspath(str(Path(path).expanduser()))
        )
        key = str(resolved)
        if key in active:
            raise ProvenanceError(
                "configuration _base_ cycle includes {}".format(resolved)
            )
        if key in visited:
            return
        active.append(key)
        role = "config.explicit[{}]".format(len(config_bases))
        record, raw = capture(resolved, role)
        bases = _python_config_bases(resolved, raw)
        config_bases[record["path"]] = [str(base) for base in bases]
        for base in bases:
            capture_config(base)
        active.pop()
        visited.add(key)

    configs = getattr(args, "configs", None)
    if configs is not None:
        capture_config(configs)

    # ``get_combined_args`` evaluates this before the explicit config merge.
    model_path = getattr(args, "model_path", None)
    if model_path:
        capture(
            Path(os.path.abspath(str(Path(model_path).expanduser())))
            / "cfg_args",
            "config.model_cfg_args",
            required=True,
        )

    public = {
        "captured_before_heavy_import": True,
        "captured_before_config_parse": True,
        "captured_before_config_execution": True,
        "files": public_files,
        "config_bases": config_bases,
        "rasterizer_binding": None,
        "binary_bindings": [],
        "python_source_binding": None,
        "bootstrap_binding": (
            {
                "protocol": 1,
                "compiled_sha256": bootstrap_execution["compiled_sha256"],
                "path": bootstrap_record["path"],
                "matches_pre_import_snapshot": True,
            }
            if bootstrap_execution is not None
            else None
        ),
    }
    artifact_roles_by_package = {
        description["package"]: description["wrapper_role"]
        for description in binary_packages.values()
        if description["wrapper_exists"]
    }
    python_module_records = {}
    for module_name, description in sorted(python_modules.items()):
        role = artifact_roles_by_package.get(
            module_name, _source_role(module_name, description[1])
        )
        matches = [
            record for record in public_files if record["role"] == role
        ]
        if len(matches) == 1:
            python_module_records[module_name] = {
                "path": matches[0]["path"],
                "role": role,
                "is_package": bool(description[1]),
                "synthetic": False,
            }
    # ``utils`` and an unwrapped simple_knn are namespace packages on disk.
    # A sealed empty package prevents their parent package resolution from
    # consulting a mutable filesystem while child modules/extensions import.
    if production_capture and "utils" not in python_module_records:
        python_module_records["utils"] = {
            "path": str((root / "utils" / "__init__.py").absolute()),
            "role": namespace_roles["utils"],
            "is_package": True,
            "synthetic": True,
        }
    for description in binary_packages.values():
        package_name = description["package"]
        if not description["wrapper_exists"]:
            wrapper_record = _record_for_role(
                {"files": public_files}, description["wrapper_role"]
            )
            python_module_records[package_name] = {
                "path": wrapper_record["path"],
                "role": description["wrapper_role"],
                "is_package": True,
                "synthetic": True,
            }
    internal = {
        "bytes_by_path": sealed_bytes,
        "config_bases": config_bases,
        "python_modules": python_module_records,
        "binary_packages": binary_packages,
    }
    return public, internal


def _record_for_role(pre_import, role):
    matches = [
        record for record in pre_import["files"] if record["role"] == role
    ]
    if len(matches) != 1:
        raise ProvenanceError(
            "pre-import snapshot requires one {} record".format(role)
        )
    return matches[0]


class _SnapshotSourceLoader(importlib.abc.Loader):
    """Execute a Python module from sealed bytes while retaining its origin."""

    def __init__(self, fullname, origin, raw, is_package):
        self.fullname = fullname
        self.origin = str(origin)
        self.raw = raw
        self._is_package = bool(is_package)
        self.snapshot_sha256 = hashlib.sha256(raw).hexdigest()

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        code = compile(self.raw, self.origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    def is_package(self, fullname):
        return self._is_package


class _SnapshotImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, sources, binary_paths):
        self.sources = sources
        if isinstance(binary_paths, dict):
            self.binary_paths = {
                name: str(path) for name, path in binary_paths.items()
            }
        else:
            # Compatibility for the narrow source-loader CPU contract.
            self.binary_paths = {
                "diff_gaussian_rasterization._C": str(binary_paths)
            }
        self.source_loaders = {}
        self.binary_loaders = {}
        self.protected_roots = {
            name.split(".", 1)[0]
            for name in set(self.sources) | set(self.binary_paths)
        }

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            origin, raw, is_package = self.sources[fullname]
            loader = _SnapshotSourceLoader(
                fullname, origin, raw, is_package
            )
            self.source_loaders[fullname] = loader
            spec = importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=str(origin),
                is_package=is_package,
            )
            # A generic Loader does not imply ``has_location``.  These sealed
            # modules deliberately retain their original __file__ so existing
            # relative profile/asset discovery keeps its historical meaning.
            spec.has_location = True
            if is_package:
                spec.submodule_search_locations = [str(Path(origin).parent)]
            return spec
        if fullname in self.binary_paths:
            binary_path = self.binary_paths[fullname]
            loader = importlib.machinery.ExtensionFileLoader(
                fullname, binary_path
            )
            self.binary_loaders[fullname] = loader
            return importlib.util.spec_from_file_location(
                fullname, binary_path, loader=loader
            )
        if fullname.split(".", 1)[0] in self.protected_roots:
            raise ImportError(
                "first-party module was not present in pre-import snapshot: {}"
                .format(fullname)
            )
        return None


def _write_private_binary_snapshot(raw, basename, component="cuda"):
    directory = tempfile.TemporaryDirectory(
        prefix="4dgs-{}-snapshot-".format(component.replace("_", "-"))
    )
    _BINARY_SNAPSHOT_DIRECTORIES.append(directory)
    target = Path(directory.name) / basename
    descriptor = os.open(
        str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(
                descriptor, raw[offset : offset + 1024 * 1024]
            )
            if written <= 0:
                raise ProvenanceError(
                    "cannot write private {} snapshot".format(component)
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target.resolve()


def _validate_snapshot_pair(pre_import, internal):
    if (pre_import is None) != (internal is None):
        raise ProvenanceError(
            "pre_import_snapshot and pre_import_internal must both be set or both be None"
        )
    if pre_import is None:
        return
    if not isinstance(pre_import, dict) or not isinstance(internal, dict):
        raise ProvenanceError("pre-import snapshot/internal must be dictionaries")
    byte_map = internal.get("bytes_by_path")
    if not isinstance(byte_map, dict):
        raise ProvenanceError("pre-import internal byte map is missing")
    for record in pre_import.get("files", []):
        raw = byte_map.get(record.get("path"))
        if record.get("exists"):
            if not isinstance(raw, bytes):
                raise ProvenanceError(
                    "sealed bytes are missing for {}".format(record.get("role"))
                )
            if (
                len(raw) != record.get("size_bytes")
                or hashlib.sha256(raw).hexdigest() != record.get("sha256")
            ):
                raise ProvenanceError(
                    "sealed bytes disagree with record for {}".format(
                        record.get("role")
                    )
                )
        elif raw is not None:
            raise ProvenanceError(
                "absent sealed input unexpectedly has bytes: {}".format(
                    record.get("role")
                )
            )


def _activate_runtime_imports(pre_import, internal):
    """Import all first-party code and CUDA extensions from sealed bytes."""

    global torch
    global _rasterizer_module
    global ModelHiddenParams, ModelParams, PipelineParams
    global GaussianModel, TackerRenderer, TwoStreamRenderer
    global deform_for_render, prepare_render_context, rasterize_state, render
    global Scene, safe_state, nvtx_range

    _validate_snapshot_pair(pre_import, internal)
    module_records = internal.get("python_modules", {})
    binary_packages = internal.get("binary_packages", {})
    required_binary_components = {"rasterizer", "head", "simple_knn"}
    if set(binary_packages) != required_binary_components:
        raise ProvenanceError(
            "runtime snapshot must contain exactly Raster, head, and simple_knn binaries"
        )
    sources = {}
    for module_name, description in module_records.items():
        if description.get("synthetic"):
            raw = b""
        else:
            raw = internal["bytes_by_path"].get(description["path"])
            if not isinstance(raw, bytes):
                raise ProvenanceError(
                    "sealed source bytes missing for module {}".format(
                        module_name
                    )
                )
        sources[module_name] = (
            description["path"],
            raw,
            bool(description["is_package"]),
        )

    binary_paths = {}
    binary_records = {}
    for component, description in sorted(binary_packages.items()):
        record = _record_for_role(pre_import, description["binary_role"])
        raw = internal["bytes_by_path"].get(record["path"])
        if not isinstance(raw, bytes):
            raise ProvenanceError(
                "sealed {} binary bytes are missing".format(component)
            )
        module_name = description["package"] + "._C"
        binary_paths[module_name] = _write_private_binary_snapshot(
            raw, Path(record["path"]).name, component=component
        )
        binary_records[module_name] = (component, record, raw)

    for module_name in sorted(set(sources) | set(binary_paths)):
        if module_name in sys.modules:
            raise ProvenanceError(
                "heavy module was imported before pre-import snapshot: {}".format(
                    module_name
                )
            )
    finder = _SnapshotImportFinder(sources, binary_paths)
    sys.meta_path.insert(0, finder)
    try:
        torch_module = importlib.import_module("torch")
        rasterizer_module = importlib.import_module(
            "diff_gaussian_rasterization"
        )
        head_module = importlib.import_module("tacker_4dgs_head")
        head_backend = importlib.import_module("tacker_4dgs_head._C")
        arguments_module = importlib.import_module("arguments")
        renderer_module = importlib.import_module("gaussian_renderer")
        scene_module = importlib.import_module("scene")
        simple_backend = importlib.import_module("simple_knn._C")
        general_utils_module = importlib.import_module("utils.general_utils")
        profiling_utils_module = importlib.import_module(
            "utils.profiling_utils"
        )
    except Exception:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        raise
    # Keep the finder installed: any later lazy first-party import must also
    # execute sealed bytes.  Holding it in ``internal`` binds its lifetime to
    # this profiling run.
    internal["runtime_finder"] = finder

    loaded_binary_modules = {
        "diff_gaussian_rasterization._C": rasterizer_module._C,
        "tacker_4dgs_head._C": head_backend,
        "simple_knn._C": simple_backend,
    }
    binary_bindings = []
    for module_name, loaded_module in sorted(loaded_binary_modules.items()):
        component, origin_record, origin_raw = binary_records[module_name]
        loaded_binary = Path(loaded_module.__file__).absolute()
        loaded_record, loaded_raw = _snapshot_file_bytes(
            loaded_binary,
            "runtime.{}.binary".format(component),
            required=True,
        )
        if (
            loaded_record["sha256"] != origin_record["sha256"]
            or loaded_record["size_bytes"] != origin_record["size_bytes"]
            or loaded_raw != origin_raw
            or str(loaded_binary) != str(binary_paths[module_name])
        ):
            raise ProvenanceError(
                "loaded {} extension is not its pre-import byte snapshot"
                .format(component)
            )
        binary_bindings.append(
            {
                "component": component,
                "module": module_name,
                "strategy": "private_extension_copy_from_pre_import_bytes",
                "origin_path": origin_record["path"],
                "origin_role": origin_record["role"],
                "loaded_path": str(loaded_binary),
                "loaded_stat": loaded_record["stat"],
                "size_bytes": loaded_record["size_bytes"],
                "sha256": loaded_record["sha256"],
                "matches_pre_import_snapshot": True,
            }
        )

    loaded_source_bindings = []
    for module_name, source in sorted(sources.items()):
        if module_name not in sys.modules:
            continue
        loader = finder.source_loaders.get(module_name)
        expected_sha256 = hashlib.sha256(source[1]).hexdigest()
        if loader is None or loader.snapshot_sha256 != expected_sha256:
            raise ProvenanceError(
                "module was not executed from its pre-import snapshot: {}".format(
                    module_name
                )
            )
        description = module_records[module_name]
        loaded_source_bindings.append(
            {
                "module": module_name,
                "origin_path": description["path"],
                "role": description["role"],
                "sha256": expected_sha256,
                "synthetic_namespace": bool(description["synthetic"]),
                "matches_pre_import_snapshot": True,
            }
        )
    first_party_roots = {
        "arguments", "gaussian_renderer", "scene", "utils"
    }
    for module_name, module in sorted(sys.modules.items()):
        if module_name.split(".", 1)[0] not in first_party_roots:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and str(module_file).endswith(".py") and module_name not in sources:
            raise ProvenanceError(
                "first-party module escaped source snapshot: {}".format(
                    module_name
                )
            )

    pre_import["binary_bindings"] = binary_bindings
    raster_bindings = [
        item for item in binary_bindings if item["component"] == "rasterizer"
    ]
    if len(raster_bindings) != 1:
        raise ProvenanceError("one Raster binary binding is required")
    pre_import["rasterizer_binding"] = dict(raster_bindings[0])
    pre_import["rasterizer_binding"]["sealed_source_modules"] = sorted(
        sources
    )
    pre_import["python_source_binding"] = {
        "strategy": "meta_path_loaders_from_pre_import_bytes",
        "sealed_module_count": len(sources),
        "loaded_module_count": len(loaded_source_bindings),
        "loaded_modules": loaded_source_bindings,
        "all_loaded_first_party_modules_match_snapshots": True,
    }

    torch = torch_module
    _rasterizer_module = rasterizer_module
    ModelHiddenParams = arguments_module.ModelHiddenParams
    ModelParams = arguments_module.ModelParams
    PipelineParams = arguments_module.PipelineParams
    GaussianModel = renderer_module.GaussianModel
    TackerRenderer = renderer_module.TackerRenderer
    TwoStreamRenderer = renderer_module.TwoStreamRenderer
    deform_for_render = renderer_module.deform_for_render
    prepare_render_context = renderer_module.prepare_render_context
    rasterize_state = renderer_module.rasterize_state
    render = renderer_module.render
    Scene = scene_module.Scene
    safe_state = general_utils_module.safe_state
    nvtx_range = profiling_utils_module.nvtx_range


def _merge_config_dicts(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_snapshot(path, pre_import, internal, active=None):
    """Execute one recursive config exclusively from pre-import bytes."""

    resolved = Path(os.path.abspath(str(Path(path).expanduser())))
    key = str(resolved)
    if active is None:
        active = []
    if key in active:
        raise ProvenanceError(
            "configuration _base_ cycle includes {}".format(resolved)
        )
    raw = internal["bytes_by_path"].get(key)
    if raw is None or key not in internal["config_bases"]:
        raise ProvenanceError(
            "configuration was not present in pre-import snapshot: {}".format(
                resolved
            )
        )
    try:
        source = raw.decode("utf-8")
        namespace = {}
        exec(compile(source, key, "exec"), {}, namespace)
    except (UnicodeError, SyntaxError) as error:
        raise ProvenanceError(
            "cannot execute snapshotted configuration {}: {}".format(
                resolved, error
            )
        )

    active.append(key)
    config = {}
    for base_path in internal["config_bases"][key]:
        config = _merge_config_dicts(
            config,
            _load_config_snapshot(
                base_path, pre_import, internal, active=active
            ),
        )
    active.pop()
    namespace.pop("_base_", None)
    local = {
        name: value
        for name, value in namespace.items()
        if not name.startswith("__")
    }
    return _merge_config_dicts(config, local)


def _merge_hparams(args, config):
    for param in (
        "OptimizationParams",
        "ModelHiddenParams",
        "ModelParams",
        "PipelineParams",
    ):
        if param in config:
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    return args


def _combined_args_from_snapshot(parser, argv, pre_import, internal):
    """Preserve ``get_combined_args`` precedence without re-reading cfg_args."""

    command_line = parser.parse_args(argv)
    records = [
        record
        for record in pre_import["files"]
        if record["role"] == "config.model_cfg_args"
    ]
    config_namespace = Namespace()
    if records:
        record = records[0]
        try:
            source = internal["bytes_by_path"][record["path"]].decode("utf-8")
            config_namespace = eval(source, {"Namespace": Namespace}, {})
        except (UnicodeError, SyntaxError, TypeError, ValueError) as error:
            raise ProvenanceError(
                "cannot evaluate snapshotted cfg_args {}: {}".format(
                    record["path"], error
                )
            )
        if not isinstance(config_namespace, Namespace):
            raise ProvenanceError("snapshotted cfg_args must evaluate to Namespace")
    merged = vars(config_namespace).copy()
    for key, value in vars(command_line).items():
        if value is not None or key not in merged:
            merged[key] = value
    return Namespace(**merged)


def _verify_runtime_import_bindings(pre_import, internal):
    finder = internal.get("runtime_finder")
    if finder is None or not sys.meta_path or sys.meta_path[0] is not finder:
        raise ProvenanceError(
            "snapshot import finder changed priority during profiling"
        )
    source_checks = []
    for expected in pre_import.get("python_source_binding", {}).get(
        "loaded_modules", []
    ):
        module = sys.modules.get(expected["module"])
        loader = finder.source_loaders.get(expected["module"])
        checks = {
            "module_present": module is not None,
            "loader": module is not None
            and getattr(module, "__loader__", None) is loader,
            "origin": module is not None
            and getattr(getattr(module, "__spec__", None), "origin", None)
            == expected["origin_path"],
            "sha256": loader is not None
            and loader.snapshot_sha256 == expected["sha256"],
        }
        if not all(checks.values()):
            raise ProvenanceError(
                "sealed source module binding changed during profiling: {}"
                .format(expected["module"])
            )
        source_checks.append(
            {"module": expected["module"], "checks": checks, "unchanged": True}
        )
    binary_checks = []
    for expected in pre_import.get("binary_bindings", []):
        module = sys.modules.get(expected["module"])
        loader = finder.binary_loaders.get(expected["module"])
        checks = {
            "module_present": module is not None,
            "loader": module is not None
            and getattr(module, "__loader__", None) is loader,
            "origin": module is not None
            and getattr(getattr(module, "__spec__", None), "origin", None)
            == expected["loaded_path"],
        }
        if not all(checks.values()):
            raise ProvenanceError(
                "sealed binary module binding changed during profiling: {}"
                .format(expected["module"])
            )
        binary_checks.append(
            {"module": expected["module"], "checks": checks, "unchanged": True}
        )
    return {
        "verified": True,
        "finder_is_first": True,
        "source_modules": source_checks,
        "binary_modules": binary_checks,
    }


def _verify_pre_import_snapshot(pre_import, internal=None):
    """Recheck every origin and loaded private binary before publication."""

    verifications = []
    for expected in pre_import.get("files", []):
        if not expected.get("exists"):
            input_path = expected["input_path"]
            try:
                resolved_now = str(Path(input_path).resolve())
                os.lstat(input_path)
            except OSError as error:
                if error.errno != getattr(os, "ENOENT", 2):
                    raise ProvenanceError(
                        "cannot verify sealed absence {} ({}): {}".format(
                            expected["role"], input_path, error
                        )
                    )
            else:
                # Never open a path whose absence was sealed.  A newly-created
                # file is already a contract violation, and reading it would
                # introduce both a second-consumption bug and an avoidable
                # attacker-controlled allocation before metadata publication.
                raise ProvenanceError(
                    "pre-import input changed during profiling: {} (exists)"
                    .format(input_path)
                )
            if resolved_now != expected["path"]:
                raise ProvenanceError(
                    "pre-import input changed during profiling: {} (path)"
                    .format(input_path)
                )
            actual = {
                "path": expected["path"],
                "exists": False,
                "stat": None,
                "size_bytes": None,
                "sha256": None,
            }
        else:
            actual, _raw = _snapshot_file_bytes(
                expected["input_path"], expected["role"], required=False
            )
        checks = {
            "path": actual.get("path") == expected.get("path"),
            "exists": actual.get("exists") == expected.get("exists"),
            "stat": actual.get("stat") == expected.get("stat"),
            "size_bytes": actual.get("size_bytes")
            == expected.get("size_bytes"),
            "sha256": actual.get("sha256") == expected.get("sha256"),
        }
        verification = {
            "role": expected["role"],
            "path": expected["path"],
            "checks": checks,
            "unchanged": all(checks.values()),
        }
        verifications.append(verification)
        if not verification["unchanged"]:
            failed = sorted(
                name for name, passed in checks.items() if not passed
            )
            raise ProvenanceError(
                "pre-import input changed during profiling: {} ({})".format(
                    expected["path"], ", ".join(failed)
                )
            )
    loaded_binary_verifications = []
    for expected in pre_import.get("binary_bindings", []):
        actual, _raw = _snapshot_file_bytes(
            expected["loaded_path"],
            "runtime.{}.binary".format(expected["component"]),
            required=True,
        )
        checks = {
            "path": actual.get("path") == expected.get("loaded_path"),
            "stat": actual.get("stat") == expected.get("loaded_stat"),
            "size_bytes": actual.get("size_bytes")
            == expected.get("size_bytes"),
            "sha256": actual.get("sha256") == expected.get("sha256"),
        }
        verification = {
            "component": expected["component"],
            "module": expected["module"],
            "path": expected["loaded_path"],
            "checks": checks,
            "unchanged": all(checks.values()),
        }
        loaded_binary_verifications.append(verification)
        if not verification["unchanged"]:
            failed = sorted(
                name for name, passed in checks.items() if not passed
            )
            raise ProvenanceError(
                "loaded private {} binary changed during profiling: {} ({})"
                .format(
                    expected["component"],
                    expected["loaded_path"],
                    ", ".join(failed),
                )
            )
    runtime_import_binding = None
    if internal is not None:
        runtime_import_binding = _verify_runtime_import_bindings(
            pre_import, internal
        )
    return {
        "verified": True,
        "verification_point": "after_run_before_metadata_publish",
        "file_count": len(verifications),
        "files": verifications,
        "loaded_binary_count": len(loaded_binary_verifications),
        "loaded_binaries": loaded_binary_verifications,
        "runtime_import_binding": runtime_import_binding,
    }


def _percentile(values, percentile):
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _safe_command(command, cwd=None, timeout=5):
    """Run a metadata-only command without making profiling depend on it."""

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, str(error)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "exit status {}".format(
            completed.returncode
        )
        return None, detail
    # Preserve leading spaces: ``git submodule status`` uses its first byte as
    # a state marker, including a literal space for a clean recorded gitlink.
    return completed.stdout.rstrip("\r\n"), None


def _safe_file_sha256(path, errors, label):
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).expanduser().open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError) as error:
        errors.append("{}: {}".format(label, error))
        return None


def _atomic_write_json(path, value):
    """Atomically replace a metadata file with finite, fsynced JSON."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name), suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(target))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _parse_submodule_status(output):
    submodules = []
    for line in output.splitlines():
        if not line:
            continue
        state = line[0]
        fields = line[1:].strip().split()
        if len(fields) < 2:
            continue
        submodules.append(
            {
                "path": fields[1],
                "commit": fields[0],
                "source": "git",
                "status": {
                    " ": "recorded",
                    "-": "uninitialized",
                    "+": "different_commit",
                    "U": "merge_conflict",
                }.get(state, "unknown"),
            }
        )
    return submodules


def _environment_submodule_commits():
    mappings = (
        (
            "submodules/depth-diff-gaussian-rasterization",
            "FOURDGS_RASTERIZER_COMMIT",
        ),
        ("submodules/simple-knn", "FOURDGS_SIMPLE_KNN_COMMIT"),
    )
    return [
        {
            "path": path,
            "commit": os.environ.get(variable),
            "source": "environment" if os.environ.get(variable) else None,
            "status": "provided" if os.environ.get(variable) else "unavailable",
        }
        for path, variable in mappings
    ]


def _collect_repository_metadata(repo_root, errors):
    metadata = {
        "source_tree": str(repo_root),
        "commit": None,
        "commit_source": None,
        "dirty": None,
        "git_error": None,
        "submodules": [],
    }
    if not (repo_root / ".git").exists():
        git_error = "source tree has no .git metadata"
        metadata["git_error"] = git_error
        errors.append("git commit: {}".format(git_error))
        source_commit = os.environ.get("FOURDGS_SOURCE_COMMIT")
        if source_commit:
            metadata["commit"] = source_commit
            metadata["commit_source"] = "environment"
        metadata["submodules"] = _environment_submodule_commits()
        return metadata

    commit, error = _safe_command(("git", "rev-parse", "HEAD"), cwd=repo_root)
    if error is None:
        metadata["commit"] = commit
        metadata["commit_source"] = "git"
    else:
        metadata["git_error"] = error
        errors.append("git commit: {}".format(error))
        source_commit = os.environ.get("FOURDGS_SOURCE_COMMIT")
        if source_commit:
            metadata["commit"] = source_commit
            metadata["commit_source"] = "environment"

    status, status_error = _safe_command(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=repo_root,
    )
    if status_error is None:
        metadata["dirty"] = bool(status)
    else:
        errors.append("git status: {}".format(status_error))
        if metadata["git_error"] is None:
            metadata["git_error"] = status_error

    submodule_output, submodule_error = _safe_command(
        ("git", "submodule", "status", "--recursive"), cwd=repo_root
    )
    if submodule_error is None:
        metadata["submodules"] = _parse_submodule_status(submodule_output)
    else:
        errors.append("git submodule status: {}".format(submodule_error))
    known_paths = {item["path"] for item in metadata["submodules"]}
    metadata["submodules"].extend(
        item
        for item in _environment_submodule_commits()
        if item["path"] not in known_paths
    )
    return metadata


_NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "temperature.gpu",
    "power.management",
    "power.draw",
    "power.limit",
)


def _nvidia_smi_metadata(logical_device_index, errors):
    command = (
        "nvidia-smi",
        "--query-gpu={}".format(",".join(_NVIDIA_SMI_FIELDS)),
        "--format=csv,noheader,nounits",
    )
    output, error = _safe_command(command)
    if error is not None:
        errors.append("nvidia-smi: {}".format(error))
        return None

    rows = []
    try:
        for values in csv.reader(io.StringIO(output)):
            if len(values) != len(_NVIDIA_SMI_FIELDS):
                continue
            rows.append(
                {
                    key: value.strip()
                    for key, value in zip(_NVIDIA_SMI_FIELDS, values)
                }
            )
    except (csv.Error, UnicodeError) as parse_error:
        errors.append("nvidia-smi output: {}".format(parse_error))
        return None
    if not rows:
        errors.append("nvidia-smi: query returned no parseable GPU rows")
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    selector = None
    if visible_devices:
        selectors = [item.strip() for item in visible_devices.split(",")]
        if logical_device_index < len(selectors):
            selector = selectors[logical_device_index]

    selected = None
    if selector is not None:
        for row in rows:
            if row["index"] == selector or row["uuid"] == selector:
                selected = row
                break
    elif logical_device_index < len(rows):
        selected = rows[logical_device_index]
    if selected is None and len(rows) == 1:
        selected = rows[0]
    if selected is None:
        errors.append(
            "nvidia-smi: cannot map logical CUDA device {} to a physical GPU".format(
                logical_device_index
            )
        )
    return selected


def _collect_environment_metadata(errors):
    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pytorch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": None,
        "nvidia_smi": None,
    }
    try:
        environment["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as error:  # Metadata must never invalidate a measured run.
        errors.append("cuDNN version: {}".format(error))

    logical_device_index = 0
    gpu = {}
    try:
        logical_device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(logical_device_index)
        gpu = {
            "logical_device_index": logical_device_index,
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
        }
    except Exception as error:  # Metadata must never invalidate a measured run.
        errors.append("CUDA device properties: {}".format(error))
    environment["gpu"] = gpu or None
    environment["nvidia_smi"] = _nvidia_smi_metadata(
        logical_device_index, errors
    )
    if environment["nvidia_smi"] is not None:
        environment["cuda_driver_version"] = environment["nvidia_smi"].get(
            "driver_version"
        )
    else:
        environment["cuda_driver_version"] = None
    return environment


def _collect_reproducibility_metadata(
    args,
    profile_manifest_sha256,
    tacker_profile_sha256_snapshot=None,
    qualification_profile_sha256_snapshot=None,
    pre_import_snapshot=None,
):
    """Snapshot provenance before warmup/timing; return partial data on failure."""

    errors = []
    repo_root = Path(__file__).absolute().parent
    try:
        repository = _collect_repository_metadata(repo_root, errors)
    except Exception as error:  # Keep optional provenance best-effort.
        errors.append("repository metadata: {}".format(error))
        repository = {
            "source_tree": str(repo_root),
            "commit": os.environ.get("FOURDGS_SOURCE_COMMIT"),
            "commit_source": (
                "environment" if os.environ.get("FOURDGS_SOURCE_COMMIT") else None
            ),
            "dirty": None,
            "git_error": str(error),
            "submodules": _environment_submodule_commits(),
        }
    try:
        environment = _collect_environment_metadata(errors)
    except Exception as error:  # Keep optional provenance best-effort.
        errors.append("environment metadata: {}".format(error))
        environment = {
            "pytorch_version": str(getattr(torch, "__version__", "unknown")),
            "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_driver_version": None,
        }

    if pre_import_snapshot is None:
        # Retain a narrow compatibility path for direct programmatic callers;
        # the CLI always supplies the strict pre-import seal.
        profile_render_sha256 = _safe_file_sha256(
            Path(__file__).resolve(), errors, "profile_render.py SHA-256"
        )
        source_files = {
            "profile_render.py": profile_render_sha256,
            "gaussian_renderer/__init__.py": _safe_file_sha256(
                repo_root / "gaussian_renderer" / "__init__.py",
                errors,
                "gaussian_renderer/__init__.py SHA-256",
            ),
            "gaussian_renderer/tacker_pipeline.py": _safe_file_sha256(
                repo_root / "gaussian_renderer" / "tacker_pipeline.py",
                errors,
                "gaussian_renderer/tacker_pipeline.py SHA-256",
            ),
            "diff_gaussian_rasterization/__init__.py": _safe_file_sha256(
                getattr(_rasterizer_module, "__file__", None),
                errors,
                "diff_gaussian_rasterization/__init__.py SHA-256",
            ),
            "diff_gaussian_rasterization._C": _safe_file_sha256(
                getattr(
                    getattr(_rasterizer_module, "_C", None), "__file__", None
                ),
                errors,
                "diff_gaussian_rasterization/_C SHA-256",
            ),
            "configs": _safe_file_sha256(
                args.configs, errors, "config SHA-256"
            ),
        }
    else:
        role_hashes = {
            record["role"]: record.get("sha256")
            for record in pre_import_snapshot["files"]
        }
        profile_render_sha256 = role_hashes.get("source.profile_render")
        source_files = {
            "profile_render.py": profile_render_sha256,
            "gaussian_renderer/__init__.py": role_hashes.get(
                "source.gaussian_renderer.__init__"
            ),
            "gaussian_renderer/tacker_pipeline.py": role_hashes.get(
                "source.gaussian_renderer.tacker_pipeline"
            ),
            "diff_gaussian_rasterization/__init__.py": role_hashes.get(
                "rasterizer.wrapper"
            ),
            "diff_gaussian_rasterization._C": role_hashes.get(
                "rasterizer.binary"
            ),
            "tacker_4dgs_head/__init__.py": role_hashes.get(
                "head.wrapper"
            ),
            "tacker_4dgs_head._C": role_hashes.get("head.binary"),
            "simple_knn/__init__.py": role_hashes.get(
                "simple_knn.wrapper"
            ),
            "simple_knn._C": role_hashes.get("simple_knn.binary"),
            "configs": role_hashes.get("config.explicit[0]"),
        }
    repository["source_files"] = source_files

    tacker_profile_sha256 = tacker_profile_sha256_snapshot
    qualification_profile_sha256 = qualification_profile_sha256_snapshot
    if pre_import_snapshot is not None:
        strict_role_hashes = {
            record["role"]: record.get("sha256")
            for record in pre_import_snapshot["files"]
        }
        if args.tacker_profile is not None:
            tacker_profile_sha256 = strict_role_hashes.get("profile.tacker")
        if args.qualification_profile is not None:
            qualification_profile_sha256 = strict_role_hashes.get(
                "profile.qualification"
            )
    if (
        pre_import_snapshot is None
        and args.tacker_profile is not None
        and tacker_profile_sha256 is None
    ):
        tacker_profile_sha256 = _safe_file_sha256(
            args.tacker_profile, errors, "tacker profile SHA-256"
        )
    if (
        pre_import_snapshot is None
        and args.qualification_profile is not None
        and qualification_profile_sha256 is None
    ):
        qualification_profile_sha256 = _safe_file_sha256(
            args.qualification_profile, errors, "qualification profile SHA-256"
        )
    active_profile_sha256 = (
        tacker_profile_sha256
        if args.tacker_profile is not None
        else qualification_profile_sha256
    )
    profile_hashes = {
        "active_profile_sha256": active_profile_sha256,
        "tacker_profile_sha256": tacker_profile_sha256,
        "qualification_profile_sha256": qualification_profile_sha256,
        "profile_manifest_sha256": profile_manifest_sha256,
    }
    return {
        "environment": environment,
        "repository": repository,
        "source_files": source_files,
        "profile_hashes": profile_hashes,
        "metadata_collection_errors": errors,
    }


def _load_profile_snapshot(path, label, pre_import=None, internal=None):
    """Load and hash the exact profile bytes used for this process."""

    _validate_snapshot_pair(pre_import, internal)
    try:
        if pre_import is not None and internal is not None:
            role = (
                "profile.qualification"
                if label == "qualification"
                else "profile.tacker"
            )
            record = _record_for_role(pre_import, role)
            requested = os.path.abspath(str(Path(path).expanduser()))
            if requested != record["input_path"]:
                raise OSError(
                    "profile argument does not match its pre-import snapshot"
                )
            if not record["exists"]:
                raise OSError(
                    "profile was absent from the pre-import byte snapshot"
                )
            raw = internal["bytes_by_path"].get(record["path"])
            if raw is None:
                raise OSError("profile snapshot bytes are missing")
        else:
            raw = Path(path).expanduser().read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("cannot load {} profile: {}".format(label, error))
    if not isinstance(value, dict):
        raise ValueError("{} profile must be a JSON object".format(label))
    return value, hashlib.sha256(raw).hexdigest()


def select_views(scene, split):
    if split == "train":
        return scene.getTrainCameras()
    if split == "test":
        return scene.getTestCameras()
    return scene.getVideoCameras()


def _profile_resolution_scale(value):
    """Normalize the profiling-only image scale without changing old runs."""

    if type(value) is not int or value not in (-1, 1, 2, 4, 8):
        raise ValueError(
            "--resolution must be one of -1, 1, 2, 4, or 8 for profiling"
        )
    return 1 if value == -1 else value


def _scaled_profile_resolution(width, height, scale):
    if type(width) is not int or type(height) is not int:
        raise ValueError("profile view dimensions must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("profile view dimensions must be positive")
    if scale not in (1, 2, 4, 8):
        raise ValueError("profile resolution scale must be 1, 2, 4, or 8")
    return (
        max(1, int(round(float(width) / scale))),
        max(1, int(round(float(height) / scale))),
    )


def _scale_profile_view(view, scale):
    """Return a profiling-only lower-resolution view with unchanged geometry."""

    if scale == 1:
        return view
    image = getattr(view, "original_image", None)
    if image is None or len(getattr(image, "shape", ())) != 3:
        raise ValueError(
            "profiling resolution scaling requires a CHW original_image"
        )
    original_height = int(image.shape[1])
    original_width = int(image.shape[2])
    target_width, target_height = _scaled_profile_resolution(
        original_width, original_height, scale
    )
    scaled_image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    scaled_view = copy(view)
    scaled_view.original_image = scaled_image
    scaled_view.image_width = target_width
    scaled_view.image_height = target_height

    # A resolution-only workload must preserve the camera rays and transforms.
    for field in (
        "FoVx",
        "FoVy",
        "world_view_transform",
        "projection_matrix",
        "full_proj_transform",
        "camera_center",
    ):
        if hasattr(view, field) and getattr(scaled_view, field) is not getattr(
            view, field
        ):
            raise RuntimeError(
                "profiling resolution scaling changed camera field {}".format(
                    field
                )
            )
    return scaled_view


def run_views(
    views,
    execution_mode,
    gaussians,
    pipeline,
    background,
    render_kwargs,
    pipeline_renderer=None,
    completion_events=None,
):
    """Submit frames without copying or encoding any rendered output."""
    fallback_reason = None
    if execution_mode in ("two_stream", "tacker"):
        if pipeline_renderer is None:
            if execution_mode == "two_stream":
                pipeline_renderer = TwoStreamRenderer(
                    gaussians,
                    pipeline,
                    background,
                    stage=render_kwargs["stage"],
                    cam_type=render_kwargs["cam_type"],
                )
            else:
                raise ValueError("tacker mode requires a configured TackerRenderer")
        iterator = pipeline_renderer.render_sequence(views)
        try:
            for frame_index in range(len(views)):
                with nvtx_range("profile/frame_{:04d}".format(frame_index)):
                    next(iterator)
                    if completion_events is not None:
                        completion = torch.cuda.Event(enable_timing=True)
                        completion.record(torch.cuda.current_stream())
                        completion_events.append(completion)
            # Resume once past the final yield so the renderer's non-reentrancy
            # guard is released before this instance is reused after warmup.
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("render_sequence yielded more frames than requested")
        finally:
            iterator.close()
        fallback_reason = pipeline_renderer.last_fallback_reason
        return fallback_reason

    for frame_index, view in enumerate(views):
        with nvtx_range("profile/frame_{:04d}".format(frame_index)):
            if execution_mode == "serial":
                render(view, gaussians, pipeline, background, **render_kwargs)
            elif execution_mode == "split_serial":
                context = prepare_render_context(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    cam_type=render_kwargs["cam_type"],
                )
                state = deform_for_render(
                    context,
                    gaussians,
                    stage=render_kwargs["stage"],
                )
                rasterize_state(context, state)
            else:
                raise ValueError("unknown execution mode: {}".format(execution_mode))
            if completion_events is not None:
                completion = torch.cuda.Event(enable_timing=True)
                completion.record(torch.cuda.current_stream())
                completion_events.append(completion)
    return fallback_reason


def _measure_trial(
    trial_index,
    views,
    execution_mode,
    gaussians,
    pipeline,
    background,
    render_kwargs,
    pipeline_renderer,
):
    """Measure one complete sequence with independent wall/CUDA clocks."""

    # Reset after the preceding warmup/trial synchronization and before either
    # timing clock starts.  Peak queries happen only after the wall boundary,
    # so this diagnostic cannot alter the primary throughput metric.
    torch.cuda.reset_peak_memory_stats()
    completion_start = torch.cuda.Event(enable_timing=True)
    completion_end = torch.cuda.Event(enable_timing=True)
    completion_events = []
    completion_start.record(torch.cuda.current_stream())
    start_time = perf_counter()
    with nvtx_range("profile/trial_{:04d}".format(trial_index)):
        with nvtx_range("profile/render_loop"):
            fallback_reason = run_views(
                views,
                execution_mode,
                gaussians,
                pipeline,
                background,
                render_kwargs,
                pipeline_renderer=pipeline_renderer,
                completion_events=completion_events,
            )
    completion_end.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_seconds = perf_counter() - start_time
    cuda_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    cuda_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

    if len(completion_events) != len(views):
        raise RuntimeError("frame completion event count does not match --frames")
    cumulative_ms = [
        float(completion_start.elapsed_time(event)) for event in completion_events
    ]
    frame_completion_ms = []
    previous_ms = 0.0
    for value in cumulative_ms:
        frame_completion_ms.append(value - previous_ms)
        previous_ms = value

    frame_count = len(views)
    total_render_ms = elapsed_seconds * 1000.0
    return {
        "trial_index": trial_index,
        "elapsed_seconds": elapsed_seconds,
        "total_render_ms": total_render_ms,
        "throughput_fps": frame_count / elapsed_seconds,
        "mean_frame_ms": total_render_ms / frame_count,
        "cuda_event_total_render_ms": float(
            completion_start.elapsed_time(completion_end)
        ),
        "cuda_event_mean_frame_ms": sum(frame_completion_ms) / frame_count,
        "p50_frame_ms": statistics.median(frame_completion_ms),
        "p95_frame_ms": _percentile(frame_completion_ms, 95.0),
        "max_frame_ms": max(frame_completion_ms),
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
        "frame_completion_ms": frame_completion_ms,
        "cumulative_frame_completion_ms": cumulative_ms,
        "fallback_reason": fallback_reason,
    }


def _aggregate_trials(trials):
    metric_names = (
        "elapsed_seconds",
        "total_render_ms",
        "throughput_fps",
        "mean_frame_ms",
        "cuda_event_total_render_ms",
        "cuda_event_mean_frame_ms",
        "p50_frame_ms",
        "p95_frame_ms",
        "max_frame_ms",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    )
    aggregates = {
        "median_{}".format(name): statistics.median(
            trial[name] for trial in trials
        )
        for name in metric_names
    }
    aggregates.update(
        {
            "max_cuda_peak_allocated_bytes_across_trials": max(
                trial["cuda_peak_allocated_bytes"] for trial in trials
            ),
            "max_cuda_peak_reserved_bytes_across_trials": max(
                trial["cuda_peak_reserved_bytes"] for trial in trials
            ),
        }
    )
    return aggregates


def _execution_state(execution_mode, pipeline_renderer, fallback_reason):
    """Snapshot the physical backend used by one completed sequence."""

    state = {
        "actual_execution_mode": execution_mode,
        "two_stream_fallback_reason": None,
        "tacker_fallback_reason": None,
        "qualification_mode_executed": False,
    }
    if execution_mode == "two_stream":
        state["two_stream_fallback_reason"] = fallback_reason
        if fallback_reason is not None:
            state["actual_execution_mode"] = "serial"
    elif execution_mode == "tacker":
        state.update(
            {
                "actual_execution_mode": pipeline_renderer.actual_execution_mode,
                "two_stream_fallback_reason": (
                    pipeline_renderer.fallback_backend_reason
                ),
                "tacker_fallback_reason": pipeline_renderer.last_fallback_reason,
                "qualification_mode_executed": bool(
                    pipeline_renderer.last_qualification_mode
                ),
            }
        )
    return state


def main(
    args,
    dataset,
    hyperparam,
    pipeline,
    pre_import_snapshot=None,
    pre_import_internal=None,
):
    _validate_snapshot_pair(pre_import_snapshot, pre_import_internal)
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    profile_resolution_value = getattr(args, "resolution", -1)
    if profile_resolution_value is None:
        profile_resolution_value = -1
    profile_resolution_argument = profile_resolution_value
    profile_resolution_scale = _profile_resolution_scale(
        profile_resolution_argument
    )
    if args.qualification_mode and args.execution_mode != "tacker":
        raise ValueError("--qualification-mode requires --execution-mode tacker")
    if args.qualification_mode and args.qualification_profile is None:
        raise ValueError("--qualification-mode requires --qualification-profile")
    if not args.qualification_mode and args.qualification_profile is not None:
        raise ValueError(
            "--qualification-profile is accepted only with --qualification-mode"
        )
    if args.tacker_profile is not None and args.qualification_profile is not None:
        raise ValueError(
            "--tacker-profile and --qualification-profile are mutually exclusive"
        )
    if args.execution_mode == "tacker":
        if args.workload_name is None:
            raise ValueError("tacker mode requires --workload-name")
        if args.tacker_profile is None and not args.qualification_mode:
            raise ValueError(
                "tacker mode requires --tacker-profile or explicit qualification mode"
            )
    elif args.tacker_profile is not None:
        raise ValueError("--tacker-profile requires --execution-mode tacker")

    qualification_override = None
    qualification_profile_sha256_snapshot = None
    qualification_profile_metadata_path = None
    tacker_override = None
    tacker_profile_snapshot_error = None
    tacker_profile_sha256_snapshot = None
    tacker_profile_metadata_path = None
    if args.qualification_mode:
        if pre_import_snapshot is not None:
            qualification_record = _record_for_role(
                pre_import_snapshot, "profile.qualification"
            )
            qualification_profile_metadata_path = qualification_record["path"]
        else:
            qualification_profile_metadata_path = str(
                Path(args.qualification_profile).expanduser().resolve()
            )
        (
            qualification_override,
            qualification_profile_sha256_snapshot,
        ) = _load_profile_snapshot(
            args.qualification_profile,
            "qualification",
            pre_import=pre_import_snapshot,
            internal=pre_import_internal,
        )
    elif args.execution_mode == "tacker":
        tacker_record = None
        if pre_import_snapshot is not None:
            tacker_record = _record_for_role(
                pre_import_snapshot, "profile.tacker"
            )
            tacker_profile_metadata_path = tacker_record["path"]
        else:
            tacker_profile_metadata_path = str(
                Path(args.tacker_profile).expanduser().resolve()
            )
        if tacker_record is not None and not tacker_record["exists"]:
            # Pass the sealed absence itself into the renderer.  Re-opening
            # the pathname here would allow an atomic create/swap to change
            # which profile controls dispatch after the pre-import seal.
            tacker_profile_snapshot_error = (
                "cannot load Tacker profile {}: profile was absent from the "
                "sealed pre-import snapshot".format(tacker_record["path"])
            )
        else:
            (
                tacker_override,
                tacker_profile_sha256_snapshot,
            ) = _load_profile_snapshot(
                args.tacker_profile,
                "Tacker",
                pre_import=pre_import_snapshot,
                internal=pre_import_internal,
            )

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        gaussians._deformation.eval()
        views = select_views(scene, args.split)
        if len(views) == 0:
            raise RuntimeError("the selected split has no cameras")

        view_indices = [index % len(views) for index in range(args.frames)]
        selected = [views[index] for index in view_indices]
        warmup_views = [views[index % len(views)] for index in range(args.warmup)]
        original_image_width = int(selected[0].image_width)
        original_image_height = int(selected[0].image_height)
        if profile_resolution_scale != 1:
            unscaled_selected = selected
            unscaled_warmup_views = warmup_views
            selected = [
                _scale_profile_view(view, profile_resolution_scale)
                for view in unscaled_selected
            ]
            warmup_views = [
                _scale_profile_view(view, profile_resolution_scale)
                for view in unscaled_warmup_views
            ]
            # Do not retain full-resolution image tensors through profiling;
            # otherwise the lower-resolution memory diagnostic is misleading.
            del unscaled_selected
            del unscaled_warmup_views

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_kwargs = {"stage": "fine", "cam_type": scene.dataset_type}
        pipeline_renderer = None
        if args.execution_mode == "two_stream":
            # Stream/event construction is setup, not per-frame rendering, and
            # therefore remains outside the timed region.
            pipeline_renderer = TwoStreamRenderer(
                gaussians,
                pipeline,
                background,
                stage=render_kwargs["stage"],
                cam_type=render_kwargs["cam_type"],
            )
        elif args.execution_mode == "tacker":
            pipeline_renderer = TackerRenderer(
                gaussians,
                pipeline,
                background,
                stage=render_kwargs["stage"],
                cam_type=render_kwargs["cam_type"],
                profile_path=None,
                profile_override=(
                    qualification_override
                    if args.qualification_mode
                    else tacker_override
                ),
                profile_snapshot_error=(
                    None
                    if args.qualification_mode
                    else tacker_profile_snapshot_error
                ),
                workload_name=args.workload_name,
                iteration=scene.loaded_iter,
                qualification_mode=args.qualification_mode,
            )

        # Bind every timed result to the source, config, loaded rasterizer binary,
        # repository, and environment observed before warmup.  Collecting these
        # hashes after all trials would let a concurrent edit relabel measurements
        # with bytes that were never used by this process.
        pre_measurement_manifest_sha256 = None
        if (
            args.execution_mode == "tacker"
            and pipeline_renderer.profile is not None
        ):
            pre_measurement_manifest_sha256 = pipeline_renderer.profile.get(
                "manifest_sha256"
            )
        reproducibility = _collect_reproducibility_metadata(
            args,
            pre_measurement_manifest_sha256,
            tacker_profile_sha256_snapshot=tacker_profile_sha256_snapshot,
            qualification_profile_sha256_snapshot=(
                qualification_profile_sha256_snapshot
            ),
            pre_import_snapshot=pre_import_snapshot,
        )

        # Stream/event creation and immutable parameter conversion are setup,
        # even when --warmup=0.  Keep both outside CUDA-event/perf timing.
        if pipeline_renderer is not None:
            prepare = getattr(pipeline_renderer, "prepare", None)
            if callable(prepare):
                prepare()

        run_views(
            warmup_views,
            args.execution_mode,
            gaussians,
            pipeline,
            background,
            render_kwargs,
            pipeline_renderer=pipeline_renderer,
        )
        torch.cuda.synchronize()

        trials = []
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for trial_index in range(1, args.trials + 1):
                trial = _measure_trial(
                    trial_index,
                    selected,
                    args.execution_mode,
                    gaussians,
                    pipeline,
                    background,
                    render_kwargs,
                    pipeline_renderer,
                )
                trial.update(
                    _execution_state(
                        args.execution_mode,
                        pipeline_renderer,
                        trial["fallback_reason"],
                    )
                )
                trials.append(trial)
        finally:
            torch.cuda.cudart().cudaProfilerStop()

        aggregates = _aggregate_trials(trials)
        representative_trial = min(
            trials,
            key=lambda trial: (
                abs(
                    trial["throughput_fps"]
                    - aggregates["median_throughput_fps"]
                ),
                trial["trial_index"],
            ),
        )
        execution_fields = (
            "actual_execution_mode",
            "two_stream_fallback_reason",
            "tacker_fallback_reason",
            "qualification_mode_executed",
        )
        execution_signatures = {
            tuple(trial[field] for field in execution_fields) for trial in trials
        }
        if len(execution_signatures) != 1:
            raise RuntimeError(
                "physical execution mode or fallback changed between trials"
            )
        execution_state = {field: trials[0][field] for field in execution_fields}
        actual_execution_mode = execution_state["actual_execution_mode"]
        pipeline_execution_counts = (
            pipeline_renderer.last_execution_counts
            if actual_execution_mode == "tacker"
            else None
        )
        two_stream_fallback_reason = execution_state[
            "two_stream_fallback_reason"
        ]
        tacker_fallback_reason = execution_state["tacker_fallback_reason"]
        qualification_executed = execution_state[
            "qualification_mode_executed"
        ]
        persistent_blocks = None
        profile_manifest_sha256 = None
        profile_selection_sha256 = None
        selected_variant_id = None
        selected_candidate_abi_sha256 = None
        if args.execution_mode == "tacker":
            persistent_blocks = pipeline_renderer.persistent_blocks
            if pipeline_renderer.profile is not None:
                active_profile = pipeline_renderer.profile
                profile_manifest_sha256 = active_profile.get("manifest_sha256")
                profile_selection_sha256 = active_profile.get("profile_sha256")
                if active_profile.get("schema_version") == 1:
                    selected_variant_id = "legacy_pos_l1"
                else:
                    selected_variant_id = active_profile.get("selected_variant_id")
                    for candidate in active_profile.get("candidates", []):
                        if candidate.get("variant_id") == selected_variant_id:
                            selected_candidate_abi_sha256 = candidate.get(
                                "abi_manifest_sha256"
                            )
                            break

        environment = reproducibility["environment"]
        repository = reproducibility["repository"]
        source_files = reproducibility["source_files"]
        profile_hashes = reproducibility["profile_hashes"]
        profile_hashes.update(
            {
                "profile_selection_sha256": profile_selection_sha256,
                "selected_candidate_abi_sha256": selected_candidate_abi_sha256,
            }
        )
        gpu_environment = environment.get("gpu") or {}
        nvidia_smi_environment = environment.get("nvidia_smi") or {}

        first_view = selected[0]
        metadata = {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_render_profile",
            "passed": True,
            "model_path": str(Path(dataset.model_path).resolve()),
            "source_path": str(Path(dataset.source_path).resolve()),
            "iteration": scene.loaded_iter,
            "split": args.split,
            "warmup_frames": args.warmup,
            "profile_frames": args.frames,
            "view_indices": view_indices,
            "execution_mode": args.execution_mode,
            "actual_execution_mode": actual_execution_mode,
            "pipeline_execution_counts": pipeline_execution_counts,
            "two_stream_fallback_reason": two_stream_fallback_reason,
            "tacker_fallback_reason": tacker_fallback_reason,
            "qualification_mode_requested": bool(args.qualification_mode),
            "qualification_mode_executed": bool(qualification_executed),
            "workload_name": args.workload_name,
            "tacker_profile": (
                tacker_profile_metadata_path
            ),
            "qualification_profile": (
                qualification_profile_metadata_path
            ),
            "profile_manifest_sha256": profile_manifest_sha256,
            "profile_selection_sha256": profile_selection_sha256,
            "selected_variant_id": selected_variant_id,
            "selected_candidate_abi_sha256": selected_candidate_abi_sha256,
            "persistent_blocks": persistent_blocks,
            "trial_count": args.trials,
            "trials": trials,
            "aggregate_method": "median_across_whole_sequence_trials",
            "primary_metric": "median_throughput_fps",
            "primary_metric_higher_is_better": True,
            "representative_trial_index": representative_trial["trial_index"],
            # Legacy top-level timing keys and their raw samples all describe
            # the same representative trial. Cross-trial summaries live only
            # under the explicit median_* names below.
            "elapsed_seconds": representative_trial["elapsed_seconds"],
            "total_render_ms": representative_trial["total_render_ms"],
            "mean_frame_ms": representative_trial["mean_frame_ms"],
            "cuda_event_total_render_ms": representative_trial[
                "cuda_event_total_render_ms"
            ],
            "cuda_event_mean_frame_ms": representative_trial[
                "cuda_event_mean_frame_ms"
            ],
            "p50_frame_ms": representative_trial["p50_frame_ms"],
            "p95_frame_ms": representative_trial["p95_frame_ms"],
            "max_frame_ms": representative_trial["max_frame_ms"],
            "cuda_peak_allocated_bytes": representative_trial[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_peak_reserved_bytes": representative_trial[
                "cuda_peak_reserved_bytes"
            ],
            "frame_completion_ms": representative_trial["frame_completion_ms"],
            "cumulative_frame_completion_ms": representative_trial[
                "cumulative_frame_completion_ms"
            ],
            "throughput_fps": representative_trial["throughput_fps"],
            "timing_method": "perf_counter_with_cuda_synchronize",
            "frame_timing_method": "cuda_event_consumer_completion_intervals",
            "io_in_timed_region": False,
            "timing_contract": {
                "unit": "whole_sequence",
                "frames_per_trial": args.frames,
                "trial_count": args.trials,
                "primary_metric": "median_throughput_fps",
                "higher_is_better": True,
                "wall_clock": "perf_counter",
                "wall_clock_completion": "cuda_synchronize_after_each_trial",
                "cuda_events": "start_end_and_per_frame_completion",
                "setup_policy": "single_load_prepare_warmup_before_all_trials",
                "io_in_timed_region": False,
                "cuda_peak_memory": (
                    "reset_before_each_trial_query_after_wall_boundary"
                ),
            },
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
            "image_width": int(first_view.image_width),
            "image_height": int(first_view.image_height),
            "profile_resolution_argument": profile_resolution_argument,
            "profile_resolution_scale": profile_resolution_scale,
            "original_image_width": original_image_width,
            "original_image_height": original_image_height,
            "original_resolution": [
                original_image_width,
                original_image_height,
            ],
            "effective_resolution": [
                int(first_view.image_width),
                int(first_view.image_height),
            ],
            "profile_resolution_contract": (
                "bilinear_original_image_only_fov_and_projection_unchanged"
            ),
            "dataset_type": scene.dataset_type,
            "convert_SHs_python": pipeline.convert_SHs_python,
            "compute_cov3D_python": pipeline.compute_cov3D_python,
            "gpu_name": gpu_environment.get("name")
            or nvidia_smi_environment.get("name"),
            "cuda_runtime": environment.get("cuda_runtime"),
            "cuda_driver_version": environment.get("cuda_driver_version"),
            "pytorch_version": environment.get("pytorch_version"),
            "source_tree": repository.get("source_tree"),
            "repository_commit": repository.get("commit"),
            "repository_commit_source": repository.get("commit_source"),
            "repository_dirty": repository.get("dirty"),
            "repository_git_error": repository.get("git_error"),
            "profile_render_sha256": source_files.get("profile_render.py"),
            "source_files": source_files,
            "submodule_commits": {
                item["path"]: item["commit"]
                for item in repository.get("submodules", [])
            },
            "active_profile_sha256": profile_hashes.get(
                "active_profile_sha256"
            ),
            "tacker_profile_sha256": profile_hashes.get(
                "tacker_profile_sha256"
            ),
            "qualification_profile_sha256": profile_hashes.get(
                "qualification_profile_sha256"
            ),
            "environment": environment,
            "repository": repository,
            "profile_hashes": profile_hashes,
            "metadata_collection_errors": reproducibility[
                "metadata_collection_errors"
            ],
            "pre_import": pre_import_snapshot,
            "loaded_binaries": (
                pre_import_snapshot.get("binary_bindings", [])
                if pre_import_snapshot is not None
                else []
            ),
            "pipeline_slot_count": (
                2
                if actual_execution_mode in ("two_stream", "tacker")
                else 1
            ),
        }
        metadata.update(aggregates)
        if pre_import_snapshot is None:
            post_run_byte_stability = {
                "verified": False,
                "verification_point": None,
                "file_count": 0,
                "files": [],
                "reason": "direct caller did not provide a pre-import snapshot",
            }
        else:
            post_run_byte_stability = _verify_pre_import_snapshot(
                pre_import_snapshot, pre_import_internal
            )
        metadata["post_run"] = {
            "byte_stability": post_run_byte_stability,
        }
        if args.metadata:
            _atomic_write_json(args.metadata, metadata)
        print(
            "Profiled {profile_frames} {split} frames in {actual_execution_mode} mode "
            "at iteration {iteration}: {median_throughput_fps:.2f} median FPS over "
            "{trial_count} trial(s), "
            "{median_mean_frame_ms:.3f} median ms/frame, {gaussian_count} Gaussians "
            "({image_width}x{image_height})".format(**metadata)
        )
        if tacker_fallback_reason is not None:
            print("tacker fell back: {}".format(tacker_fallback_reason))
        if two_stream_fallback_reason is not None:
            print(
                "two_stream backend fell back to serial: {}".format(
                    two_stream_fallback_reason
                )
            )


def _preparse_snapshot_args(argv):
    """Locate every CLI-controlled file without importing project modules."""

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--tacker-profile", type=str)
    parser.add_argument("--qualification-profile", type=str)
    parser.add_argument("--model_path", "-m", type=str)
    parsed, _unknown = parser.parse_known_args(argv)
    return parsed


def _build_parser():
    parser = ArgumentParser(description="Nsight render-only profiling")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--split", choices=("train", "test", "video"), default="test")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--frames", default=50, type=int)
    parser.add_argument("--trials", default=1, type=int)
    parser.add_argument(
        "--execution-mode",
        "--execution_mode",
        dest="execution_mode",
        choices=("serial", "split_serial", "two_stream", "tacker"),
        default="serial",
    )
    parser.add_argument("--tacker-profile", type=str)
    parser.add_argument("--qualification-mode", action="store_true")
    parser.add_argument("--qualification-profile", type=str)
    parser.add_argument("--workload-name", type=str)
    parser.add_argument("--metadata", type=str)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, hyperparam, pipeline


def _cli_main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    bootstrap_args = _preparse_snapshot_args(argv)
    pre_import_snapshot, pre_import_internal = (
        _capture_pre_import_snapshot(bootstrap_args)
    )
    _activate_runtime_imports(pre_import_snapshot, pre_import_internal)
    parser, model, hyperparam, pipeline = _build_parser()

    parsed = _combined_args_from_snapshot(
        parser, argv, pre_import_snapshot, pre_import_internal
    )
    if parsed.configs:
        parsed = _merge_hparams(
            parsed,
            _load_config_snapshot(
                parsed.configs, pre_import_snapshot, pre_import_internal
            ),
        )

    safe_state(parsed.quiet)
    main(
        parsed,
        model.extract(parsed),
        hyperparam.extract(parsed),
        pipeline.extract(parsed),
        pre_import_snapshot=pre_import_snapshot,
        pre_import_internal=pre_import_internal,
    )


if __name__ == "__main__":
    _cli_main()

from pathlib import Path


def _merge_dicts(base, override):
    """Recursively merge config dictionaries without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    """Load the small Python config format used by this repository.

    The original entry points rely on ``mmcv.Config`` only for ``_base_``
    inheritance. Keeping that behavior here removes a large, otherwise unused
    runtime dependency on newer PyTorch/CUDA installations.
    """
    path = Path(path).expanduser().resolve()
    namespace = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {}, namespace)

    config = {}
    bases = namespace.pop("_base_", None)
    if bases:
        if isinstance(bases, (str, Path)):
            bases = [bases]
        for base in bases:
            config = _merge_dicts(config, load_config(path.parent / base))

    local = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
    }
    return _merge_dicts(config, local)


def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)

    return args

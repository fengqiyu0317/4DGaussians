"""Execute ``profile_render.py`` from one stable, sealed file image.

This launcher intentionally imports only the Python standard library.  It is
the provenance boundary for profiling: the bytes compiled below are the same
bytes injected into ``profile_render`` for its pre-import and post-run seal.
"""

import hashlib
import os
import stat
import sys


def _stat_identity(value):
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "mtime_ns": int(
            getattr(value, "st_mtime_ns", int(value.st_mtime * 1000000000))
        ),
        "ctime_ns": int(
            getattr(value, "st_ctime_ns", int(value.st_ctime * 1000000000))
        ),
    }


def _identity(value):
    result = _stat_identity(value)
    result["size"] = int(value.st_size)
    return result


def _read_stable_source(path):
    target = os.path.abspath(os.path.expanduser(path))
    if os.path.realpath(target) != target:
        raise RuntimeError(
            "profile_render source path contains a symbolic link: {}".format(
                target
            )
        )
    before_path = os.lstat(target)
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(
        before_path.st_mode
    ):
        raise RuntimeError(
            "profile_render source is not a regular non-symlink file: {}"
            .format(target)
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    try:
        before_fd = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(target)
    finally:
        os.close(descriptor)
    identities = {
        tuple(sorted(_identity(value).items()))
        for value in (before_path, before_fd, after_fd, after_path)
    }
    raw = b"".join(chunks)
    if (
        len(identities) != 1
        or len(raw) != int(after_fd.st_size)
        or stat.S_ISLNK(after_path.st_mode)
        or os.path.realpath(target) != target
    ):
        raise RuntimeError(
            "profile_render source changed during bootstrap snapshot: {}"
            .format(target)
        )
    digest = hashlib.sha256(raw).hexdigest()
    return (
        {
            "role": "source.profile_render",
            "input_path": target,
            "path": target,
            "required": True,
            "exists": True,
            "stat": _stat_identity(after_fd),
            "size_bytes": len(raw),
            "sha256": digest,
        },
        raw,
    )


def main():
    source_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "profile_render.py",
    )
    record, raw = _read_stable_source(source_path)
    code = compile(raw, source_path, "exec", dont_inherit=True)
    namespace = {
        "__name__": "__main__",
        "__file__": source_path,
        "__package__": None,
        "__cached__": None,
        "__spec__": None,
        "__builtins__": __builtins__,
        "_PROFILE_RENDER_BOOTSTRAP": {
            "protocol": 1,
            "record": record,
            "bytes": raw,
            "execution": {
                "compiled_sha256": record["sha256"],
                "compile_mode": "exec",
                "dont_inherit": True,
            },
        },
    }
    exec(code, namespace, namespace)


if __name__ == "__main__":
    main()

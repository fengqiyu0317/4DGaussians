"""Default-off Green streams for Deform-prefix/Raster-prepare experiments.

No CUDA library or PyTorch import occurs just by importing this module.
PyTorch's allocator retains external stream handles through record_stream(),
including for model tensors that outlive a renderer. Published pairs therefore
live in a bounded per-configuration process pool and are reclaimed by CUDA at
process exit, not destroyed when a renderer's Graphs are released.
"""
import ctypes
import hashlib
import os
from pathlib import Path


_PROCESS_PAIRS = {}


def handoff_enabled():
    value = os.environ.get("TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF", "0")
    if value not in ("0", "1"):
        raise ValueError("TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF must be 0 or 1")
    if value == "1" and (os.environ.get("TACKER_EXPERIMENTAL_GREEN_CONTEXTS", "0") != "1"
                         or os.environ.get("TACKER_GREEN_DEFORM_SMS", "4") != "8"):
        raise ValueError("Green prepare handoff requires the 8+8 Green experiment")
    return value == "1"


def get_green_context_pair(device, deform_sms, library):
    key = (str(device), deform_sms, library)
    if key not in _PROCESS_PAIRS:
        pair = GreenContextPair(device, deform_sms, library)
        pair._published = True
        _PROCESS_PAIRS[key] = pair
    return _PROCESS_PAIRS[key]


def experiment_config():
    value = os.environ.get("TACKER_EXPERIMENTAL_GREEN_CONTEXTS", "0")
    if value not in ("0", "1"):
        raise ValueError("TACKER_EXPERIMENTAL_GREEN_CONTEXTS must be 0 or 1")
    if value == "0":
        return None
    sms = os.environ.get("TACKER_GREEN_DEFORM_SMS", "4")
    if sms not in ("4", "8", "12"):
        raise ValueError("TACKER_GREEN_DEFORM_SMS must be 4, 8, or 12")
    library = Path(os.environ.get("TACKER_GREEN_CONTEXTS_LIBRARY", ""))
    if not library.is_absolute() or not library.is_file():
        raise ValueError("TACKER_GREEN_CONTEXTS_LIBRARY must name an existing absolute library path")
    if library.is_symlink() or library.resolve() != library:
        raise ValueError("Green Contexts library path must not contain symlinks")
    return {"deform_sms": int(sms), "library": str(library)}


def _load_native(path):
    native = ctypes.CDLL(path)
    native.tacker_green_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    native.tacker_green_create.restype = ctypes.c_int
    native.tacker_green_destroy.argtypes = [ctypes.c_void_p]
    native.tacker_green_destroy.restype = ctypes.c_int
    native.tacker_green_stream.argtypes = [ctypes.c_void_p, ctypes.c_int]
    native.tacker_green_stream.restype = ctypes.c_size_t
    native.tacker_green_stream_sms.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
    native.tacker_green_stream_sms.restype = ctypes.c_int
    native.tacker_green_error_name.argtypes = [ctypes.c_int]
    native.tacker_green_error_name.restype = ctypes.c_char_p
    native.tacker_green_last_stage.restype = ctypes.c_char_p
    return native


class GreenContextPair:
    def __init__(self, device, deform_sms, library):
        import torch

        self._handle = ctypes.c_void_p()
        self._published = False
        self.deform_stream = self.raster_stream = None
        self._library = Path(library)
        digest = hashlib.sha256(self._library.read_bytes()).hexdigest()
        self._native = _load_native(str(self._library))
        if hashlib.sha256(self._library.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Green Contexts library changed during loading")
        with torch.cuda.device(device):
            index = torch.cuda.current_device()
            self._check(self._native.tacker_green_create(index, deform_sms, ctypes.byref(self._handle)))
            try:
                handles, counts = [], []
                for side in range(2):
                    handle = self._native.tacker_green_stream(self._handle, side)
                    count = ctypes.c_uint()
                    self._check(self._native.tacker_green_stream_sms(self._handle, side, ctypes.byref(count)))
                    if not handle or count.value != (deform_sms if side == 0 else 16 - deform_sms):
                        raise RuntimeError("Green Context stream/resource verification failed")
                    handles.append(handle)
                    counts.append(count.value)
                self.deform_stream = torch.cuda.ExternalStream(handles[0], device=index)
                self.raster_stream = torch.cuda.ExternalStream(handles[1], device=index)
                self.metadata = {"enabled": True, "scope": "deform_prefix_raster_prepare",
                    "deform_sms": counts[0], "raster_sms": counts[1], "tail_sms": 16,
                    "raster_prepare_sms": counts[1], "raster_prepass_sms": 16,
                    "library_path": str(self._library), "library_sha256": digest,
                    "stream_resource_binding_verified": True,
                    "lifetime": "process_pool_for_pytorch_allocator_stream_references",
                    "deform_stream": handles[0], "raster_stream": handles[1]}
            except Exception:
                self.close()
                raise

    def _check(self, code):
        if code:
            stage = self._native.tacker_green_last_stage().decode()
            name = self._native.tacker_green_error_name(code).decode()
            raise RuntimeError("Green Contexts {} failed: {} ({})".format(stage, name, code))

    def close(self):
        if self._published:
            raise RuntimeError("Published Green streams must live until process exit")
        handle = self._handle
        if handle.value is None:
            return
        self._handle = ctypes.c_void_p()
        self.deform_stream = self.raster_stream = None
        self._check(self._native.tacker_green_destroy(handle))

"""Audio decoding for non-WAV sources (optional plugin, always falls back).

Windows 10/11 ships AAC (M4A) + MP3 decoders inside Media Foundation, so the
user-provided ``demo音源集/*.mp3`` (they are mostly M4A containers) can be turned
into PCM WAV without ffmpeg or pip installs.  The decoded result is cached under
``work/decoded/`` and the rest of the app keeps using its zero-dependency WAV
path.

Decoder chain (first success wins):
  1. already PCM WAV                  -> use as is
  2. cache hit                        -> use cache
  3. Windows Media Foundation         -> native AAC/MP3/WMA decoder (ctypes)
  4. ffmpeg on PATH                   -> if the user has it
  5. soundfile / audioread            -> if installed

Any failure raises :class:`DecodeError` with the reasons; the caller must fall
back to a known-good WAV (the demo must never die because of a codec).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
import wave
from ctypes import POINTER, byref, c_int32, c_int64, c_ubyte, c_uint32, c_void_p
from pathlib import Path

import numpy as np

from ..config import ASSETS, ROOT

DECODE_DIR = ROOT / "work" / "decoded"
SOURCE_DIR = ROOT / "demo音源集"
DEFAULT_SR = 44100
HRESULT = c_int32
GUID = c_ubyte * 16
MF_VERSION = 0x00020070
MF_SOURCE_READER_FIRST_AUDIO_STREAM = 0xFFFFFFFD
MF_SOURCE_READER_ALL_STREAMS = 0xFFFFFFFE
MF_SOURCE_READERF_ENDOFSTREAM = 0x00000002


class DecodeError(RuntimeError):
    """Raised when no decoder could read the file; message lists the attempts."""


# ---------------------------------------------------------------- helpers

def _guid(text: str) -> GUID:
    return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


MF_MT_MAJOR_TYPE = _guid("48eba18e-f8c9-4687-bf11-0a74c9f96a8f")
MF_MT_SUBTYPE = _guid("f7e34c9a-42e8-4714-b74b-cb29d72c35e5")
MF_MT_AUDIO_NUM_CHANNELS = _guid("37e48bf5-645e-4c5b-89de-ada9e29b696a")
MF_MT_AUDIO_SAMPLES_PER_SECOND = _guid("5faeeae7-0290-4c31-9e8a-c534f68d9dba")
MF_MT_AUDIO_BITS_PER_SAMPLE = _guid("f2deb57f-40fa-4764-aa33-ed4f2d1ff669")
MFMediaType_Audio = _guid("73647561-0000-0010-8000-00AA00389B71")
MFAudioFormat_PCM = _guid("00000001-0000-0010-8000-00AA00389B71")


def _vt(obj, index: int, restype, *argtypes):
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[index])


def _check(hr: int, what: str) -> None:
    if hr < 0:
        raise OSError(f"{what} HRESULT=0x{hr & 0xFFFFFFFF:08X}")


def _release(obj) -> None:
    try:
        if obj and getattr(obj, "value", None):
            _vt(obj, 2, HRESULT)(obj)
            obj.value = None
    except Exception:  # noqa: BLE001 - release must never raise
        pass


def is_pcm_wav(path: str | Path) -> bool:
    """True for a plain 16-bit PCM WAV (the only format winsound/load_wav need)."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(44)
    except OSError:
        return False
    if len(head) < 24 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return False
    # scan "fmt " chunk for the format tag / bits
    pos = 12
    data = head
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        if cid == b"fmt " and pos + 8 + 16 <= len(data):
            fmt = int.from_bytes(data[pos + 8:pos + 10], "little")
            bits = int.from_bytes(data[pos + 22:pos + 24], "little")
            return fmt == 1 and bits in (8, 16, 24, 32)
        pos += 8 + size + (size & 1)
        if pos > 512:
            break
    return False


def _cache_path(src: Path, sr: int) -> Path:
    st = src.stat()
    key = f"{src.name}|{st.st_size}|{st.st_mtime_ns}|{sr}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return DECODE_DIR / f"{src.stem[:40]}_{digest}_{sr}.wav"


def _write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    tmp = path.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    tmp.replace(path)


# ---------------------------------------------------------------- Media Foundation

def decode_mediafoundation(src: str | Path, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Decode with Windows Media Foundation (AAC/M4A/MP3/WMA/...)."""
    if sys.platform != "win32":
        raise OSError("Media Foundation is Windows-only")
    mfplat = ctypes.WinDLL("mfplat")
    mfreadwrite = ctypes.WinDLL("mfreadwrite")
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoInitialize(None)
    mfplat.MFStartup.argtypes = [c_uint32, c_uint32]
    mfplat.MFStartup.restype = HRESULT
    _check(mfplat.MFStartup(MF_VERSION, 0), "MFStartup")
    reader = c_void_p()
    mt = c_void_p()
    try:
        mfreadwrite.MFCreateSourceReaderFromURL.argtypes = [
            ctypes.c_wchar_p, c_void_p, POINTER(c_void_p)]
        mfreadwrite.MFCreateSourceReaderFromURL.restype = HRESULT
        _check(mfreadwrite.MFCreateSourceReaderFromURL(
            str(src), None, byref(reader)), "MFCreateSourceReaderFromURL")

        sel = _vt(reader, 4, HRESULT, c_uint32, c_int32)
        _check(sel(reader, MF_SOURCE_READER_ALL_STREAMS, 0), "SetStreamSelection(all)")
        _check(sel(reader, MF_SOURCE_READER_FIRST_AUDIO_STREAM, 1), "SetStreamSelection(audio)")

        mfplat.MFCreateMediaType.argtypes = [POINTER(c_void_p)]
        mfplat.MFCreateMediaType.restype = HRESULT
        _check(mfplat.MFCreateMediaType(byref(mt)), "MFCreateMediaType")
        set_guid = _vt(mt, 24, HRESULT, POINTER(GUID), POINTER(GUID))
        set_u32 = _vt(mt, 21, HRESULT, POINTER(GUID), c_uint32)
        _check(set_guid(mt, byref(MF_MT_MAJOR_TYPE), byref(MFMediaType_Audio)), "set major")
        _check(set_guid(mt, byref(MF_MT_SUBTYPE), byref(MFAudioFormat_PCM)), "set subtype")
        _check(set_u32(mt, byref(MF_MT_AUDIO_NUM_CHANNELS), 1), "set channels")
        _check(set_u32(mt, byref(MF_MT_AUDIO_SAMPLES_PER_SECOND), sr), "set rate")
        _check(set_u32(mt, byref(MF_MT_AUDIO_BITS_PER_SAMPLE), 16), "set bits")
        _check(_vt(reader, 7, HRESULT, c_uint32, c_void_p, c_void_p)(
            reader, MF_SOURCE_READER_FIRST_AUDIO_STREAM, None, mt), "SetCurrentMediaType")

        cur = c_void_p()
        _check(_vt(reader, 6, HRESULT, c_uint32, POINTER(c_void_p))(
            reader, MF_SOURCE_READER_FIRST_AUDIO_STREAM, byref(cur)), "GetCurrentMediaType")
        get_u32 = _vt(cur, 7, HRESULT, POINTER(GUID), POINTER(c_uint32))
        ch = c_uint32(1)
        rate = c_uint32(sr)
        get_u32(cur, byref(MF_MT_AUDIO_NUM_CHANNELS), byref(ch))
        get_u32(cur, byref(MF_MT_AUDIO_SAMPLES_PER_SECOND), byref(rate))
        _release(cur)

        chunks: list[np.ndarray] = []
        read = _vt(reader, 9, HRESULT, c_uint32, c_uint32, POINTER(c_uint32),
                   POINTER(c_uint32), POINTER(c_int64), POINTER(c_void_p))
        while True:
            actual = c_uint32(0)
            flags = c_uint32(0)
            ts = c_int64(0)
            sample = c_void_p()
            _check(read(reader, MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0,
                        byref(actual), byref(flags), byref(ts), byref(sample)), "ReadSample")
            if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
                _release(sample)
                break
            if sample.value:
                buf = c_void_p()
                _check(_vt(sample, 41, HRESULT, POINTER(c_void_p))(sample, byref(buf)),
                       "ConvertToContiguousBuffer")
                data = c_void_p()
                maxlen = c_uint32(0)
                curlen = c_uint32(0)
                _check(_vt(buf, 3, HRESULT, POINTER(c_void_p), POINTER(c_uint32),
                           POINTER(c_uint32))(buf, byref(data), byref(maxlen),
                                              byref(curlen)), "Lock")
                if curlen.value:
                    chunks.append(np.frombuffer(
                        ctypes.string_at(data, curlen.value), dtype="<i2").copy())
                _vt(buf, 4, HRESULT)(buf)
                _release(buf)
                _release(sample)
        if not chunks:
            raise OSError("Media Foundation produced no PCM samples")
        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        if ch.value > 1:
            pcm = pcm.reshape(-1, ch.value).mean(axis=1)
        return pcm, int(rate.value or sr)
    finally:
        _release(mt)
        _release(reader)
        try:
            mfplat.MFShutdown()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- other backends

def decode_ffmpeg(src: Path, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise OSError("ffmpeg not found on PATH")
    cmd = [exe, "-v", "error", "-i", str(src), "-f", "s16le", "-acodec", "pcm_s16le",
           "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=180)
    if proc.returncode != 0 or not proc.stdout:
        raise OSError((proc.stderr or b"").decode("utf-8", "replace")[-300:])
    pcm = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return pcm, sr


def decode_soundfile(src: Path, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    import soundfile as sf  # type: ignore

    data, file_sr = sf.read(str(src), always_2d=True, dtype="float32")
    pcm = data.mean(axis=1)
    if file_sr != sr:
        # cheap linear resample (only used if soundfile is installed)
        n = int(round(len(pcm) * sr / file_sr))
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n),
                        np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm, sr


BACKENDS = (
    ("mediafoundation", decode_mediafoundation),
    ("ffmpeg", decode_ffmpeg),
    ("soundfile", decode_soundfile),
)


# ---------------------------------------------------------------- public API

def ensure_wav(path: str | Path, sr: int = DEFAULT_SR, force: bool = False) -> tuple[Path, str]:
    """Return a playable/analysable PCM WAV for *path* plus a human note.

    Raises :class:`DecodeError` if every backend failed.
    """
    src = Path(path)
    if not src.exists():
        raise DecodeError(f"音源不存在：{src}")
    if is_pcm_wav(src):
        return src, "WAV 原样"
    cached = _cache_path(src, sr)
    if cached.exists() and not force:
        return cached, "缓存"
    reasons: list[str] = []
    for name, backend in BACKENDS:
        try:
            pcm, got_sr = backend(src, sr)
            if pcm.size < 100:
                raise OSError("too few samples")
            _write_wav(cached, pcm, got_sr)
            index = DECODE_DIR / "index.json"
            try:
                data = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
            except Exception:  # noqa: BLE001
                data = {}
            data[src.name] = {"cache": str(cached), "sr": got_sr,
                              "sec": round(len(pcm) / got_sr, 2), "backend": name}
            index.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return cached, f"{name} 解码"
        except Exception as exc:  # noqa: BLE001 - try the next backend
            reasons.append(f"{name}: {type(exc).__name__}: {exc}")
    raise DecodeError("无法解码该音源（" + "；".join(reasons) + "）")


def list_sources() -> list[dict]:
    """All user/demo sources with cache status (for `--list-sources`)."""
    out: list[dict] = []
    if SOURCE_DIR.exists():
        for p in sorted(SOURCE_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"):
                out.append({"path": p, "kind": "demo音源集",
                            "cached": is_pcm_wav(p) or _cache_path(p, DEFAULT_SR).exists()})
    for p in sorted(ASSETS.glob("*.wav")):
        out.append({"path": p, "kind": "assets", "cached": True})
    return out

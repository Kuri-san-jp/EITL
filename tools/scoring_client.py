#!/usr/bin/env python3
"""Client for the shared scoring server (`tools/scoring_server.py`).

Has **the same interface** as the existing in-process scorers:

    pq = AudioboxPQClient()          # drop-in for AudioboxPQScorer
    sb = SongBenchClient()           # drop-in for SongBenchMixingEar
    pq.score(audio, sr) -> float
    sb.score(audio, sr) -> float

audio is (C, N) float32 / sr=44100, exactly as it is today.

============================================================================
Absolute constraints on this file
============================================================================
1. **No imports from the project whatsoever.** stdlib + numpy only.
   Having zero dependencies structurally avoids the import-order problem in
   the project setup notes (outside the container, importing `_common` breaks the scipy in
   python_packages; inside the container, cffi collides unless `_common` is
   imported first). Neither torch nor audiobox nor muq is **loaded**. The CPU
   client never creates a CUDA context.
2. **Audio is sent as raw float32 bytes, untouched.** It does not go through
   encode / quantization / resample. A raw float32 byte round trip is the
   identity map mathematically, so the ndarray that reaches the server is
   bit-for-bit identical to the one the caller passed.
3. **If it is not configured, it is not used.** With no endpoint env set,
   `enabled()` returns False and the caller builds the in-process scorer as
   before.

============================================================================
Environment variables
============================================================================
AUTOMIX_SCORER_ADDR        Endpoints. Comma-separated "host:port".
                           e.g. "127.0.0.1:9631,127.0.0.1:9632,127.0.0.1:9633,127.0.0.1:9634"
                           **If unset (or empty), remote scoring is not used.**
AUTOMIX_SCORE_SERVERS      Alias of AUTOMIX_SCORER_ADDR (either is fine; if both
                           are set, AUTOMIX_SCORER_ADDR wins).
AUTOMIX_CLIENT_INDEX       Serial number of this process (0..N-1). The primary
                           server is chosen as index mod len(addrs). 0 if unset.
                           22 clients / 4 servers -> a fixed 6,6,5,5 split.
AUTOMIX_SCORER_TIMEOUT_SEC socket timeout in seconds for one request. Default 300.
                           Make it long enough to absorb full-song scoring
                           (measured: pq 3.5s / sb 0.53s) plus time spent in the
                           queue. Cutting it short induces duplicate computation.
AUTOMIX_SCORER_RETRY_BUDGET_SEC
                           Total retry budget in seconds for transport failures.
                           Default 600. Once exhausted, ScoringServerUnavailable
                           is raised.
AUTOMIX_SCORER_FP_RELAX    Comma-separated keys to drop from fingerprint matching.
                           **Default is empty = every key matched strictly.**
                           Using it prints a warning.
AUTOMIX_SCORER_ALLOW_CAST  1 to cast float64 input down to float32 before sending.
                           Default 0 = raise. Reasoning in the `_as_wire_f32`
                           docstring.

============================================================================
Retry semantics (at-least-once)
============================================================================
Resending after the connection drops between scoring and the reply still
returns the same value, because the server's `score()` is a **stateless pure
function**. This is relied on completely. Adding a result cache, adaptive
batching, accumulated statistics or randomness on the server side would
silently break that safety.

- transport failure (ConnectionRefused / ConnectionReset / EPIPE / timeout /
  disconnect before HELLO) -> throw the connection away, reconnect and **start
  over from HELLO** (always re-running fingerprint verification). backoff
  0.5,1,2,4,8,16,30,30,... capped at 30 s.
- 2 consecutive failures against the same server -> fail over to the next one.
- If the server returns an ERROR frame -> **a deterministic failure, so do not
  retry.** Raise ScoringServerError immediately.
- **There is no fallback to local scoring.** It is deliberately not implemented,
  so that no mechanism exists for "taking the whole run down to save one song".
  With a fallback, a search process that failed to reach a server would silently
  drop to in-process scoring and **keep running while holding a GPU**. The
  premise of the shared-server scheme (search processes do not use the GPU)
  would quietly collapse, and since bit-exactness is preserved,
  **the results would show no sign of it**.
  It is prevented in the only reliable way: if the code does not exist, it
  cannot happen.
  (The old design launched search processes with CUDA_VISIBLE_DEVICES="" so it
  was "structurally impossible because no GPU is visible", but since execution
  is being unified on the cluster job wrapper, that premise is not used.
  internal notes)

Usage (CPU only; the cluster job wrapper is not used):
    AUTOMIX_SCORER_ADDR=127.0.0.1:9631 python3 -c \
      "import sys; sys.path.insert(0,'tools'); import scoring_client as c; \
       print(c.selftest())"
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _metadata
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ===========================================================================
# protocol (**keep byte-for-byte identical with tools/scoring_server.py**)
#   If you change this block, change the block of the same name on the server
#   to match. If they drift, the "proto" hash in HELLO disagrees and the
#   connection dies within 0 seconds.
# ===========================================================================
MAGIC = b"AMSC"
PROTO_VERSION = 1

T_HELLO = 1          # client -> server : exactly once, right after connecting
T_SCORE = 2          # client -> server : scoring request
T_REPLY = 3          # server -> client : scoring reply (success)
T_ERROR = 4          # server -> client : error
T_HELLO_ACK = 5      # server -> client : HELLO accepted + server-side fingerprint

# Variable-length frame used for HELLO / HELLO_ACK / ERROR-to-HELLO (12 B + body)
HELLO_FMT = "<4sHBBI"            # magic, version, type, _pad, body_len
HELLO_SIZE = struct.calcsize(HELLO_FMT)          # = 12

# SCORE request header (fixed 48 B) + payload
REQ_FMT = "<4sHBBQIIQBBHdI"
# magic(4s) version(H) type(B) which(B) req_id(Q) sr(I) n_ch(I) n_smp(Q)
# dtype(B) layout(B) flags(H) sb_chunk(d) nbytes(I)
REQ_SIZE = struct.calcsize(REQ_FMT)              # = 48

# SCORE reply header (fixed 36 B) + an err_len-byte message
REP_FMT = "<4sHBBQddI"
# magic(4s) version(H) type(B) status(B) req_id(Q) value(d) elapsed(d) err_len(I)
REP_SIZE = struct.calcsize(REP_FMT)              # = 36

WHICH_PQ = 1
WHICH_SB = 2
WHICH_NAME = {WHICH_PQ: "pq", WHICH_SB: "sb"}

DTYPE_F32 = 1                    # only '<f4' is allowed
LAYOUT_CN = 1                    # (C, N) C-contiguous
LAYOUT_N = 2                     # (N,)  1-D mono

ST_OK = 0
ST_BAD_REQUEST = 1               # deterministic: must not be retried
ST_SCORER_ERROR = 2              # deterministic: must not be retried
ST_TOO_LARGE = 3                 # deterministic: must not be retried
ST_FATAL = 4                     # the server kills itself after this reply

# The fingerprint keys **that both client and server can compute, and that are
# matched**. Values only the server has (cuda device name / ckpt sha256 /
# audiobox weights id / muq id) are not matched; they are reported to the
# client in HELLO_ACK and recorded.
SHARED_FP_KEYS: Tuple[str, ...] = (
    "python",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "torch",
    "sb_chunk_sec_env",
    "omp_num_threads",
    "mkl_num_threads",
    "openblas_num_threads",
    "numexpr_num_threads",
)


def proto_hash() -> str:
    """Hash of the protocol definition itself. Catches a mis-copy between the
    two files at connection time."""
    blob = "|".join([
        MAGIC.decode(), str(PROTO_VERSION), HELLO_FMT, REQ_FMT, REP_FMT,
        f"{T_HELLO}{T_SCORE}{T_REPLY}{T_ERROR}{T_HELLO_ACK}",
        f"{WHICH_PQ}{WHICH_SB}{DTYPE_F32}{LAYOUT_CN}{LAYOUT_N}",
        f"{ST_OK}{ST_BAD_REQUEST}{ST_SCORER_ERROR}{ST_TOO_LARGE}{ST_FATAL}",
        ",".join(SHARED_FP_KEYS),
    ])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
# ===========================================================================
# end of the protocol block
# ===========================================================================


DEFAULT_TIMEOUT_SEC = 300.0
DEFAULT_RETRY_BUDGET_SEC = 600.0
_BACKOFF = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_FAILS_BEFORE_FAILOVER = 2


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------
class ScoringClientError(RuntimeError):
    """Base class of the exceptions raised by this module."""


class ScoringServerError(ScoringClientError):
    """The server returned an ERROR frame (deterministic failure).
    **Do not retry.**"""


class ScoringServerUnavailable(ScoringClientError):
    """The transport retry budget was exhausted.

    If the caller (the per-song try/except in `agreement_loop_all.py`) catches
    this and records `status="error"`, `is_song_done` does not treat `error`
    as terminal, so the song is re-run on resume. No data is lost.
    """


class ScoringFingerprintMismatch(ScoringClientError):
    """Fingerprint matching in HELLO failed. **Never retry.**

    A safety device that turns "notice the bit-exactness problem after running
    for 3 hours" into "the connection dies within 0 seconds". If this fires,
    fix the configuration. Do not swallow it.
    """


class _TransportError(ScoringClientError):
    """Internal: a failure that may go away if the connection is re-established."""


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def _dist_version(name: str) -> Optional[str]:
    """Get a package version **without importing it** (metadata read only).

    The point is not to import torch. This upholds the design premise that the
    CPU client never creates a CUDA context.
    """
    try:
        return _metadata.version(name)
    except Exception:                                             # noqa: BLE001
        return None


def _norm_sb_chunk_env() -> str:
    v = os.environ.get("SB_CHUNK_SEC")
    if v is None or v == "":
        return "unset"
    try:
        return repr(float(v))
    except ValueError:
        return f"invalid:{v}"


def _thread_env(name: str) -> str:
    return os.environ.get(name, "unset")


def local_fingerprint() -> Dict[str, Optional[str]]:
    """Client-side fingerprint. Uses **the same normalization** as the
    server-side function of the same name."""
    return {
        "proto": proto_hash(),
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "numpy": _dist_version("numpy"),
        "scipy": _dist_version("scipy"),
        "librosa": _dist_version("librosa"),
        "soundfile": _dist_version("soundfile"),
        "torch": _dist_version("torch"),
        "sb_chunk_sec_env": _norm_sb_chunk_env(),
        "omp_num_threads": _thread_env("OMP_NUM_THREADS"),
        "mkl_num_threads": _thread_env("MKL_NUM_THREADS"),
        "openblas_num_threads": _thread_env("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": _thread_env("NUMEXPR_NUM_THREADS"),
    }


# ---------------------------------------------------------------------------
# socket helpers
# ---------------------------------------------------------------------------
def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    if n == 0:
        return buf
    mv = memoryview(buf)
    got = 0
    while got < n:
        try:
            k = sock.recv_into(mv[got:], n - got)
        except socket.timeout as ex:
            raise _TransportError(f"recv timeout after {got}/{n} B") from ex
        except OSError as ex:
            raise _TransportError(f"recv failed after {got}/{n} B: {ex!r}") from ex
        if k == 0:
            raise _TransportError(f"peer closed after {got}/{n} B")
        got += k
    return buf


def _sendall(sock: socket.socket, data) -> None:
    try:
        sock.sendall(data)
    except socket.timeout as ex:
        raise _TransportError("send timeout") from ex
    except OSError as ex:
        raise _TransportError(f"send failed: {ex!r}") from ex


def _parse_addr(text: str) -> Tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port:
        raise ScoringClientError(
            f"malformed endpoint (give it as host:port): {text!r}")
    return host.strip(), int(port)


def _coerce_addrs(spec) -> List[Tuple[str, int]]:
    """Normalize "h:p" / "h:p,h:p" / ["h:p", ...] / [("h", p), ...]."""
    if isinstance(spec, str):
        return [_parse_addr(p) for p in spec.split(",") if p.strip()]
    out: List[Tuple[str, int]] = []
    for item in spec:
        if isinstance(item, str):
            out.extend(_parse_addr(p) for p in item.split(",") if p.strip())
        else:
            host, port = item
            out.append((str(host), int(port)))
    if not out:
        raise ScoringClientError(f"cannot interpret the endpoints: {spec!r}")
    return out


def endpoints_from_env() -> List[Tuple[str, int]]:
    raw = (os.environ.get("AUTOMIX_SCORER_ADDR")
           or os.environ.get("AUTOMIX_SCORE_SERVERS") or "").strip()
    if not raw:
        return []
    return [_parse_addr(p) for p in raw.split(",") if p.strip()]


def enabled() -> bool:
    """Whether to use remote scoring. **If the env is unset, False = in-process
    as before.**"""
    return bool(endpoints_from_env())


# ---------------------------------------------------------------------------
# input validation and wire conversion
# ---------------------------------------------------------------------------
def _as_wire_f32(audio: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """(C, N) / (N,) float32 to the wire representation. **Not a single bit of
    the values changes.**

    - `np.ascontiguousarray` is a no-op if the array is already contiguous.
      Even for a non-contiguous array the copy does not change the values (only
      the memory layout). The result is the same even though the current
      in-process path hands non-contiguous views straight to the scorer:
      on the PQ side `audio.T` -> `clip` -> `astype(f32)` always builds a new
      array, and on the SB side `audio.mean(axis=0)` is a+b over a reduction of
      length 2, so no ordering difference arises.
    - **float64 is rejected by default.** On the current hot path
      `normalize_for_eval` returns `x.astype(np.float32)`, so only float32
      arrives (= this branch never fires in production). The reason we still do
      not cast silently is that the bit-safety of the cast depends on the
      condition "n_ch == 2 and the mean rounds exactly once", and it breaks
      silently the moment that condition changes. If you really need it, allow
      it explicitly with AUTOMIX_SCORER_ALLOW_CAST=1.
    """
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    if audio.dtype != np.float32:
        if os.environ.get("AUTOMIX_SCORER_ALLOW_CAST", "0") != "1":
            raise ScoringClientError(
                f"audio dtype={audio.dtype} is not allowed. "
                "The scoring server accepts only raw float32 bytes (for bit-exactness). "
                "Set AUTOMIX_SCORER_ALLOW_CAST=1 only if the down-cast is intended.")
        audio = audio.astype(np.float32)
    arr = np.ascontiguousarray(audio)
    if arr.ndim == 2:
        n_ch, n_smp = int(arr.shape[0]), int(arr.shape[1])
        layout = LAYOUT_CN
    elif arr.ndim == 1:
        n_ch, n_smp = 1, int(arr.shape[0])
        layout = LAYOUT_N
    else:
        raise ScoringClientError(
            f"invalid audio shape: {arr.shape} (only (C,N) or (N,))")
    if n_smp == 0:
        raise ScoringClientError("audio is empty")
    return arr, n_ch, n_smp, layout


# ---------------------------------------------------------------------------
# a single connection
# ---------------------------------------------------------------------------
class _Connection:
    """A synchronous connection to one server. Always exactly 1 request in flight."""

    def __init__(self, host: str, port: int, timeout_sec: float,
                 fp_relax: Sequence[str]):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.fp_relax = tuple(fp_relax)
        self.sock: Optional[socket.socket] = None
        self.server_fingerprint: Dict[str, object] = {}

    # -- connect and HELLO -----------------------------------------------
    def open(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout_sec)
        try:
            sock.connect((self.host, self.port))
        except OSError as ex:
            sock.close()
            raise _TransportError(
                f"connect {self.host}:{self.port} failed: {ex!r}") from ex
        self.sock = sock
        try:
            self._hello()
        except ScoringFingerprintMismatch:
            self.close()
            raise                       # config mismatch. Retrying will not fix it.
        except Exception:
            self.close()
            raise

    def _hello(self) -> None:
        assert self.sock is not None
        body = json.dumps({"fingerprint": local_fingerprint(),
                           "relax": list(self.fp_relax),
                           "pid": os.getpid(),
                           "client_index": os.environ.get(
                               "AUTOMIX_CLIENT_INDEX", "unset")},
                          sort_keys=True).encode("utf-8")
        hdr = struct.pack(HELLO_FMT, MAGIC, PROTO_VERSION, T_HELLO, 0, len(body))
        _sendall(self.sock, hdr)
        _sendall(self.sock, body)

        rhdr = _recv_exact(self.sock, HELLO_SIZE)
        magic, version, mtype, _pad, blen = struct.unpack(HELLO_FMT, rhdr)
        if magic != MAGIC:
            raise _TransportError(f"bad magic in HELLO reply: {magic!r}")
        if version != PROTO_VERSION:
            raise ScoringFingerprintMismatch(
                f"protocol version mismatch: client={PROTO_VERSION} server={version}")
        rbody = bytes(_recv_exact(self.sock, blen))
        if mtype == T_ERROR:
            raise ScoringFingerprintMismatch(
                f"the server rejected HELLO ({self.host}:{self.port}): "
                f"{rbody.decode('utf-8', 'replace')}")
        if mtype != T_HELLO_ACK:
            raise _TransportError(f"bad type in HELLO reply: {mtype}")
        try:
            self.server_fingerprint = json.loads(rbody.decode("utf-8"))
        except ValueError as ex:
            raise _TransportError(f"corrupt JSON in HELLO_ACK: {ex!r}") from ex

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # -- scoring ---------------------------------------------------------
    def request(self, which: int, req_id: int, arr: np.ndarray, sr: int,
                n_ch: int, n_smp: int, layout: int,
                sb_chunk: float) -> Tuple[float, float]:
        assert self.sock is not None
        nbytes = n_ch * n_smp * 4
        if nbytes > 0xFFFFFFFF:
            raise ScoringClientError(f"payload exceeds 4 GiB: {nbytes} B")
        hdr = struct.pack(REQ_FMT, MAGIC, PROTO_VERSION, T_SCORE, which,
                          req_id, int(sr), n_ch, n_smp,
                          DTYPE_F32, layout, 0, float(sb_chunk), nbytes)
        # The header and the body go out as two separate sends, so TCP_NODELAY
        # is mandatory (with Nagle on, a 40 ms delay is added). Set in open().
        _sendall(self.sock, hdr)
        _sendall(self.sock, memoryview(arr).cast("B"))   # no copy

        rhdr = _recv_exact(self.sock, REP_SIZE)
        (magic, version, mtype, status, r_req_id,
         value, elapsed, err_len) = struct.unpack(REP_FMT, rhdr)
        if magic != MAGIC or version != PROTO_VERSION:
            raise _TransportError(
                f"bad reply header: magic={magic!r} version={version}")
        err = bytes(_recv_exact(self.sock, err_len)).decode("utf-8", "replace")
        if r_req_id != req_id:
            raise _TransportError(
                f"req_id mismatch (pipeline desync): sent={req_id} got={r_req_id}")
        if mtype == T_ERROR or status != ST_OK:
            # A deterministic failure. **Do not retry.**
            #  - ST_BAD_REQUEST / ST_SCORER_ERROR / ST_TOO_LARGE:
            #      resending gives the same result.
            #  - ST_FATAL (CUDA OOM on the server):
            #      the server kills itself right after this reply. Resending to
            #      another server would let one and the same request take down
            #      all 4 machines one after another
            #      (a `Lushlife - Toynbee Suite` style accident). So we **do not
            #      move it**. Only this one song gets status="error", and since
            #      `is_song_done` does not treat error as terminal, it is re-run
            #      on resume.
            note = ("  NOTE: the server is shutting down due to CUDA OOM. Only this "
                    "song is marked error and left to resume." if status == ST_FATAL else "")
            raise ScoringServerError(
                f"[{self.host}:{self.port}] {WHICH_NAME.get(which, which)} "
                f"status={status}: {err}{note}")
        if mtype != T_REPLY:
            raise _TransportError(f"bad reply type: {mtype}")
        return float(value), float(elapsed)


# ---------------------------------------------------------------------------
# public classes
# ---------------------------------------------------------------------------
class RemoteScorer:
    """A proxy to the shared scoring server exposing `score(audio, sr) -> float`.

    A drop-in for the existing `AudioboxPQScorer` / `SongBenchMixingEar`.
    The hot path (`_score_candidate` at `agreement_loop_qwen.py:1093` and
    friends) takes the scorer as an argument and duck-types it, so **not one
    line of caller code changes.**
    """

    def __init__(self, which: str,
                 addrs: Optional[Sequence[Tuple[str, int]]] = None,
                 client_index: Optional[int] = None,
                 timeout_sec: Optional[float] = None,
                 retry_budget_sec: Optional[float] = None,
                 servers=None):
        """`servers` is an alias of `addrs`. Used by the verification harness
        (`tools/verify_scoring_server.py`) to pin the call to a single server.
        Accepts a "host:port" string, a comma-separated list of those, a list of
        those, or a list of (host, port) tuples.
        """
        w = which.lower()
        if w not in ("pq", "sb"):
            raise ScoringClientError(f"which must be 'pq' or 'sb': {which!r}")
        self.which_name = w
        self.which = WHICH_PQ if w == "pq" else WHICH_SB

        if addrs is None and servers is not None:
            addrs = _coerce_addrs(servers)
        self.addrs = list(addrs) if addrs is not None else endpoints_from_env()
        if not self.addrs:
            raise ScoringClientError(
                "no endpoints configured. Set AUTOMIX_SCORER_ADDR (or "
                "AUTOMIX_SCORE_SERVERS) to 'host:port,...', or, when "
                "scoring_client.enabled() is False, use the in-process scorer "
                "instead.")

        if client_index is None:
            try:
                client_index = int(os.environ.get("AUTOMIX_CLIENT_INDEX", "0"))
            except ValueError:
                client_index = 0
        self.client_index = int(client_index)
        # primary = index mod N. 22 clients / 4 servers -> a fixed 6,6,5,5 split.
        self._ep = self.client_index % len(self.addrs)

        self.timeout_sec = float(
            timeout_sec if timeout_sec is not None
            else os.environ.get("AUTOMIX_SCORER_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
        self.retry_budget_sec = float(
            retry_budget_sec if retry_budget_sec is not None
            else os.environ.get("AUTOMIX_SCORER_RETRY_BUDGET_SEC",
                                DEFAULT_RETRY_BUDGET_SEC))
        relax = os.environ.get("AUTOMIX_SCORER_FP_RELAX", "").strip()
        self.fp_relax = tuple(k.strip() for k in relax.split(",") if k.strip())
        if self.fp_relax:
            print(f"[scoring_client] WARNING: fingerprint matching relaxed: "
                  f"{self.fp_relax}  (weakens the bit-exactness guarantee)", flush=True)

        self._conn: Optional[_Connection] = None
        self._lock = threading.Lock()
        self._req_id = 0
        self.n_calls = 0
        self.n_reconnects = 0
        self.server_fingerprint: Dict[str, object] = {}

    # -- connection management -------------------------------------------
    def _ensure_conn(self) -> _Connection:
        if self._conn is not None and self._conn.sock is not None:
            return self._conn
        host, port = self.addrs[self._ep]
        conn = _Connection(host, port, self.timeout_sec, self.fp_relax)
        conn.open()
        self._conn = conn
        self.server_fingerprint = conn.server_fingerprint
        return conn

    def _drop_conn(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def close(self) -> None:
        with self._lock:
            self._drop_conn()

    # -- API compatible with the existing scorers ------------------------
    def score(self, audio: np.ndarray, sr: int, window=None) -> float:
        """Same arguments and same return value as today.

        `window` is not part of the protocol because the current hot path never
        passes it. If it is passed, raise instead of silently ignoring it
        (silently ignoring it is the worst failure mode).
        """
        if window is not None:
            raise ScoringClientError(
                "window= is not supported for remote scoring. Slice the audio on "
                "the caller side and pass that (the current hot path does not use "
                "window).")
        arr, n_ch, n_smp, layout = _as_wire_f32(audio)
        sb_chunk = self._sb_chunk_sec()

        with self._lock:
            self._req_id += 1
            req_id = self._req_id
            value = self._request_with_retry(req_id, arr, int(sr),
                                             n_ch, n_smp, layout, sb_chunk)
            self.n_calls += 1
        return value

    def _sb_chunk_sec(self) -> float:
        """Carry the effective SB_CHUNK_SEC with every request.

        `_score_7dim` at `songbench_mixing.py:157` reads
        `os.environ["SB_CHUNK_SEC"]` on every scoring call. If the server-side
        env differs, SB changes by up to 2.8 on audio longer than 300 seconds.
        A 12 s excerpt shows no difference, so **it only bares its teeth on
        full-song scoring.** Lifting the env dependency into the protocol kills
        it. The default 300.0 is the same as the default argument of
        `_score_7dim`.
        """
        v = os.environ.get("SB_CHUNK_SEC")
        if v is None or v == "":
            return 300.0
        return float(v)

    def _request_with_retry(self, req_id: int, arr, sr, n_ch, n_smp,
                            layout, sb_chunk) -> float:
        deadline = time.monotonic() + self.retry_budget_sec
        attempt = 0
        fails_here = 0
        last: Optional[BaseException] = None
        while True:
            try:
                conn = self._ensure_conn()
                value, _elapsed = conn.request(
                    self.which, req_id, arr, sr, n_ch, n_smp, layout, sb_chunk)
                return value
            except _TransportError as ex:
                last = ex
            except OSError as ex:
                last = _TransportError(f"socket error: {ex!r}")
            # ScoringServerError (a deterministic scoring error returned by the
            # server) and ScoringFingerprintMismatch (config mismatch) are
            # **not caught**. Resending gives the same result, so let them
            # propagate to the caller as they are.

            # ---- we only get here on a transport failure ----
            self._drop_conn()
            self.n_reconnects += 1
            fails_here += 1
            if fails_here >= _FAILS_BEFORE_FAILOVER and len(self.addrs) > 1:
                self._ep = (self._ep + 1) % len(self.addrs)
                fails_here = 0
                print(f"[scoring_client] failover -> {self.addrs[self._ep]} "
                      f"({self.which_name}, idx={self.client_index})", flush=True)
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise ScoringServerUnavailable(
                    f"cannot reach the scoring server for {self.retry_budget_sec:.0f} s "
                    f"({self.which_name}, addrs={self.addrs}): {last!r}") from last
            wait = min(_BACKOFF[min(attempt, len(_BACKOFF) - 1)], remain)
            attempt += 1
            if attempt == 1 or attempt % 5 == 0:
                print(f"[scoring_client] retry {attempt} in {wait:.1f}s "
                      f"({self.which_name} -> {self.addrs[self._ep]}): {last!r}",
                      flush=True)
            time.sleep(wait)

    # -- placeholders for API the existing scorers expose -----------------
    def release_model(self) -> None:
        """A no-op for API compatibility with the in-process scorer.

        The model lives on the server; the client does not hold a single byte
        of it.
        """
        return None

    def __repr__(self) -> str:                                    # pragma: no cover
        return (f"<RemoteScorer {self.which_name} -> {self.addrs[self._ep]} "
                f"calls={self.n_calls} reconnects={self.n_reconnects}>")


class AudioboxPQClient(RemoteScorer):
    """Drop-in for `AudioboxPQScorer`. `score(audio, sr) -> Audiobox-PQ`."""

    def __init__(self, **kw):
        super().__init__("pq", **kw)


class SongBenchClient(RemoteScorer):
    """Drop-in for `SongBenchMixingEar`. `score(audio, sr) -> Mixing [1,10]`."""

    name = "songbench_mixing"
    tier = 6
    uses_gpu_model = False       # this process does not use the GPU

    def __init__(self, **kw):
        super().__init__("sb", **kw)

    def score_all(self, audio: np.ndarray, sr: int) -> dict:
        raise NotImplementedError(
            "score_all is not supported for remote scoring (the hot path uses only "
            "score()). If you need the 7 dimensions, use the in-process "
            "SongBenchMixingEar.")


# Aliases (absorb spelling variants on the caller side).
AutoboxPQClient = AudioboxPQClient
AudioboxPQScorerClient = AudioboxPQClient
SongBenchMixingClient = SongBenchClient


def make_scorers(client_index: Optional[int] = None
                 ) -> Optional[Dict[str, RemoteScorer]]:
    """Return `{"pq":..., "sb":...}` if the env is set, otherwise None.

    Caller side:
        rs = scoring_client.make_scorers()
        scorers = rs if rs is not None else {build the in-process ones}
    """
    if not enabled():
        return None
    return {"pq": AudioboxPQClient(client_index=client_index),
            "sb": SongBenchClient(client_index=client_index)}


# ---------------------------------------------------------------------------
# standalone check (CPU only; assumes a server is running)
# ---------------------------------------------------------------------------
def selftest(sec: float = 12.0, sr: int = 44100, seed: int = 0) -> str:
    """Score fixed-seed dummy audio once per scorer and print value and timing."""
    if not enabled():
        return "AUTOMIX_SCORER_ADDR is unset (remote scoring disabled)"
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((2, int(sec * sr))) * 0.1).astype(np.float32)
    out = []
    for cls in (AudioboxPQClient, SongBenchClient):
        s = cls()
        t0 = time.perf_counter()
        v = s.score(a, sr)
        dt = time.perf_counter() - t0
        out.append(f"{s.which_name}={v!r} ({dt*1000:.1f} ms) "
                   f"server={s.server_fingerprint.get('server', '?')}")
        s.close()
    return "  ".join(out)


if __name__ == "__main__":                                        # pragma: no cover
    print(f"proto={proto_hash()} req_hdr={REQ_SIZE}B rep_hdr={REP_SIZE}B")
    print(f"enabled={enabled()} endpoints={endpoints_from_env()}")
    print(json.dumps(local_fingerprint(), indent=2, sort_keys=True))
    if enabled():
        print(selftest())

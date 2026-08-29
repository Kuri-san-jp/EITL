#!/usr/bin/env python3
"""Shared scoring server: keeps Audiobox-PQ and SongBench-Mixing resident in a
single process so that search processes can call them over TCP loopback.

The goal is **to increase the number of search processes per GPU**. Once the
scoring models are removed from the search process, search runs with 0 GPUs and
the parallelism can be raised as far as the CPU allows.

============================================================================
[HARD REQUIREMENT: bit-identical] Scores must not differ by a single bit from
the current in-process scoring
============================================================================
Otherwise comparison with the existing `outputs/runs/exc10k_random_pq`
(22 songs completed) becomes impossible. What we do to guarantee this:

1. **No batching. Batching is impossible.**
   There is **exactly one scoring thread: the main thread**. It pops one item
   at a time from the queue and calls
   `pq_scorer.score(audio, sr)` / `sb_ear.score(audio, sr)`.
   There is no code path that bundles multiple requests, so neither padding nor
   attention-mask changes **can possibly occur**. We assert `in_flight == 1` on
   every iteration to preserve this structure.
   Scoring lives on the main thread because `AudioboxPQScorer.score`
   internally calls `asyncio.run(self.ear.evaluate(...))`
   (`src/mix_orchestrator/eval/audiobox_pq.py`). Today it runs on the search
   process's main thread, so we keep the execution context identical to the
   current one.

2. **Audio is received as raw float32 bytes.** No encode / quantization /
   resample of any kind. It is restored with
   `np.frombuffer(buf, "<f4").reshape(n_ch, n_smp)`.
   A raw float32 byte round-trip is mathematically the identity map, and
   reshape is only a metadata operation.

3. **Scorers are called with exactly the same arguments as today.** `window` is
   never passed by the current hot path, so it is not part of the protocol.

4. **SB_CHUNK_SEC is carried per request.**
   `_score_7dim` in `songbench_mixing.py` reads
   `os.environ["SB_CHUNK_SEC"]` on every scoring call. If the server's env
   drifts, SB changes by up to 2.8 on audio longer than 300 seconds. There is
   no difference on 12-second excerpts, so **it only bares its teeth on
   full-song scoring**. We kill the env dependency by lifting it into the
   protocol. Since in-flight is always 1, rewriting os.environ right before
   scoring cannot race.

5. **Never enable cudnn.benchmark.** Autotuning picks kernels from runtime
   measurements, so values can wobble. We assert it is False at startup and
   abort if it is True.

6. **The server holds no state.**
   `AudioboxEar._run_real` and `SongBenchMixingEar._score_7dim` are pure
   functions under `eval()` + `torch.no_grad()` with no internal state. That is
   why, however the requests of 22 clients are interleaved, each request's value
   matches a standalone run, and client retransmission (at-least-once) is safe.
   **Do not introduce result caches, adaptive batching, dynamic chunk tuning,
   or anything that feeds statistics back into the scores.** The moment you do,
   this guarantee breaks silently.

============================================================================
Thread layout
============================================================================
  main thread     : scoring loop (only one). pop 1 item from queue -> score
                    -> Event set
  acceptor thread : accepts and spawns a per-connection reader
  reader thread   : one per connection. Reads a frame, puts it on the queue,
                    waits on the Event and **writes the reply to its own
                    socket**.
                    -> the scoring thread never blocks on a socket write.

Back pressure: queue full -> reader blocks in put() -> the TCP receive window
closes -> the client's sendall blocks. The client is synchronous (1 in-flight
request per connection), so this is the correct behaviour.

============================================================================
Startup (uses the GPU, so go through the cluster job wrapper)
============================================================================
  JOB_TIME=12:00:00 /path/to/the cluster job wrapper \
      python3 tools/scoring_server.py --port 9631

  ready marker: outputs/score_server/ready_<port>  (written after warm-up)

Failure policy:
  - CUDA OOM -> return an ERROR frame and **then terminate this process**.
    The CUDA context after an OOM cannot be trusted. The supervisor restarts
    it and clients recover through transport-level retries.
  - bind failure -> **exit with an error immediately instead of falling back to
    another port**. Moving silently makes clients keep connecting to the old
    port and fail without a word.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# !!! import ORDER !!!
# Inside the container, `_common` must be imported **first** or cffi conflicts
# (the project setup notes). Keep it before numpy / torch. Do not move these 4 lines.
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]

# --print-fingerprint is sometimes run outside the container to cross-check against the
# client side. Outside the container, `_common` puts python_packages on sys.path and
# breaks scipy (the project setup notes), and it also changes what importlib.metadata
# resolves to for the fingerprint. Do not import it when we only print the
# fingerprint.
# AUTOMIX_SCORER_SKIP_COMMON=1 has the same effect (for
# tools/scoring_selftest_cpu.py).
# **Never set it in production.** If _common is not imported first inside the
# container, cffi conflicts.
_SKIP_COMMON = ("--print-fingerprint" in _sys.argv
                or _os.environ.get("AUTOMIX_SCORER_SKIP_COMMON") == "1")
if not _SKIP_COMMON:
    _sys.path.insert(0, str(ROOT / "experiments"))
    import _common  # noqa: E402,F401  (avoids cffi conflict + HF cache env. import first)
else:
    _common = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
import argparse                                                   # noqa: E402
import hashlib                                                    # noqa: E402
import importlib.metadata as _metadata                            # noqa: E402
import json                                                       # noqa: E402
import os                                                         # noqa: E402
import queue                                                      # noqa: E402
import signal                                                     # noqa: E402
import socket                                                     # noqa: E402
import struct                                                     # noqa: E402
import sys                                                        # noqa: E402
import threading                                                  # noqa: E402
import time                                                       # noqa: E402
import traceback                                                  # noqa: E402
from typing import Dict, List, Optional, Tuple                    # noqa: E402

import numpy as np                                                # noqa: E402

# ===========================================================================
# protocol (**must be kept exactly identical to** tools/scoring_client.py)
#   If you change this block, change the block of the same name on the client
#   side to the same content. If they drift, the HELLO "proto" hash disagrees
#   and the connection dies at 0 seconds.
# ===========================================================================
MAGIC = b"AMSC"
PROTO_VERSION = 1

T_HELLO = 1          # client -> server : exactly once, right after connecting
T_SCORE = 2          # client -> server : scoring request
T_REPLY = 3          # server -> client : scoring reply (success)
T_ERROR = 4          # server -> client : error
T_HELLO_ACK = 5      # server -> client : HELLO accepted + server-side fingerprint

HELLO_FMT = "<4sHBBI"            # magic, version, type, _pad, body_len
HELLO_SIZE = struct.calcsize(HELLO_FMT)          # = 12

REQ_FMT = "<4sHBBQIIQBBHdI"
# magic(4s) version(H) type(B) which(B) req_id(Q) sr(I) n_ch(I) n_smp(Q)
# dtype(B) layout(B) flags(H) sb_chunk(d) nbytes(I)
REQ_SIZE = struct.calcsize(REQ_FMT)              # = 48

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
ST_BAD_REQUEST = 1               # deterministic: the client does not retry
ST_SCORER_ERROR = 2              # deterministic: the client does not retry
ST_TOO_LARGE = 3                 # deterministic: the client does not retry
ST_FATAL = 4                     # the server kills itself after this reply

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
    """Hash of the protocol definition itself. Detects copy errors between the
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
# end of protocol block
# ===========================================================================


# Upper bound on duration (seconds) above which we reject without attempting to
# score.
# `Lushlife - Toynbee Suite` (628.6 s) needs ~47 GiB for full-song SB scoring,
# which does not fit on an A100-40GB, and it is already treated as `excluded` by
# `agreement_loop_all.is_song_done`. On a shared server **a single request for
# this one song takes down the 5-6 clients sharing the process with it**. Reject
# it at the protocol level.
DEFAULT_MAX_SEC = 600.0
HARD_MAX_BYTES = 512 * 1024 * 1024        # prevents huge allocs from corrupt headers


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def _dist_version(name: str) -> Optional[str]:
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
    """Built with **the same keys and the same normalization** as the client-side
    `local_fingerprint()`."""
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


def _sha256_file(path: _Path, chunk: int = 8 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def server_only_fingerprint(pq_scorer, sb_ear) -> Dict[str, Optional[str]]:
    """Identity of the scoring configuration itself, which the client cannot
    compute.

    Not used for verification (the client does not have it, so there is nothing
    to compare against). It is returned to the client in HELLO_ACK so it can be
    recorded with the run. The point is to be able to audit afterwards "which
    server produced these scores".
    """
    import torch
    ckpt = sb_ear.repo / "ckpt" / "songbench.safetensors"
    cfg = sb_ear.repo / "configs" / "songbench.yaml"
    dev = None
    try:
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name()
    except Exception:                                             # noqa: BLE001
        dev = None
    return {
        "server": f"{socket.gethostname()}/pid{os.getpid()}",
        "audiobox_weights": getattr(pq_scorer.ear, "weights", None),
        "muq_model_id": getattr(sb_ear, "muq_model_id", None),
        "songbench_repo": str(sb_ear.repo),
        "songbench_ckpt_sha256": _sha256_file(ckpt),
        "songbench_cfg_sha256": _sha256_file(cfg),
        "cuda_device_name": dev,
        "torch_runtime": getattr(torch, "__version__", None),
        "cudnn_benchmark": str(bool(torch.backends.cudnn.benchmark)),
    }


def compare_fingerprints(client_fp: Dict[str, object],
                         server_fp: Dict[str, object],
                         relax: Tuple[str, ...]) -> List[str]:
    """Verification. An empty return list means they match.

    Rules:
      - "proto" **can never be relaxed**. A mismatch means the protocol
        definition was copied incorrectly, so reject immediately.
      - SHARED_FP_KEYS: None on only one side is a mismatch (the image differs
        from what we expect). None on both sides is taken as agreement that
        "neither side has this package" and is allowed through (the caller
        emits the warning).
      - Keys that only the server has are not verified (the client cannot
        compute them).
    """
    bad: List[str] = []
    if client_fp.get("proto") != server_fp.get("proto"):
        bad.append(f"proto: client={client_fp.get('proto')} "
                   f"server={server_fp.get('proto')} "
                   "(the protocol blocks of scoring_client.py and "
                   "scoring_server.py disagree)")
    for k in SHARED_FP_KEYS:
        if k in relax:
            continue
        cv, sv = client_fp.get(k), server_fp.get(k)
        if cv is None and sv is None:
            continue
        if cv != sv:
            bad.append(f"{k}: client={cv!r} server={sv!r}")
    return bad


# ---------------------------------------------------------------------------
# socket helpers
# ---------------------------------------------------------------------------
class _PeerGone(Exception):
    pass


def _recv_exact_into(sock: socket.socket, mv: memoryview, n: int) -> None:
    got = 0
    while got < n:
        try:
            k = sock.recv_into(mv[got:], n - got)
        except OSError as ex:
            raise _PeerGone(f"recv failed after {got}/{n} B: {ex!r}") from ex
        if k == 0:
            raise _PeerGone(f"peer closed after {got}/{n} B")
        got += k


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    if n:
        _recv_exact_into(sock, memoryview(buf), n)
    return buf


def _sendall(sock: socket.socket, data) -> None:
    try:
        sock.sendall(data)
    except OSError as ex:
        raise _PeerGone(f"send failed: {ex!r}") from ex


_DRAIN_CHUNK = 1 << 20
_DRAIN_DEADLINE_SEC = 60.0


def _drain(sock: socket.socket, n: int) -> bool:
    """After rejecting a request, read the payload the client is still sending
    to completion and discard it.

    **Without this, operation breaks long before bit-identity is at stake.**
    Closing without reading the payload sends an RST to a client that is still
    in the middle of sending 4.2 MB, which also discards the ERROR frame we
    just sent. From the client's point of view this looks only like "the
    connection dropped mid-send" = a transport failure, so it retries a
    deterministic rejection (audio that is too long, etc.) forever and, through
    failover, sends the same request around to all 4 servers. Reproduced in
    self-test T6.

    Reading it to completion preserves framing, so we can keep the connection
    and move on to the next request.
    Returns: True if it was fully drained.
    """
    if n <= 0:
        return True
    buf = bytearray(min(_DRAIN_CHUNK, n))
    mv = memoryview(buf)
    left = n
    deadline = time.monotonic() + _DRAIN_DEADLINE_SEC
    try:
        while left > 0:
            remain = deadline - time.monotonic()
            if remain <= 0:
                return False
            sock.settimeout(remain)
            k = sock.recv_into(mv, min(len(buf), left))
            if k == 0:
                return False
            left -= k
        return True
    except OSError:
        return False
    finally:
        try:
            sock.settimeout(None)
        except OSError:
            pass


def _pack_reply(req_id: int, status: int, value: float, elapsed: float,
                err: str = "") -> bytes:
    eb = err.encode("utf-8")[:65535]
    mtype = T_REPLY if status == ST_OK else T_ERROR
    return struct.pack(REP_FMT, MAGIC, PROTO_VERSION, mtype, status,
                       req_id, float(value), float(elapsed), len(eb)) + eb


def _pack_hello(mtype: int, body: bytes) -> bytes:
    return struct.pack(HELLO_FMT, MAGIC, PROTO_VERSION, mtype, 0,
                       len(body)) + body


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------
class _Job:
    __slots__ = ("which", "req_id", "audio", "sr", "sb_chunk", "peer",
                 "done", "status", "value", "elapsed", "err")

    def __init__(self, which: int, req_id: int, audio: np.ndarray, sr: int,
                 sb_chunk: float, peer: str):
        self.which = which
        self.req_id = req_id
        self.audio = audio
        self.sr = sr
        self.sb_chunk = sb_chunk
        self.peer = peer
        self.done = threading.Event()
        self.status = ST_OK
        self.value = 0.0
        self.elapsed = 0.0
        self.err = ""


# ---------------------------------------------------------------------------
# statistics (**never affects the scores**. display only)
# ---------------------------------------------------------------------------
class _Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.n_ok = {WHICH_PQ: 0, WHICH_SB: 0}
        self.sum_sec = {WHICH_PQ: 0.0, WHICH_SB: 0.0}
        self.max_sec = {WHICH_PQ: 0.0, WHICH_SB: 0.0}
        self.n_err = 0
        self.n_conn_open = 0
        self.n_conn_close = 0
        self.last_err = ""
        self.t0 = time.monotonic()
        # most recent window (reset on every periodic print)
        self._w_n = 0
        self._w_sec = 0.0
        self._w_err = 0
        self._w_t0 = time.monotonic()
        self._w_qwait = 0.0

    def record(self, which: int, sec: float, ok: bool, err: str,
               qwait: float) -> None:
        with self.lock:
            if ok:
                self.n_ok[which] += 1
                self.sum_sec[which] += sec
                if sec > self.max_sec[which]:
                    self.max_sec[which] = sec
                self._w_n += 1
                self._w_sec += sec
                self._w_qwait += qwait
            else:
                self.n_err += 1
                self._w_err += 1
                self.last_err = err[:300]

    def conn(self, delta_open: int = 0, delta_close: int = 0) -> None:
        with self.lock:
            self.n_conn_open += delta_open
            self.n_conn_close += delta_close

    def render_and_reset_window(self, qsize: int) -> str:
        with self.lock:
            now = time.monotonic()
            wall = max(now - self._w_t0, 1e-9)
            n, sec, errs, qwait = self._w_n, self._w_sec, self._w_err, self._w_qwait
            self._w_n = 0
            self._w_sec = 0.0
            self._w_err = 0
            self._w_qwait = 0.0
            self._w_t0 = now
            tot_pq, tot_sb = self.n_ok[WHICH_PQ], self.n_ok[WHICH_SB]
            avg_pq = self.sum_sec[WHICH_PQ] / tot_pq if tot_pq else 0.0
            avg_sb = self.sum_sec[WHICH_SB] / tot_sb if tot_sb else 0.0
            mx_pq, mx_sb = self.max_sec[WHICH_PQ], self.max_sec[WHICH_SB]
            live = self.n_conn_open - self.n_conn_close
            tot_err = self.n_err
            up = now - self.t0
            last_err = self.last_err
        rate = n / wall
        avg_w = (sec / n * 1000.0) if n else 0.0
        avg_q = (qwait / n * 1000.0) if n else 0.0
        line = (f"[score-server] up={up:7.0f}s conn={live:2d} q={qsize:3d} "
                f"| window: {n:5d} req {rate:6.2f} req/s "
                f"score={avg_w:7.1f}ms qwait={avg_q:7.1f}ms err={errs} "
                f"| total: pq={tot_pq} (avg {avg_pq*1000:.1f} max {mx_pq*1000:.0f} ms) "
                f"sb={tot_sb} (avg {avg_sb*1000:.1f} max {mx_sb*1000:.0f} ms) "
                f"err={tot_err}")
        if last_err:
            line += f"\n[score-server]   last_err: {last_err}"
        return line


# ---------------------------------------------------------------------------
# the server itself
# ---------------------------------------------------------------------------
class ScoringServer:
    """1 process = 1 CUDA context = 1 set of models = **1 scoring thread**.

    Serialization is guaranteed structurally by "a dedicated thread + a queue",
    not by a lock. Since only one scoring thread exists, batching is not
    something we "avoid": it is **impossible**.
    """

    def __init__(self, host: str, port: int, max_sec: float,
                 stats_sec: float, backlog: int, max_clients: int,
                 ready_dir: _Path, strict_fp: bool = False):
        self.host = host
        self.port = port
        self.max_sec = max_sec
        self.stats_sec = stats_sec
        self.backlog = backlog
        self.strict_fp = strict_fp
        self.ready_dir = ready_dir
        self.ready_path = ready_dir / f"ready_{port}"

        self.q: "queue.Queue[_Job]" = queue.Queue(maxsize=max(2 * max_clients, 4))
        self.stats = _Stats()
        self.stop = threading.Event()
        self.srv: Optional[socket.socket] = None
        # live connections. On shutdown we wake them up so clients are not left
        # waiting.
        self._conns: "set[socket.socket]" = set()
        self._conns_lock = threading.Lock()

        self.pq_scorer = None
        self.sb_ear = None
        self.fp_shared: Dict[str, object] = {}
        self.fp_server: Dict[str, object] = {}

        # For explicitly verifying serialization. Only the scoring thread
        # touches it.
        self._in_flight = 0

    # -- models -----------------------------------------------------------
    def build_scorers(self) -> None:
        """Build the real models (no proxies allowed).

        Must be **the same construction** (defaults, no arguments) as
        `agreement_loop_all.py:1559` / `size_sweep_run.py:809`.
        """
        import torch
        # cudnn.benchmark makes values wobble because autotuning picks kernels
        # from runtime measurements. No code in the project enables it (grepped),
        # but if something does, stop here. **Never set it to True.**
        if torch.backends.cudnn.benchmark:
            raise RuntimeError(
                "torch.backends.cudnn.benchmark is True. Autotuning can make "
                "scores wobble, so it is not allowed on the scoring server.")

        from mix_orchestrator.eval.audiobox_pq import AudioboxPQScorer
        from mix_orchestrator.ears.tier6_music_reward import SongBenchMixingEar
        self.pq_scorer = AudioboxPQScorer()
        self.sb_ear = SongBenchMixingEar()

        self.fp_shared = local_fingerprint()
        print(f"[score-server] torch={torch.__version__} "
              f"cuda_available={torch.cuda.is_available()}", flush=True)

    def warmup(self, sec: float, sr: int = 44100) -> None:
        """Load the models and score one fixed dummy input **before** listen().

        Prevents 22 clients from hitting a cold server, triggering false failure
        detection and a retry storm. Warm-up is a deterministic forward pass, so
        it does not affect later values (assuming cudnn.benchmark is not
        enabled, which build_scorers asserts).
        """
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((2, int(sec * sr))) * 0.1).astype(np.float32)
        t0 = time.perf_counter()
        pq = self.pq_scorer.score(a, sr)
        t1 = time.perf_counter()
        prev = os.environ.get("SB_CHUNK_SEC")
        try:
            sb = self.sb_ear.score(a, sr)
        finally:
            if prev is None:
                os.environ.pop("SB_CHUNK_SEC", None)
            else:
                os.environ["SB_CHUNK_SEC"] = prev
        t2 = time.perf_counter()
        print(f"[score-server] warmup {sec:.1f}s audio: "
              f"pq={pq:.6f} ({(t1-t0)*1000:.0f}ms) "
              f"sb={sb:.6f} ({(t2-t1)*1000:.0f}ms)", flush=True)

        self.fp_server = server_only_fingerprint(self.pq_scorer, self.sb_ear)
        for k, v in sorted(self.fp_server.items()):
            print(f"[score-server]   {k} = {v}", flush=True)

    # -- listen / accept ---------------------------------------------------
    def bind_listen(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # **Do not use** SO_REUSEPORT. If connections are spread across multiple
        # listeners, the client allocation skews as a multinomial distribution
        # (expected 5.5, worst case 9).
        # Allocation is decided deterministically by AUTOMIX_CLIENT_INDEX on the
        # client side.
        try:
            srv.bind((self.host, self.port))
        except OSError as ex:
            # With --network=host the port is shared across the whole node.
            # **Do not fall back to another port.** If we did, clients would
            # keep connecting to the old port and fail without a word. Exit with
            # an error immediately.
            raise SystemExit(
                f"[score-server] FATAL: bind {self.host}:{self.port} failed: {ex!r}\n"
                f"  Ports are shared across the whole node because of --network=host.\n"
                f"  Check usage with `ss -tln` and pass a free port with --port.\n"
                f"  (No automatic fallback to another port: clients would keep using the old one)")
        srv.listen(self.backlog)
        srv.settimeout(0.5)
        self.srv = srv
        print(f"[score-server] listening on {self.host}:{self.port} "
              f"backlog={self.backlog}", flush=True)

    def write_ready(self) -> None:
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "addr": f"{self.host}:{self.port}",
            "started_at": time.time(),
            "max_sec": self.max_sec,
            "fingerprint": self.fp_shared,
            "server_fingerprint": self.fp_server,
        }
        tmp = self.ready_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
        tmp.replace(self.ready_path)
        print(f"[score-server] ready -> {self.ready_path}", flush=True)

    def clear_ready(self) -> None:
        try:
            self.ready_path.unlink()
        except OSError:
            pass

    def acceptor(self) -> None:
        assert self.srv is not None
        while not self.stop.is_set():
            try:
                conn, addr = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                if self.stop.is_set():
                    return
                continue
            conn.setblocking(True)      # do not inherit the listener's timeout
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Do not leave reader threads for dead clients around forever. We do
            # not set a recv timeout: idling for seconds to tens of seconds
            # while waiting on the LLM is normal behaviour, and cutting the
            # connection there would induce duplicate computation.
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            peer = f"{addr[0]}:{addr[1]}"
            t = threading.Thread(target=self._reader, args=(conn, peer),
                                 name=f"reader-{peer}", daemon=True)
            t.start()

    # -- per-connection reader ---------------------------------------------
    def _reader(self, sock: socket.socket, peer: str) -> None:
        self.stats.conn(delta_open=1)
        with self._conns_lock:
            self._conns.add(sock)
        try:
            self._handshake(sock, peer)
            self._serve_loop(sock, peer)
        except _PeerGone as ex:
            print(f"[score-server] conn {peer} closed: {ex}", flush=True)
        except Exception:                                         # noqa: BLE001
            print(f"[score-server] conn {peer} reader error:\n"
                  f"{traceback.format_exc()}", flush=True)
        finally:
            self.stats.conn(delta_close=1)
            with self._conns_lock:
                self._conns.discard(sock)
            try:
                sock.close()
            except OSError:
                pass

    def shutdown(self) -> None:
        """Shutdown. **Never disappear silently while clients are waiting.**

        1. set stop to end the acceptor / score_loop
        2. close the listen socket
        3. **shut down every live connection** -> readers wake up in recv/send
           and the client side can reconnect / fail over, treating it as a
           transport failure
        4. reply to jobs left in the queue (if the Event is not set, the reader
           waits forever)

        Without 3 and 4, clients hang on every graceful stop. Reproduced in
        self-test T9.
        """
        self.stop.set()
        if self.srv is not None:
            try:
                self.srv.close()
            except OSError:
                pass
        with self._conns_lock:
            socks = list(self._conns)
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._fail_pending("server is shutting down")
        self.clear_ready()

    def _fail_pending(self, reason: str) -> None:
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                return
            job.status = ST_SCORER_ERROR
            job.value = 0.0
            job.elapsed = 0.0
            job.err = reason
            job.done.set()

    def _handshake(self, sock: socket.socket, peer: str) -> None:
        hdr = _recv_exact(sock, HELLO_SIZE)
        magic, version, mtype, _pad, blen = struct.unpack(HELLO_FMT, hdr)
        if magic != MAGIC:
            _sendall(sock, _pack_hello(T_ERROR, b"bad magic"))
            raise _PeerGone(f"bad magic from {peer}: {magic!r}")
        if version != PROTO_VERSION:
            msg = (f"protocol version mismatch: server={PROTO_VERSION} "
                   f"client={version}").encode("utf-8")
            _sendall(sock, _pack_hello(T_ERROR, msg))
            raise _PeerGone(f"version mismatch from {peer}")
        if mtype != T_HELLO:
            _sendall(sock, _pack_hello(T_ERROR, b"first frame must be HELLO"))
            raise _PeerGone(f"first frame not HELLO from {peer}")
        if blen > 1 << 20:
            _sendall(sock, _pack_hello(T_ERROR, b"HELLO body too large"))
            raise _PeerGone(f"HELLO body too large from {peer}")
        body = bytes(_recv_exact(sock, blen))
        try:
            info = json.loads(body.decode("utf-8"))
            client_fp = dict(info.get("fingerprint") or {})
            relax = tuple(info.get("relax") or ())
        except Exception as ex:                                   # noqa: BLE001
            _sendall(sock, _pack_hello(T_ERROR,
                                       f"HELLO JSON is malformed: {ex!r}"
                                       .encode("utf-8")))
            raise _PeerGone(f"bad HELLO json from {peer}") from ex

        if self.strict_fp and relax:
            msg = (f"connected with relax={list(relax)} to a server running --strict. "
                   "Unset AUTOMIX_SCORER_FP_RELAX.")
            print(f"[score-server] REJECT {peer}: {msg}", flush=True)
            _sendall(sock, _pack_hello(T_ERROR, msg.encode("utf-8")))
            raise _PeerGone(f"relax not allowed from {peer}")

        bad = compare_fingerprints(client_fp, self.fp_shared, relax)
        if bad:
            # **The most valuable safety device in this design.**
            # It turns a bit mismatch from "you notice after running 3 hours"
            # into "it dies at 0 seconds of connection".
            msg = ("config fingerprint mismatch (rejected; bit-identity not guaranteed):\n  "
                   + "\n  ".join(bad))
            print(f"[score-server] REJECT {peer}\n{msg}", flush=True)
            _sendall(sock, _pack_hello(T_ERROR, msg.encode("utf-8")))
            raise _PeerGone(f"fingerprint mismatch from {peer}")
        if relax:
            print(f"[score-server] WARNING {peer} is relaxing the fingerprint "
                  f"check: {relax}", flush=True)

        ack = dict(self.fp_shared)
        ack.update(self.fp_server)
        ack["max_sec"] = self.max_sec
        _sendall(sock, _pack_hello(
            T_HELLO_ACK,
            json.dumps(ack, sort_keys=True, default=str).encode("utf-8")))
        print(f"[score-server] accepted {peer} "
              f"(client_index={info.get('client_index')} pid={info.get('pid')})",
              flush=True)

    def _serve_loop(self, sock: socket.socket, peer: str) -> None:
        while not self.stop.is_set():
            try:
                hdr = _recv_exact(sock, REQ_SIZE)
            except _PeerGone as ex:
                if "closed after 0/" in str(ex):
                    return                       # the client just closed cleanly
                raise
            (magic, version, mtype, which, req_id, sr, n_ch, n_smp,
             dtype, layout, flags, sb_chunk, nbytes) = struct.unpack(REQ_FMT, hdr)

            if magic != MAGIC or version != PROTO_VERSION or mtype != T_SCORE:
                _sendall(sock, _pack_reply(
                    req_id, ST_BAD_REQUEST, 0.0, 0.0,
                    f"bad frame: magic={magic!r} version={version} type={mtype}"))
                raise _PeerGone(f"bad frame from {peer}")

            status, msg = self._validate(which, sr, n_ch, n_smp, dtype, layout,
                                         flags, nbytes)
            if status != ST_OK:
                # When rejecting, **do not alloc the payload, but still read it
                # to the end**.
                #   - no alloc: so we do not OOM ourselves by trusting a corrupt
                #               nbytes.
                #   - read it:  closing without reading makes the RST swallow
                #               the ERROR frame, and the client retries a
                #               deterministic rejection forever (see the _drain
                #               docstring).
                _sendall(sock, _pack_reply(req_id, status, 0.0, 0.0, msg))
                self.stats.record(which if which in WHICH_NAME else WHICH_PQ,
                                  0.0, False, msg, 0.0)
                print(f"[score-server] reject {peer} status={status}: {msg}",
                      flush=True)
                if _drain(sock, nbytes):
                    continue         # framing recovered. keep the connection.
                raise _PeerGone(f"drain failed after reject from {peer}")

            buf = bytearray(nbytes)
            if nbytes:
                _recv_exact_into(sock, memoryview(buf), nbytes)

            # Turn the raw float32 bytes straight back into an ndarray.
            # **The identity map.** reshape is only a metadata operation.
            arr = np.frombuffer(buf, dtype="<f4")
            arr = arr.reshape(n_ch, n_smp) if layout == LAYOUT_CN \
                else arr.reshape(n_smp)

            job = _Job(which, req_id, arr, int(sr), float(sb_chunk), peer)
            t_enq = time.perf_counter()
            # If full, put blocks = back pressure (the TCP receive window closes
            # and the client's sendall stalls). But never wait forever while
            # shutting down.
            while True:
                try:
                    self.q.put(job, timeout=0.25)
                    break
                except queue.Full:
                    if self.stop.is_set():
                        raise _PeerGone("server stopping (queue full)")
            while not job.done.wait(timeout=0.25):
                if self.stop.is_set():
                    # The scoring loop is no longer running. Drop the connection
                    # and let the client reconnect / fail over. score() is a
                    # pure function, so a resend produces the same value
                    # (at-least-once is safe).
                    raise _PeerGone("server stopping (request in flight)")
            qwait = max(0.0, (time.perf_counter() - t_enq) - job.elapsed)
            self.stats.record(which, job.elapsed, job.status == ST_OK,
                              job.err, qwait)
            _sendall(sock, _pack_reply(job.req_id, job.status, job.value,
                                       job.elapsed, job.err))
            if job.status == ST_FATAL:
                # The CUDA context after an OOM cannot be trusted. The reply has
                # been sent. Let the supervisor restart us; that is safer than
                # continuing to return values from a broken process.
                # We cut other readers even mid-send: the clients caught in the
                # blast radius reconnect / fail over as a transport failure.
                print("[score-server] FATAL: exiting the process due to CUDA OOM "
                      "(the supervisor is expected to restart it)", flush=True)
                self.clear_ready()
                sys.stdout.flush()
                os._exit(17)

    def _validate(self, which, sr, n_ch, n_smp, dtype, layout, flags, nbytes
                  ) -> Tuple[int, str]:
        """(status, message). Valid iff status == ST_OK.

        The size check comes first: so that we never alloc while trusting a
        corrupt nbytes.
        """
        if nbytes > HARD_MAX_BYTES:
            return (ST_TOO_LARGE,
                    f"payload {nbytes} B > hard limit {HARD_MAX_BYTES} B")
        if which not in (WHICH_PQ, WHICH_SB):
            return (ST_BAD_REQUEST, f"invalid which: {which}")
        if dtype != DTYPE_F32:
            return (ST_BAD_REQUEST, f"only '<f4' dtype is allowed: got {dtype}")
        if layout not in (LAYOUT_CN, LAYOUT_N):
            return (ST_BAD_REQUEST, f"invalid layout: {layout}")
        if flags != 0:
            # Do not let a future addition such as window be silently ignored.
            return (ST_BAD_REQUEST, f"flags must be 0 (unsupported option): {flags}")
        if n_ch < 1 or n_smp < 1:
            return (ST_BAD_REQUEST, f"invalid shape: n_ch={n_ch} n_smp={n_smp}")
        if layout == LAYOUT_N and n_ch != 1:
            return (ST_BAD_REQUEST, f"layout=N but n_ch={n_ch}")
        if sr <= 0:
            return (ST_BAD_REQUEST, f"invalid sr: {sr}")
        expect = n_ch * n_smp * 4
        if nbytes != expect:
            return (ST_BAD_REQUEST,
                    f"nbytes mismatch: header={nbytes} expected={expect}")
        dur = n_smp / float(sr)
        if dur > self.max_sec:
            return (ST_TOO_LARGE,
                    f"audio length {dur:.1f}s > limit {self.max_sec:.1f}s. "
                    "Rejected without attempting to score (so an OOM does not take "
                    "the shared server down with it. Change with --max-sec).")
        return (ST_OK, "")

    # -- scoring loop (main thread. **only one exists**) --------------------
    def score_loop(self) -> None:
        next_stats = time.monotonic() + self.stats_sec
        while not self.stop.is_set():
            try:
                job = self.q.get(timeout=0.25)
            except queue.Empty:
                if time.monotonic() >= next_stats:
                    print(self.stats.render_and_reset_window(self.q.qsize()),
                          flush=True)
                    next_stats = time.monotonic() + self.stats_sec
                continue

            self._in_flight += 1
            # Explicit verification of serialization. On a violation, do not
            # silently continue.
            # Not 1 here = concurrent scoring is happening somewhere = a state
            # where batching or value wobble can occur. Die immediately.
            if self._in_flight != 1:
                print(f"[score-server] FATAL: in_flight={self._in_flight} "
                      "(scoring is not serialized)", flush=True)
                self.clear_ready()
                sys.stdout.flush()
                os._exit(18)
            try:
                self._run_one(job)
            finally:
                self._in_flight -= 1
                job.done.set()

            if time.monotonic() >= next_stats:
                print(self.stats.render_and_reset_window(self.q.qsize()),
                      flush=True)
                next_stats = time.monotonic() + self.stats_sec

    def _run_one(self, job: _Job) -> None:
        t0 = time.perf_counter()
        try:
            if job.which == WHICH_PQ:
                # Exactly the same arguments as today. window is not passed
                # (the current code does not pass it either).
                job.value = float(self.pq_scorer.score(job.audio, job.sr))
            else:
                # SB_CHUNK_SEC is applied per request.
                # Since in-flight is always 1, a race is impossible by
                # construction.
                os.environ["SB_CHUNK_SEC"] = repr(float(job.sb_chunk))
                job.value = float(self.sb_ear.score(job.audio, job.sr))
            job.status = ST_OK
        except Exception as ex:                                   # noqa: BLE001
            job.value = 0.0
            job.err = f"{type(ex).__name__}: {ex}"
            if _is_cuda_oom(ex):
                job.status = ST_FATAL
                print(f"[score-server] CUDA OOM on {WHICH_NAME[job.which]} "
                      f"from {job.peer}:\n{traceback.format_exc()}", flush=True)
            else:
                job.status = ST_SCORER_ERROR
                print(f"[score-server] scorer error on "
                      f"{WHICH_NAME[job.which]} from {job.peer}: {job.err}",
                      flush=True)
        finally:
            job.elapsed = time.perf_counter() - t0


def _is_cuda_oom(ex: BaseException) -> bool:
    if type(ex).__name__ == "OutOfMemoryError":
        return True
    s = str(ex)
    return ("CUDA out of memory" in s
            or "CUBLAS_STATUS_ALLOC_FAILED" in s
            or "out of memory" in s.lower() and "cuda" in s.lower())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="shared scoring server (Audiobox-PQ + SongBench-Mixing)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to bind (default 127.0.0.1 = loopback only)")
    ap.add_argument("--port", type=int, default=None,
                    help=("port to bind (required; omittable only with --print-fingerprint). "
                          "On a bind failure it exits immediately instead of using another port"))
    ap.add_argument("--max-clients", type=int, default=8,
                    help="expected number of concurrent clients (only used as queue length = 2x)")
    ap.add_argument("--backlog", type=int, default=64)
    ap.add_argument("--max-sec", type=float, default=DEFAULT_MAX_SEC,
                    help=("audio longer than this returns ERROR without being scored. "
                          "Default 600s. Lushlife - Toynbee Suite (628.6s) needs "
                          "~47GiB for full-song SB scoring, which does not fit on an "
                          "A100-40GB, so the shared server rejects it"))
    ap.add_argument("--warmup-sec", type=float, default=12.0,
                    help="length of the dummy audio used for warm-up (match the production excerpt length)")
    ap.add_argument("--stats-sec", type=float, default=30.0,
                    help="interval in seconds between statistics prints")
    ap.add_argument("--ready-dir", default=str(ROOT / "outputs" / "score_server"))
    ap.add_argument("--strict", action="store_true",
                    help=("never accept AUTOMIX_SCORER_FP_RELAX from the client. "
                          "Pass it on the reported runs that claim bit-identity"))
    ap.add_argument("--print-fingerprint", action="store_true",
                    help="print only the fingerprint and exit (does not load the models)")
    args = ap.parse_args(argv)

    if args.print_fingerprint:
        # `_common` has not been imported (see the argv check at the top of the
        # file). Used to cross-check against the output of
        # `python3 tools/scoring_client.py` on the client side.
        print(json.dumps(local_fingerprint(), indent=2, sort_keys=True))
        return 0
    if args.port is None:
        ap.error("--port is required")

    if _SKIP_COMMON:
        print("[score-server] WARNING: AUTOMIX_SCORER_SKIP_COMMON=1, so "
              "_common was not imported. Do not set it in production "
              "(cffi conflicts inside the container).", flush=True)
    print(f"[score-server] proto={proto_hash()} "
          f"req_hdr={REQ_SIZE}B rep_hdr={REP_SIZE}B pid={os.getpid()}",
          flush=True)
    print(f"[score-server] fingerprint: "
          f"{json.dumps(local_fingerprint(), sort_keys=True)}", flush=True)

    srv = ScoringServer(host=args.host, port=args.port, max_sec=args.max_sec,
                        stats_sec=args.stats_sec, backlog=args.backlog,
                        max_clients=args.max_clients,
                        ready_dir=_Path(args.ready_dir),
                        strict_fp=args.strict)

    def _on_signal(signum, _frame):
        print(f"[score-server] received signal {signum}. Stopping.", flush=True)
        srv.stop.set()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    t_load = time.perf_counter()
    srv.build_scorers()
    print(f"[score-server] scorers built in {time.perf_counter()-t_load:.1f}s",
          flush=True)

    # Finish warm-up **before listen()** (prevents a thundering herd).
    srv.warmup(args.warmup_sec)

    srv.bind_listen()
    acc = threading.Thread(target=srv.acceptor, name="acceptor", daemon=True)
    acc.start()
    srv.write_ready()

    try:
        srv.score_loop()                 # main thread = scoring loop
    finally:
        srv.shutdown()
        print(srv.stats.render_and_reset_window(srv.q.qsize()), flush=True)
        print("[score-server] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

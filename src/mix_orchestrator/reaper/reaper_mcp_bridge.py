"""Phase-2 stub for talking to bonfire-audio/reaper-mcp (58 tools).

The bridge speaks JSON over MCP and provides a thin sync wrapper:
    bridge = ReaperMCPBridge()
    bridge.connect()
    bridge.add_track("vocal")
    bridge.add_fx(track_idx=0, fx_name="ReaEQ", params={...})
    audio = bridge.render_master(start_sec=0, end_sec=180, sr=44100)
    bridge.close()

This file is intentionally not wired into the orchestrator yet — Phase 1
results are produced with pedalboard. Phase 2 will validate
pedalboard↔REAPER agreement on a small subset before swapping back-ends.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ReaperMCPConfig:
    server_cmd: List[str] = None     # e.g. ["node", "/path/to/reaper-mcp/dist/index.js"]
    reaper_app: Optional[str] = None # path to REAPER binary
    timeout_sec: float = 30.0


class ReaperMCPBridge:
    """Process-based REAPER MCP bridge (stub).

    The real implementation will launch `reaper-mcp` as a subprocess and
    speak JSON-RPC over stdio. For Phase 1 we only stub the interface so
    the renderer file can compile and so unit tests can verify the
    public surface.
    """

    def __init__(self, config: Optional[ReaperMCPConfig] = None):
        self.config = config or ReaperMCPConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._connected = False

    # ---------- lifecycle ----------

    def connect(self) -> None:
        cmd = self.config.server_cmd
        if not cmd:
            raise RuntimeError(
                "ReaperMCPBridge.connect: config.server_cmd not set. "
                "Provide e.g. ['node', '<reaper-mcp>/dist/index.js'] or set "
                "REAPER_MCP_CMD")
        if shutil.which(cmd[0]) is None:
            raise FileNotFoundError(f"server binary not in PATH: {cmd[0]}")
        # PHASE-2: actually spawn the subprocess and handshake.
        self._proc = None
        self._connected = True

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._connected = False

    # ---------- session ----------

    def reset_session(self) -> None:
        self._call("reset_session", {})

    def add_track(self, name: str) -> int:
        return int(self._call("add_track", {"name": name})["track_idx"])

    def import_audio(self, track_idx: int, wav_path: str,
                     start_sec: float = 0.0) -> None:
        self._call("import_audio", {"track_idx": track_idx,
                                    "path": wav_path,
                                    "start_sec": start_sec})

    def set_track_volume(self, track_idx: int, gain_db: float) -> None:
        self._call("set_track_volume", {"track_idx": track_idx, "gain_db": gain_db})

    def set_track_pan(self, track_idx: int, pan: float) -> None:
        self._call("set_track_pan", {"track_idx": track_idx, "pan": pan})

    def add_fx(self, track_idx: int, fx_name: str, params: Dict[str, Any]) -> int:
        return int(self._call("add_fx", {
            "track_idx": track_idx, "fx_name": fx_name, "params": params,
        })["fx_idx"])

    def render_master(self, start_sec: float, end_sec: float,
                      sr: int = 44100, out_wav: Optional[str] = None
                      ) -> np.ndarray:
        out = out_wav or "/tmp/reaper_render.wav"
        self._call("render_master", {
            "start_sec": start_sec, "end_sec": end_sec, "sample_rate": sr,
            "path": out,
        })
        import soundfile as sf
        audio, _sr = sf.read(out, dtype="float32", always_2d=True)
        return audio.T

    # ---------- transport ----------

    def _call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._connected:
            raise RuntimeError("ReaperMCPBridge.connect() must be called first")
        if self._proc is None:
            # Phase-1 stub: simulate a no-op response so callers can be exercised
            return {"track_idx": 0, "fx_idx": 0, "status": "stub"}
        # Phase-2: write JSON-RPC line, read response with timeout
        req = json.dumps({"method": method, "params": params}) + "\n"
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(req)
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        return json.loads(line)

"""Infrastructure management — GPU stats, llama-server lifecycle, node health."""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_LLAMA_SERVER_BIN = "/programs/llama.cpp/build/bin/llama-server"


@dataclass
class GpuStats:
    name: str
    vram_used_mb: int
    vram_total_mb: int
    utilization_pct: int
    temperature_c: int
    processes: list[dict]  # [{pid, name, vram_mb}]


@dataclass
class ServerInfo:
    pid: int
    node: str
    port: int
    model_path: str
    status: str  # "healthy", "unreachable", "unknown"


def get_gpu_stats(node: str = "local") -> GpuStats | None:
    """Parse nvidia-smi output for GPU stats. Returns None on failure."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc_cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]

    try:
        if node == "local":
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            proc_result = subprocess.run(proc_cmd, capture_output=True, text=True, timeout=10)
        else:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", node, " ".join(cmd)],
                capture_output=True, text=True, timeout=15,
            )
            proc_result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", node, " ".join(proc_cmd)],
                capture_output=True, text=True, timeout=15,
            )

        if result.returncode != 0:
            return None

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 5:
            return None

        processes = []
        for line in proc_result.stdout.strip().splitlines():
            pparts = [p.strip() for p in line.split(",")]
            if len(pparts) >= 3:
                processes.append({
                    "pid": int(pparts[0]),
                    "name": pparts[1],
                    "vram_mb": int(pparts[2]),
                })

        return GpuStats(
            name=parts[0],
            vram_used_mb=int(parts[1]),
            vram_total_mb=int(parts[2]),
            utilization_pct=int(parts[3]),
            temperature_c=int(parts[4]),
            processes=processes,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def get_running_servers(node: str = "local", remote_host: str | None = None) -> list[ServerInfo]:
    """Find llama-server processes via ps. Returns list of ServerInfo."""
    ssh_host = remote_host or node
    try:
        if node == "local":
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=10,
            )
        else:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", ssh_host, "ps aux"],
                capture_output=True, text=True, timeout=15,
            )

        if result.returncode != 0:
            return []

        servers = []
        for line in result.stdout.splitlines():
            if "llama-server" not in line or "grep" in line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            pid = int(parts[1])
            cmdline = " ".join(parts[10:])

            port = _extract_flag(cmdline, "--port", default="8080")
            model_path = _extract_flag(cmdline, "-m") or _extract_flag(cmdline, "--model") or "unknown"

            node_label = "node1" if node == "local" else node
            servers.append(ServerInfo(
                pid=pid,
                node=node_label,
                port=int(port),
                model_path=model_path,
                status="unknown",
            ))

        return servers
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return []


def start_server(
    node: str,
    model_path: str,
    port: int = 8080,
    ctx_size: int = 8192,
    flash_attn: bool = True,
    threads: int | None = None,
    parallel: int = 1,
    gpu_layers: int = 999,
    llama_server_bin: str = "",
    remote_host: str = "",
) -> ServerInfo | None:
    """Launch llama-server locally or via SSH. Returns ServerInfo on success."""
    server_bin = llama_server_bin or _DEFAULT_LLAMA_SERVER_BIN
    args = [
        server_bin,
        "-m", model_path,
        "--port", str(port),
        "-c", str(ctx_size),
        "-ngl", str(gpu_layers),
        "--parallel", str(parallel),
    ]
    args.extend(["--flash-attn", "on" if flash_attn else "off"])
    if threads:
        args.extend(["-t", str(threads)])

    try:
        if node in ("local", "node1"):
            proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            # Grace period — check if process survives startup
            import time
            time.sleep(1)
            exit_code = proc.poll()
            if exit_code is not None:
                # Process already died — read stderr for clues
                err = ""
                try:
                    err = proc.stderr.read().decode(errors="replace").strip()[-200:]
                except Exception:
                    pass
                return ServerInfo(
                    pid=proc.pid,
                    node="node1",
                    port=port,
                    model_path=model_path,
                    status=f"crashed: {err}" if err else "crashed",
                )
            # Detach stderr so the fd doesn't leak
            proc.stderr.close()
            pid = proc.pid
            node_label = "node1"
        else:
            # Launch on remote node via SSH
            ssh_target = remote_host or node
            remote_cmd = " ".join(args)
            ssh_cmd = [
                "ssh", "-o", "ConnectTimeout=5", ssh_target,
                f"nohup {remote_cmd} > /dev/null 2>&1 & echo $!"
            ]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return None
            pid = int(result.stdout.strip())
            node_label = node

        return ServerInfo(
            pid=pid,
            node=node_label,
            port=port,
            model_path=model_path,
            status="unknown",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
        return None


def stop_server(pid: int, node: str = "local", remote_host: str = "") -> bool:
    """Kill a llama-server process by PID. Returns True if signal was sent."""
    try:
        if node in ("local", "node1"):
            os.kill(pid, signal.SIGTERM)
        else:
            ssh_target = remote_host or node
            subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", ssh_target, f"kill {pid}"],
                capture_output=True, text=True, timeout=10,
            )
        return True
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        return False


def health_check(base_url: str) -> dict:
    """GET /health on a llama-server endpoint. Returns status dict."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/health", timeout=5)
        data = resp.json()
        return {"status": data.get("status", "ok"), "slots": data.get("slots_idle", "?")}
    except Exception:
        return {"status": "unreachable", "slots": 0}


def check_node_reachable(host: str) -> bool:
    """Quick SSH connectivity test."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", host, "echo ok"],
            capture_output=True, text=True, timeout=8,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _extract_flag(cmdline: str, flag: str, default: str | None = None) -> str | None:
    """Extract value after a flag in a command line string."""
    match = re.search(rf"{re.escape(flag)}\s+(\S+)", cmdline)
    return match.group(1) if match else default

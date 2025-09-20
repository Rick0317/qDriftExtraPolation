# utils_io.py
"""
Utilities for writing experiment metadata (JSON) and results (CSV).
- Metadata: human-readable JSON, written atomically for reproducibility.
- Results: CSV with fixed columns, safe init + append.
"""

from __future__ import annotations
import csv, datetime, json, os, pathlib, subprocess, tempfile, warnings
from dataclasses import asdict, is_dataclass
from typing import Iterable, Any


# ───────────────────────────────
# Git commit provenance
# ───────────────────────────────
def ensure_git_commit_env() -> str:
    """Ensure GIT_COMMIT env var is set. Try `git rev-parse HEAD`, else fallback."""
    current = os.getenv("GIT_COMMIT")
    if current:
        return current
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    os.environ["GIT_COMMIT"] = commit
    return commit


# ───────────────────────────────
# JSON helpers
# ───────────────────────────────
def write_json_atomic(path: pathlib.Path, data: Any, indent: int = 2) -> None:
    """Write JSON atomically with pretty formatting for readability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), prefix=path.name, suffix=".tmp", encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=indent, sort_keys=True)
        tmp.write("\n")  # final newline for POSIX friendliness
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def write_metadata(
    path: pathlib.Path,
    cfgs: Iterable[dict],
    verbose: bool = False,
    num_system_qubits: int | None = None,
    replication_seeds: list[int] | None = None,
    times: list[float] | None = None,
    num_ancilla: list[int] | None = None,
    num_qdrift_segments: list[int] | None = None,
    circuits_per_datapoint: list[int] | None = None,
    shots_per_circuit: list[int] | None = None,
    hamiltonians: dict[str, Any] | None = None,
    estimate_ground_state: list[bool] | None = None,
    test_id: str | None = None,
) -> None:
    """
    Write experiment metadata JSON (human-readable, reproducible).

    - verbose=True: dump the full config grid (all configs).
    - Otherwise: record static parameters and sweep space explicitly.
    - Fields mirror the original driver code for reproducibility.
    """
    git_commit = ensure_git_commit_env()
    utc = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if verbose:
        payload = {
            "generated_utc": utc,
            "git_commit": git_commit,
            "num_configs": len(list(cfgs)),
            "configs": list(cfgs),
            "test_id": test_id,
        }
    else:
        ham_list = []
        if hamiltonians:
            for ty, H in hamiltonians.items():
                ham_list.append({
                    "type": ty,
                    "coeffs": str(H.coeffs),
                    "paulis": H.paulis.to_labels(),
                    "eigenvalue to estimate": (
                        getattr(__import__("numpy").linalg, "eigvals")(H.to_matrix()).real.tolist()
                    )
                })

        payload = {
            "generated_utc": utc,
            "git_commit": git_commit,
            "test_id": test_id,
            "metadata": {
                "num_configs": len(list(cfgs)),
                "static_parameters": {
                    "num_system_qubits": num_system_qubits,
                    "Random_seed(s)": replication_seeds,
                },
                "sweep_space": {
                    "t": list(times or []),
                    "num_ancilla": num_ancilla or [],
                    "num_qdrift_segments_per_qdrift_channel_invocation": num_qdrift_segments or [],
                    "num_independent_stochastic_circuits_per_datapoint": circuits_per_datapoint or [],
                    "num_shots_per_circuit": shots_per_circuit or [],
                    "Hamiltonians": ham_list,
                    "calculate_ground_state": estimate_ground_state or [],
                },
            },
        }
    write_json_atomic(path, payload, indent=2)


# ───────────────────────────────
# CSV helpers
# ───────────────────────────────
def init_csv(path: pathlib.Path, fieldnames: list[str]) -> pathlib.Path:
    """Ensure CSV exists and has header. Return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with path.open("x", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
        except FileExistsError:
            pass
    return path


def append_csv(path: pathlib.Path, fieldnames: list[str], row: dict | Any) -> None:
    """Append a row (dict or dataclass) to CSV."""
    if is_dataclass(row):
        row = asdict(row)
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)

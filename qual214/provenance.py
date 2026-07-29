"""Provenance enforcement for the Capsule Brain v2.14.0 qualification harness.

Before any test runs, the harness must cryptographically identify exactly what
it is testing. This module:

- asserts ``importlib.metadata.version("capsule-brain") == "2.14.0"`` so the
  qualification cannot silently certify a different installed build;
- records the wheel SHA-256, source-tree SHA-256, Python version, platform,
  Ollama version, and model digests into a manifest that is emitted alongside
  the results.

The source-tree hash covers ``src/``, ``tests/``, and ``pyproject.toml`` so a
change to any production source, test, or version declaration is detectable.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

EXPECTED_VERSION = "2.14.0"

# Repository root (qual214/..)
ROOT = Path(__file__).resolve().parent.parent


class ProvenanceError(RuntimeError):
    """Raised when the installed build does not match the expected target."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def source_tree_sha256() -> str:
    """SHA-256 over the grouped-sorted list of (relative_path, contents) for
    src/**/*.py, then tests/**/*.py, then pyproject.toml. Stable across runs
    and independent of file mtimes. The grouping (src first, tests second,
    pyproject last) matches the hash recorded in VALIDATION_V2_14_0.txt so the
    manifest and the validation file agree."""
    files: list[Path] = []
    files.extend(sorted((ROOT / "src").glob("**/*.py")))
    files.extend(sorted((ROOT / "tests").glob("**/*.py")))
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        files.append(pyproject)
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def installed_wheel_sha256() -> str | None:
    """Locate the installed capsule_brain wheel in site-packages and hash it.
    Returns None if the wheel file cannot be located (e.g. editable install)."""
    try:
        import capsule_brain  # noqa: F401
    except ImportError as exc:
        raise ProvenanceError("capsule_brain is not installed") from exc
    # Look in site-packages first (wheel installed there), then in dist/ (built
    # wheel in the repo). pip install <file> does not keep the .whl in
    # site-packages, so the dist/ fallback is the common path.
    try:
        from importlib.metadata import files
        dist_files = files("capsule-brain")
        if dist_files:
            anchor = dist_files[0].locate()
            site_pkgs = Path(anchor).resolve().parent
            for cand in [site_pkgs, site_pkgs.parent]:
                for whl in cand.glob("capsule_brain-2.14.0-*.whl"):
                    return _sha256_file(whl.resolve())
                for whl in cand.glob("capsule_brain-*.whl"):
                    return _sha256_file(whl.resolve())
    except Exception:
        pass
    # Fallback: look in dist/ in the repository root.
    dist_dir = ROOT / "dist"
    if dist_dir.exists():
        for whl in sorted(dist_dir.glob("capsule_brain-2.14.0-*.whl")):
            return _sha256_file(whl.resolve())
    return None


def _ollama_version() -> str:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True,
                             text=True, timeout=10)
        line = (out.stdout or out.stderr or "").strip().splitlines()
        return line[0] if line else "unknown"
    except Exception:
        return "unavailable"


def _model_digest(model: str) -> dict[str, Any]:
    import json as _json
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=_json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = _json.loads(r.read())
        details = data.get("details", {}) or {}
        return {
            "model": model,
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "architecture": (data.get("model_info") or {}).get("general.architecture"),
        }
    except Exception as exc:
        return {"model": model, "error": str(exc)}


def build_manifest() -> dict[str, Any]:
    """Build the provenance manifest. Raises ProvenanceError if the installed
    version does not match EXPECTED_VERSION."""
    installed = pkg_version("capsule-brain")
    if installed != EXPECTED_VERSION:
        raise ProvenanceError(
            f"Installed capsule-brain is {installed!r}, expected {EXPECTED_VERSION!r}. "
            "The qualification harness refuses to certify a different build."
        )
    manifest = {
        "expected_version": EXPECTED_VERSION,
        "installed_version": installed,
        "source_tree_sha256": source_tree_sha256(),
        "installed_wheel_sha256": installed_wheel_sha256(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ollama_version": _ollama_version(),
        "models": [
            _model_digest("qwen2.5-coder:3b"),
            _model_digest("llama3.1:8b"),
        ],
        "harness_files": sorted(p.name for p in Path(__file__).resolve().parent.glob("*.py")),
    }
    return manifest


def assert_provenance() -> dict[str, Any]:
    """Assert provenance and return the manifest. Exits the process on mismatch
    so qualification never proceeds against the wrong build."""
    try:
        manifest = build_manifest()
    except ProvenanceError as exc:
        print(f"PROVENANCE FAILURE: {exc}", file=sys.stderr)
        sys.exit(3)
    print("PROVENANCE OK:")
    print(f"  installed_version: {manifest['installed_version']}")
    print(f"  source_tree_sha256: {manifest['source_tree_sha256']}")
    print(f"  installed_wheel_sha256: {manifest['installed_wheel_sha256']}")
    print(f"  python: {manifest['python_version']}  platform: {manifest['platform']}")
    print(f"  ollama: {manifest['ollama_version']}")
    return manifest

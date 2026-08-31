import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_manifest = None


def app_root() -> Path:
    """
    Directory that owns the running application.
    - Frozen (PyInstaller COLLECT / onedir): folder containing the executable.
    - Frozen (one-file): still the folder containing the executable (user-facing
      assets live next to it, not inside the ephemeral _MEIPASS).
    - Source: directory of this config.py (project root).
    """
    if getattr(sys, "frozen", False):
        # Fallback: look in the same directory as the frozen executable
        return Path(sys.executable).parent

    # Look for it relative to the script location (works after PyInstaller in some configs)
    return Path(os.path.dirname(os.path.abspath(__file__)))


def _candidate_data_dirs(preferred: Optional[str] = None) -> List[Path]:
    """Ordered list of places to look for the data tree."""
    candidates = []

    if preferred:
        candidates.append(Path(preferred).resolve())

    root = app_root()
    # 1. <app_root>/_internal/data (PyInstaller onedir internal folder)
    candidates.append(root / "_internal" / "data")
    # 2. <app_root>/data
    candidates.append(root / "data")
    # 3. ./data (current working directory)
    candidates.append(Path.cwd() / "data")

    return candidates


def find_data_dir(
    preferred: Optional[str] = None,
    *,
    create_if_missing: bool = False,
) -> Path:
    """
    Locate the application data directory.
    Call after load_manifest() so any path hints in the manifest can be used.
    Search order:
      1. <app_root>/_internal/data
      2. <app_root>/data
      3. ./data (cwd)
    Returns an absolute Path. Raises FileNotFoundError only when
    create_if_missing is False and nothing usable is found.
    """
    candidates = _candidate_data_dirs(preferred)

    # First pass: look for an existing directory
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    # If we get here, no existing directory was found.
    if create_if_missing:
        # Pick the most logical place to create it: preferred, or <app_root>/data
        target = Path(preferred).resolve() if preferred else (app_root() / "data")
        try:
            target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError as e:
            raise RuntimeError(f"Failed to create data directory at {target}: {e}")

    raise FileNotFoundError(
        f"Could not find data directory. Searched: {[str(c) for c in candidates]}"
    )


def load_manifest(manifest_path: str = None) -> dict:
    """Load the resource manifest. Call this once at startup."""
    global _manifest
    if _manifest is not None:
        return _manifest

    if manifest_path is None:
        # Unified logic: use app_root() to get the correct base directory
        base_dir = app_root()
        manifest_path = base_dir / "url_manifest.json"

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            _manifest = json.load(f)
        print(f"Manifest loaded successfully from: {manifest_path}")
    except Exception as e:
        print(f"Failed to load manifest: {e}")
        _manifest = {}

    return _manifest


def get_manifest() -> dict:
    """Get the manifest dictionary (call after load_manifest)."""
    global _manifest
    if _manifest is None:
        load_manifest()  # auto-load if not loaded yet
    return _manifest


def get_base_url(key: str) -> str:
    """Helper to get base URLs easily."""
    manifest = get_manifest()
    return manifest.get("base_urls", {}).get(key, "")

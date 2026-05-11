import json
import os
from pathlib import Path
import sys

_manifest = None


def load_manifest(manifest_path: str = None) -> dict:
    """Load the resource manifest. Call this once at startup."""
    global _manifest

    if _manifest is not None:
        return _manifest

    if manifest_path is None:
        # Look for it relative to the executable or script location (works after PyInstaller)
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        manifest_path = base_dir / "url_manifest.json"

        # Fallback: look in the same directory as the frozen executable
        if not manifest_path.exists() and getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).parent
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

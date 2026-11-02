"""Optional, validated paths for native-reference and myocardium evaluation data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ReferencePaths:
    native_reference_root: Path | None
    myocardium_mask_root: Path | None
    warnings: tuple[str, ...]

    @property
    def skip_metrics(self) -> bool:
        return self.native_reference_root is None or self.myocardium_mask_root is None


def _usable_native_reference(path: Path | None) -> tuple[Path | None, str | None]:
    if path is None:
        return None, "Native reference evaluation skipped: native_reference_root is not configured."
    if not (path / "native_reference_manifest.json").is_file():
        return None, f"Native reference evaluation skipped: missing {path / 'native_reference_manifest.json'}."
    return path, None


def _usable_myocardium_bundle(path: Path | None) -> tuple[Path | None, str | None]:
    if path is None:
        return None, "Myocardium evaluation skipped: myocardium_mask_root is not configured."
    required = (path / "manifest.json", path / "sax_myocardium_masks.npz")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        return None, "Myocardium evaluation skipped: missing " + ", ".join(missing) + "."
    return path, None


def load_reference_paths(config_path: str | Path, *, native_reference_root: str | Path | None = None, myocardium_mask_root: str | Path | None = None) -> ReferencePaths:
    """Load optional roots, returning diagnostics instead of fabricating metrics."""

    config = Path(config_path)
    if not config.is_file():
        raise FileNotFoundError(f"Reference-path configuration does not exist: {config}")
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    section = payload.get("reference_data", {})
    if not isinstance(section, dict):
        raise ValueError("reference_paths.yaml reference_data must be a mapping.")
    native_value = native_reference_root if native_reference_root is not None else section.get("native_reference_root")
    mask_value = myocardium_mask_root if myocardium_mask_root is not None else section.get("myocardium_mask_root")
    native, native_warning = _usable_native_reference(Path(native_value).expanduser() if native_value else None)
    masks, mask_warning = _usable_myocardium_bundle(Path(mask_value).expanduser() if mask_value else None)
    return ReferencePaths(native, masks, tuple(item for item in (native_warning, mask_warning) if item is not None))

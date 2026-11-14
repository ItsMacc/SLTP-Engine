# -------- config.py --------
# One small JSON file for machine-level choices: where bars come from and
# whether the R2 cache refreshes itself. Defaults keep a fresh install on the
# managed cache with auto-update on; `sltp reset` deletes the file.
# Every read tolerates a missing or broken file, so the browser (which never
# calls the CLI) and read-only machines always get sane defaults.

# -------- Imports --------
import json
import os

from pathlib import Path


# -------- Constants --------
# The override wins over this file when set, so CI keeps working.
ENV_VAR: str = "SLTP_DATA_DIR"
CONFIG_PATH: Path = Path.home() / ".sltp" / "config.json"
DEFAULTS: dict = {"auto_update": True, "data_dir": None}


# -------- Read & Write --------
# The stored choices merged over the defaults; broken file, same defaults.
def _read() -> dict:
    merged: dict = dict(DEFAULTS)
    try:
        stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return merged
    if isinstance(stored, dict):
        for key in DEFAULTS:
            if key in stored:
                merged[key] = stored[key]
    return merged


def _write(values: dict) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(values, indent=2), encoding="utf-8")
    except OSError:
        pass


# -------- Choices --------
# True unless the user turned refresh off; fresh installs update every run.
def auto_update() -> bool:
    return bool(_read().get("auto_update", True))


def set_auto_update(enabled: bool) -> None:
    values: dict = _read()
    values["auto_update"] = bool(enabled)
    _write(values)


# An explicit bar directory from `sltp data use`, or None for the default hunt.
def data_override() -> Path | None:
    raw = os.environ.get(ENV_VAR) or _read().get("data_dir")
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def set_data_override(path: str | None) -> None:
    values: dict = _read()
    values["data_dir"] = path
    _write(values)


# Forget every choice; the next read behaves like a fresh install.
def reset() -> None:
    try:
        CONFIG_PATH.unlink(missing_ok=True)
    except OSError:
        pass

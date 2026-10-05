"""
Lecture de la configuration (équivalent de rag.config.settings + rag.utils.config_utils).

La configuration est un fichier .ini chargé une fois par processus puis lu
via les helpers get_str / get_int / get_float / get_bool.
"""
import configparser
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "eval.local.ini"

_CONFIG: configparser.ConfigParser | None = None


def load_config(path: str | Path = DEFAULT_CONFIG) -> configparser.ConfigParser:
    """Charge le fichier .ini et le garde en mémoire pour le processus courant."""
    global _CONFIG
    cfg = configparser.ConfigParser()
    read = cfg.read(path, encoding="utf-8")
    if not read:
        raise FileNotFoundError(f"Configuration introuvable : {path}")
    _CONFIG = cfg
    return cfg


def get_config(section: str, key: str) -> str | None:
    if _CONFIG is None:
        load_config()
    assert _CONFIG is not None
    if not _CONFIG.has_section(section):
        return None
    return _CONFIG[section].get(key)


def get_str(section: str, key: str, default: str) -> str:
    """Lit une chaîne depuis la config, sinon `default`."""
    val = get_config(section, key)
    return val.strip() if val not in (None, "") else default


def get_int(section: str, key: str, default: int) -> int:
    """Lit un int depuis la config, sinon `default`."""
    val = get_config(section, key)
    try:
        return int(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def get_float(section: str, key: str, default: float) -> float:
    """Lit un float depuis la config, sinon `default`."""
    val = get_config(section, key)
    try:
        return float(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def get_bool(section: str, key: str, default: bool) -> bool:
    """Lit un booléen depuis la config, sinon `default`."""
    val = get_config(section, key)
    if val in (None, ""):
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

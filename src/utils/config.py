"""Configuration utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable."""
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get a boolean environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get an integer environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge multiple config dicts (later overrides earlier)."""
    result = {}
    for config in configs:
        _deep_update(result, config)
    return result


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Deep update one dict with another."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


class Config:
    """Centralized configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config: Dict[str, Any] = {}
        if config_path:
            self.config = load_yaml_config(config_path)
        self._load_env_overrides()

    def _load_env_overrides(self) -> None:
        """Load environment variable overrides."""
        env_mappings = {
            "BROWSER_HEADLESS": ("browser", "headless"),
            "BROWSER_VIEWPORT_WIDTH": ("browser", "viewport", "width"),
            "BROWSER_VIEWPORT_HEIGHT": ("browser", "viewport", "height"),
            "BROWSER_TIMEOUT_MS": ("browser", "timeout_ms"),
            "MODEL_NAME": ("model", "name"),
            "MODEL_DEVICE": ("model", "device"),
            "LOG_LEVEL": ("logging", "level"),
        }
        for env_key, path in env_mappings.items():
            val = os.getenv(env_key)
            if val is not None:
                self._set_nested(self.config, path, val)

    def _set_nested(self, d: Dict[str, Any], path: tuple, value: Any) -> None:
        """Set a nested value in a dict."""
        for key in path[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation."""
        keys = key.split(".")
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def to_dict(self) -> Dict[str, Any]:
        """Return the config as a dict."""
        return self.config

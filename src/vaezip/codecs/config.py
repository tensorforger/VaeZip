"""Resolving codec configs from dicts, files and shortcut names."""

import copy
import json
import os
from typing import Any

from vaezip.codecs.shortcuts import (
    CODEC_CLASS_KEY,
    SHORTCUT_CONFIGS,
    VERSION_KEY,
    list_shortcuts,
)

ConfigSource = str | os.PathLike[str] | dict[str, Any]


def load_config(source: ConfigSource) -> dict[str, Any]:
    """Config dict from a config dict, a shortcut name or a path to a JSON file."""
    if isinstance(source, dict):
        return copy.deepcopy(source)
    if isinstance(source, str) and source in SHORTCUT_CONFIGS:
        return copy.deepcopy(SHORTCUT_CONFIGS[source])
    path = os.fspath(source)
    if not os.path.isfile(path):
        raise ValueError(
            f"{path!r} is neither a codec config file nor a shortcut name, "
            f"available shortcuts: {list_shortcuts()}"
        )
    with open(path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Codec config {path!r} must contain a JSON object")
    return config


def split_config(
    source: ConfigSource, overrides: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    """
    returns:
        class name from the config (if present), constructor parameters with the
        overrides applied
    """
    config = load_config(source)
    class_name = config.pop(CODEC_CLASS_KEY, None)
    parameters = {
        name: value for name, value in config.items() if not name.startswith("_")
    }
    parameters.update(overrides)
    return class_name, parameters


def without_version(config: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in config.items() if name != VERSION_KEY}

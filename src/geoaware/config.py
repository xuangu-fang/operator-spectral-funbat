#!/usr/bin/env python3
"""Experiment configuration, so a new run is a YAML file rather than an edit.

Every setting that a study might vary lives in a config: the field, the nominal
coefficients the prior is told, the model's budgets, the optimiser, and the
evaluation protocol.  Configs compose -- a variant names its parent under
``inherits`` and overrides only what differs -- so a new operator family or a
new room is a dozen lines rather than a copy of the whole file.

Three properties are deliberate.

Overrides are explicit.  ``--set field.reaction=0.004`` changes one leaf and
records itself in the run's summary, so a result can always be traced to the
exact configuration that produced it.

Unknown keys are an error, not a silent no-op.  A typo in a hyper-parameter name
is otherwise indistinguishable from a null result, and this project has already
lost time to a setting that was never read.

Nothing is mutated in place.  The scripts this replaces reached into another
module and rewrote its globals to switch operator families, restoring them
afterwards; that is unsafe the moment two studies share a process, and it hides
what a run actually used.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    """Override leaves of ``base`` with ``override``, refusing unknown keys."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        where = f"{path}.{key}" if path else key
        if key not in result:
            raise KeyError(
                f"unknown configuration key '{where}'.  Known keys here: "
                f"{sorted(result)}")
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _deep_merge(result[key], value, where)
        else:
            result[key] = value
    return result


def _coerce(text: str) -> Any:
    """Turn a command-line value into the type it looks like."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.startswith("[") or text.startswith("("):
        return yaml.safe_load(text.replace("(", "[").replace(")", "]"))
    return text


@dataclass
class Config:
    """A resolved configuration, with the provenance needed to reproduce it."""

    data: dict
    name: str
    overrides: list[str] = dataclass_field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a nested value by dotted path: ``config.get('model.ranks')``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"configuration has no '{dotted}'")
        return value

    def field_kwargs(self) -> dict:
        """The solver's keyword arguments, with tuples where the solver wants them."""
        settings = copy.deepcopy(self.require("field"))
        for key in ("grid", "diffusivity", "drift"):
            if key in settings and settings[key] is not None:
                settings[key] = tuple(settings[key])
        if "sources" in settings and settings["sources"] is not None:
            settings["sources"] = tuple(tuple(s) for s in settings["sources"])
        settings.pop("solver", None)
        return settings

    def nominal(self) -> dict:
        """What the prior is told, which is deliberately not the truth."""
        told = copy.deepcopy(self.require("nominal"))
        for key in ("diffusivity", "drift"):
            if key in told and told[key] is not None:
                told[key] = tuple(told[key])
        return told

    def as_record(self) -> dict:
        """What to write into a summary so the run can be reproduced exactly."""
        return {"config_name": self.name, "overrides": list(self.overrides),
                "resolved": copy.deepcopy(self.data)}


def load_config(name: str | Path = "base", *, overrides: list[str] | None = None,
                seen: tuple[str, ...] = ()) -> Config:
    """Load ``configs/<name>.yaml``, resolving ``inherits`` and applying overrides.

    ``overrides`` are ``dotted.key=value`` strings from the command line.
    """
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{path}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no config at {path}; available: {available}")
    if path.stem in seen:
        raise ValueError(f"configuration inheritance loops through '{path.stem}'")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent_name = raw.pop("inherits", None)
    if parent_name is not None:
        parent = load_config(parent_name, seen=seen + (path.stem,))
        data = _deep_merge(parent.data, raw)
    else:
        data = raw

    applied: list[str] = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not of the form key=value")
        dotted, _, text = item.partition("=")
        node = data
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"override '{dotted}' has no section '{part}'")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"override '{dotted}' names a key that does not exist; "
                           f"known keys there: {sorted(node)}")
        node[parts[-1]] = _coerce(text)
        applied.append(item)

    return Config(data=data, name=path.stem, overrides=applied)


def add_config_arguments(parser) -> None:
    """Give a script ``--config`` and repeatable ``--set key=value``."""
    parser.add_argument("--config", default="base",
                        help="name under configs/ (or a path to a YAML file)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override one leaf, e.g. --set field.reaction=0.004; "
                             "repeatable, and recorded in the run's summary")

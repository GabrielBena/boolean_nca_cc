"""
Back-compat helpers for reading the fixed task name out of a Hydra config.

Configs were refactored to decouple circuit topology (``cfg.circuit``) from
task identity (``cfg.tasks``). Old checkpointed configs still carry the task
name at ``cfg.circuit.task``; this module bridges the two so downstream
tools (demos, exporters, oracle scripts) work uniformly with either schema.
"""

from typing import Any


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Resolve ``obj.key`` / ``obj[key]`` / ``obj.get(key)`` uniformly across
    OmegaConf DictConfig, dataclasses, and plain dicts."""
    if obj is None:
        return default
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def get_fixed_task_name(cfg: Any) -> str | None:
    """Resolve the fixed-library task name from either config schema.

    New schema: ``cfg.tasks.name`` (when ``cfg.tasks.type == "fixed"``).
    Old schema: ``cfg.circuit.task``.

    Returns None if neither is set — e.g., sampler-mode configs, where no
    single fixed task exists (callers must handle this explicitly).
    """
    tasks_cfg = _safe_get(cfg, "tasks")
    if tasks_cfg is not None and _safe_get(tasks_cfg, "type") == "fixed":
        return _safe_get(tasks_cfg, "name")
    circuit_cfg = _safe_get(cfg, "circuit")
    return _safe_get(circuit_cfg, "task")


def get_task_text(cfg: Any) -> str | None:
    """Resolve the per-task ``text`` field (used by the ``text`` library task)
    from either schema. Returns None if not set in either location."""
    tasks_cfg = _safe_get(cfg, "tasks")
    if tasks_cfg is not None and _safe_get(tasks_cfg, "type") == "fixed":
        text = _safe_get(tasks_cfg, "text")
        if text is not None:
            return text
    circuit_cfg = _safe_get(cfg, "circuit")
    return _safe_get(circuit_cfg, "text")

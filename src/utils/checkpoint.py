from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("checkpoint")

PHASE_FILES = {
    "1": "phase_1.json",
    "2": "phase_2.json",
    "3": "phase_3.json",
    "4": "phase_4.json",
    "5": "phase_5.json",
    "6": "phase_6.json",
    "7": "phase_7.json",
    "link_discovery": "phase_1.json",
    "download": "phase_2.json",
    "extraction": "phase_3.json",
    "normalization": "phase_4.json",
    "validation": "phase_5.json",
    "db_ingestion": "phase_6.json",
    "ml_prediction": "phase_7.json",
}


class CheckpointManager:
    def __init__(self, checkpoints_dir: str | Path) -> None:
        self._dir = Path(checkpoints_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _phase_path(self, phase: str) -> Path:
        filename = PHASE_FILES.get(phase, f"{phase}.json")
        return self._dir / filename

    def save_checkpoint(self, phase: str, data: dict[str, Any]) -> None:
        path = self._phase_path(phase)
        payload = {
            "phase": phase,
            "updated_at": datetime.utcnow().isoformat(),
            "data": data,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.debug("Checkpoint saved for phase '%s' at %s", phase, path)

    def load_checkpoint(self, phase: str) -> Optional[dict[str, Any]]:
        path = self._phase_path(phase)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("data", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load checkpoint for phase '%s': %s", phase, e)
            return None

    def mark_task_complete(self, phase: str, task_id: str) -> None:
        data = self.load_checkpoint(phase) or {}
        completed: list[str] = data.get("completed_tasks", [])
        if task_id not in completed:
            completed.append(task_id)
        data["completed_tasks"] = completed
        data["last_completed_task"] = task_id
        data["last_completed_at"] = datetime.utcnow().isoformat()
        self.save_checkpoint(phase, data)

    def is_task_complete(self, phase: str, task_id: str) -> bool:
        data = self.load_checkpoint(phase) or {}
        completed: list[str] = data.get("completed_tasks", [])
        return task_id in completed

    def get_phase_status(self, phase: str) -> dict[str, Any]:
        data = self.load_checkpoint(phase) or {}
        completed_tasks: list[str] = data.get("completed_tasks", [])
        total_tasks: int = data.get("total_tasks", len(completed_tasks))
        return {
            "phase": phase,
            "completed": len(completed_tasks),
            "total": total_tasks,
            "percent": (len(completed_tasks) / total_tasks * 100) if total_tasks > 0 else 0.0,
            "completed_tasks": completed_tasks,
            "last_completed_at": data.get("last_completed_at"),
        }

    def clear_phase(self, phase: str) -> None:
        path = self._phase_path(phase)
        if path.exists():
            path.unlink()
            logger.info("Cleared checkpoint for phase '%s'", phase)

    def clear_all(self) -> None:
        for path in self._dir.glob("phase_*.json"):
            path.unlink()
        logger.info("Cleared all checkpoints")

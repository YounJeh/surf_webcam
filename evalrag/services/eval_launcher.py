from __future__ import annotations

import configparser
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path


ConfigValue = str | int | float | bool
ConfigOverrides = dict[str, dict[str, ConfigValue]]
RUNS: dict[str, dict] = {}


class EvalLauncher:
    """
    Service pour lancer un run d'évaluation.
    """

    @staticmethod
    def _stringify_config_value(value: ConfigValue) -> str:
        """
        Convertit une valeur Python en chaîne pour pouvoir l'écrire dans un fichier .ini.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _apply_config_overrides(
        self,
        config: configparser.ConfigParser,
        overrides: ConfigOverrides,
    ) -> None:
        """
        Applique les surcharges de configuration au ConfigParser.
        """
        for section_name, section_values in overrides.items():
            if not config.has_section(section_name):
                config.add_section(section_name)

            for key, value in section_values.items():
                config.set(section_name, key, self._stringify_config_value(value))

    def _make_temp_config(
        self,
        run_name: str,
        base_config_path: str,
        overrides: ConfigOverrides,
    ) -> str:
        """
        Crée un fichier config.ini temporaire à partir d'une config de base
        et des overrides fournis par l'utilisateur.
        """
        cfg = configparser.ConfigParser()
        read_files = cfg.read(base_config_path, encoding="utf-8")

        if not read_files:
            raise FileNotFoundError(
                f"Impossible de lire le fichier de configuration de base : {base_config_path}"
            )

        self._apply_config_overrides(cfg, overrides)

        fd, temp_path = tempfile.mkstemp(prefix=f"{run_name}_", suffix=".ini")
        os.close(fd)

        with open(temp_path, "w", encoding="utf-8") as file:
            cfg.write(file)

        return temp_path

    def launch_run(
        self,
        *,
        run_name: str,
        mode: str,
        rows_concurrency: int,
        metrics_concurrency: int,
        config_path: str,
        config_overrides: ConfigOverrides,
    ) -> dict[str, str | datetime | None]:
        run_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        temp_config_path = self._make_temp_config(
            run_name=run_name,
            base_config_path=config_path,
            overrides=config_overrides,
        )

        cmd = [
            sys.executable,
            "-m",
            "evalrag.runners.run_eval",
            "--config",
            temp_config_path,
            "--mode",
            mode,
            "--rows-concurrency",
            str(rows_concurrency),
            "--metrics-concurrency",
            str(metrics_concurrency),
            "--run-name",
            run_name,
        ]

        proc = subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
        )

        RUNS[run_id] = {
            "run_id": run_id,
            "run_name": run_name,
            "log_path": None,
            "created_at": created_at,
            "process": proc,
        }

        return {
            "run_id": run_id,
            "status": "running",
            "run_name": run_name,
            "log_path": None,
            "created_at": created_at,
        }
    
    def get_run_status(self, run_id: str) -> dict[str, str | datetime | int | None] | None:
        print(f"[EVAL] lookup run_id={run_id} | RUNS connus={list(RUNS.keys())}")
        run = RUNS.get(run_id)
        if run is None:
            return None

        proc = run["process"]
        return_code = proc.poll()

        if return_code is None:
            status = "running"
        elif return_code == 0:
            status = "succeeded"
        else:
            status = "failed"

        return {
            "run_id": run["run_id"],
            "status": status,
            "run_name": run["run_name"],
            "log_path": run["log_path"],
            "created_at": run["created_at"],
            "return_code": return_code,
        }
"""
Lance plusieurs runs d'évaluation (un sous-processus par configuration) pour comparer
des réglages du RAG, ici la distance sémantique maximale du retriever.
"""
import configparser
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, TypedDict

from evalrag.config import DEFAULT_CONFIG, PACKAGE_ROOT

BASE_CONFIG = DEFAULT_CONFIG
EVAL_MODULE = "evalrag.runners.run_eval"

MAX_PARALLEL = 5

ConfigValue = str | int | float | bool
ConfigOverrides = dict[str, dict[str, ConfigValue]]


class RunDefinition(TypedDict):
    """
    Définition d'un run d'évaluation.

    Attributes:
        name: Nom logique du run, utilisé pour les logs et le nom du run Langfuse.
        config_overrides: Surcharges à appliquer au config.ini temporaire.
    """
    name: str
    config_overrides: ConfigOverrides


RUNS: list[RunDefinition] = [
    {
        "name": "embedding_max_semantic_0",
        "config_overrides": {
            "Database": {
                "PG_DATABASE": "embedding_demo",
            },
            "Response": {
                "TEMPERATURE": 0.2
            },
            "Retriever": {
                "MAX_SEMANTIC_DISTANCE": 0.85
            }
        },
    },
    {
        "name": "embedding_max_semantic_1",
        "config_overrides": {
            "Database": {
                "PG_DATABASE": "embedding_demo",
            },
            "Response": {
                "TEMPERATURE": 0.2
            },
            "Retriever": {
                "MAX_SEMANTIC_DISTANCE": 0.90
            }
        },
    },
    {
        "name": "embedding_max_semantic_2",
        "config_overrides": {
            "Database": {
                "PG_DATABASE": "embedding_demo",
            },
            "Response": {
                "TEMPERATURE": 0.2
            },
            "Retriever": {
                "MAX_SEMANTIC_DISTANCE": 0.95
            }
        },
    },
]


def _stringify_config_value(value: ConfigValue) -> str:
    """
    Convertit une valeur Python en chaîne exploitable par configparser.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _apply_config_overrides(
    config: configparser.ConfigParser,
    overrides: Mapping[str, Mapping[str, ConfigValue]],
) -> None:
    """
    Applique récursivement les overrides au ConfigParser.

    Si une section n'existe pas, elle est créée.

    Args:
        config: Configuration à modifier.
        overrides: Mapping de la forme :
            {
                "Section": {
                    "KEY": value,
                    ...
                },
                ...
            }
    """
    for section_name, section_values in overrides.items():
        if not config.has_section(section_name):
            config.add_section(section_name)

        for key, value in section_values.items():
            config.set(section_name, key, _stringify_config_value(value))

def make_temp_config(run: RunDefinition) -> str:
    """
    Crée un fichier config.ini temporaire pour un run donné.

    Args:
        run: Définition du run.

    Returns:
        Chemin du fichier temporaire créé.
    """
    cfg = configparser.ConfigParser()
    cfg.read(BASE_CONFIG, encoding="utf-8")

    _apply_config_overrides(cfg, run["config_overrides"])

    fd, temp_path = tempfile.mkstemp(prefix=f"{run['name']}_", suffix=".ini")
    os.close(fd)

    with open(temp_path, "w", encoding="utf-8") as file:
        cfg.write(file)

    return temp_path


def launch_eval(run: RunDefinition) -> dict[str, str | subprocess.Popen[str]]:
    """
    Lance un run d'évaluation dans un sous-processus.

    Args:
        run: Définition du run.

    Returns:
        Dictionnaire contenant les informations utiles sur le job lancé.
    """
    temp_config: str = make_temp_config(run)

    logs_dir = PACKAGE_ROOT / "out" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{run['name']}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    process: subprocess.Popen[str] = subprocess.Popen(
        [
            sys.executable,
            "-m",
            EVAL_MODULE,
            "--config",
            temp_config,
            "--mode",
            "fast",
            "--rows-concurrency",
            "4",
            "--metrics-concurrency",
            "2",
            "--run-name",
            f"rag-demo-eval-{run['name']}",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PACKAGE_ROOT,
    )

    return {
        "run_name": run["name"],
        "process": process,
        "temp_config": temp_config,
        "log_file": log_file,
        "log_path": str(log_path),
    }


def cleanup(job: dict[str, str | subprocess.Popen[str] | object]) -> None:
    """
    Ferme le fichier de log et supprime le fichier de config temporaire.
    """
    log_file = job["log_file"]
    if hasattr(log_file, "close"):
        log_file.close()

    temp_config = job["temp_config"]
    if isinstance(temp_config, str):
        try:
            os.remove(temp_config)
        except OSError:
            pass


def run_in_batches(
    runs: list[RunDefinition],
    max_parallel: int,
) -> list[dict[str, str | int]]:
    """
    Exécute les runs par batch de taille max_parallel.

    Args:
        runs: Liste des runs à lancer.
        max_parallel: Nombre maximal de processus lancés en parallèle.

    Returns:
        Liste des résultats d'exécution.
    """
    results: list[dict[str, str | int]] = []

    for index in range(0, len(runs), max_parallel):
        batch: list[RunDefinition] = runs[index:index + max_parallel]
        jobs: list[dict[str, str | subprocess.Popen[str]]] = [launch_eval(run) for run in batch]

        for job in jobs:
            process = job["process"]
            if not isinstance(process, subprocess.Popen):
                raise TypeError("Le champ 'process' n'est pas un subprocess.Popen valide.")

            return_code: int = process.wait()
            cleanup(job)

            run_name = job["run_name"]
            log_path = job["log_path"]

            if not isinstance(run_name, str):
                raise TypeError("Le champ 'run_name' doit être une chaîne.")
            if not isinstance(log_path, str):
                raise TypeError("Le champ 'log_path' doit être une chaîne.")

            results.append(
                {
                    "run_name": run_name,
                    "return_code": return_code,
                    "log_path": log_path,
                }
            )

            print(f"[{run_name}] terminé | code={return_code} | log={log_path}")

    return results


if __name__ == "__main__":
    results = run_in_batches(RUNS, MAX_PARALLEL)

    print("\nRésumé :")
    for result in results:
        print(
            f"- {result['run_name']}: "
            f"code={result['return_code']} | "
            f"log={result['log_path']}"
        )
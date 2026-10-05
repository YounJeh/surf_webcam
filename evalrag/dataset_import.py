"""
Import d'un dataset d'évaluation (fichier métier) vers Langfuse.

En production, le fichier métier est un Excel lu avec pandas. Ici il est fourni
au format JSON Lines (une ligne = une ligne de l'Excel, mêmes colonnes).
"""
import argparse
import json
from pathlib import Path
from typing import Any

from evalrag.config import load_config
from evalrag.stubs.langfuse_client import get_langfuse_client

REQUIRED_COLUMNS = {
    "request",
    "request_id",
    "expected_response",
    "expected_retrieved_context",
    "source_user",
    "source_tag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Importe un fichier métier en dataset Langfuse."
    )
    parser.add_argument("--file", required=True, help="Chemin vers le fichier (.jsonl)")
    parser.add_argument("--dataset-name", required=True, help="Nom du dataset Langfuse à créer/utiliser")
    return parser.parse_args()


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def load_rows(file_path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(file_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    columns = set().union(*(r.keys() for r in rows)) if rows else set()
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le fichier : "
            + ", ".join(sorted(missing))
        )

    return rows


def ensure_dataset(langfuse, dataset_name: str, source_file: str) -> None:
    try:
        langfuse.get_dataset(name=dataset_name)
        print(f"Dataset existant trouvé : {dataset_name}")
    except Exception:
        langfuse.create_dataset(
            name=dataset_name,
            description="Dataset importé depuis un fichier Excel",
            metadata={"source_file": source_file},
        )
        print(f"Dataset créé : {dataset_name}")


def import_rows(langfuse, dataset_name: str, rows: list[dict]) -> tuple[int, int]:
    created = 0
    skipped = 0

    for index, row in enumerate(rows):
        request = clean_value(row.get("request"))
        request_id = clean_value(row.get("request_id"))
        expected_response = clean_value(row.get("expected_response"))
        source_user = clean_value(row.get("source_user"))
        source_tag = clean_value(row.get("source_tag"))

        if request is None:
            skipped += 1
            print(f"Ligne {index + 2} ignorée : colonne 'request' vide")
            continue

        metadata_payload = {}
        if request_id is not None:
            metadata_payload["request_id"] = request_id
        if source_user is not None:
            metadata_payload["source_user"] = source_user
        if source_tag is not None:
            metadata_payload["source_tag"] = source_tag

        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=request,
            expected_output=expected_response or None,
            metadata=metadata_payload or None,
        )

        created += 1

    return created, skipped


def main() -> None:
    args = parse_args()
    load_config()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    rows = load_rows(file_path)
    langfuse = get_langfuse_client()

    ensure_dataset(langfuse=langfuse, dataset_name=args.dataset_name, source_file=str(file_path.name))
    created, skipped = import_rows(langfuse=langfuse, dataset_name=args.dataset_name, rows=rows)

    langfuse.flush()

    print("\nImport terminé")
    print(f"- Items créés : {created}")
    print(f"- Lignes ignorées : {skipped}")


if __name__ == "__main__":
    main()

"""
[STUB] Juge LLM hors ligne.

Remplace le LLM juge (endpoint compatible OpenAI, sortie structurée). Au lieu de
générer, il REJOUE des jugements annotés à la main dans data/judge_annotations.json :

- décomposition d'un texte en affirmations : table `texts[*].statements` ;
- une affirmation est "déductible du contexte" si TOUS les chunks listés dans son
  `supported_by` sont présents dans le contexte reçu (liste vide = jamais soutenue) ;
- questions régénérées depuis une réponse : table `texts[*].questions` (rejouées dans l'ordre) ;
- utilité d'un chunk pour une question : `items[*].useful_chunks` ;
- pertinence du contexte : `items[*].relevant_chunks` (tous présents = 2, certains = 1, aucun = 0) ;
- ancrage de la réponse : part des affirmations soutenues (toutes = 2, certaines = 1, aucune = 0) ;
- classification TP/FP/FN et notes 0/2/4 : tables `correctness` et `accuracy` (par paire de textes).

Un texte absent des annotations est traité comme un texte sans contenu exploitable
(aucune affirmation, note 0), ce que ferait à peu près un vrai juge.

Chaque appel est journalisé dans `self.calls` (utile pour dérouler une trace).
Il n'est pas nécessaire de lire ce fichier pour répondre aux questions.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evalrag.config import PACKAGE_ROOT

DATA_DIR = PACKAGE_ROOT / "data"


class StubJudgeLLM:
    def __init__(self, model: str, base_url: str = "", temperature: float = 0.0, max_tokens: int = 4096,
                 annotations_path: Path = DATA_DIR / "judge_annotations.json",
                 corpus_path: Path = DATA_DIR / "corpus.jsonl"):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        ann = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        self.texts = {t["text"]: t for t in ann["texts"].values()}
        self.items_by_question = {i["question"]: i for i in ann["items"].values()}
        self.correctness = ann["correctness"]
        self.accuracy = ann["accuracy"]
        self.texts_by_id = ann["texts"]
        self.text_ids = {t["text"]: tid for tid, t in ann["texts"].items()}

        self.statement_support: dict[str, list[str]] = {}
        for t in ann["texts"].values():
            for st in t.get("statements", []):
                self.statement_support[st["s"]] = st["supported_by"]

        corpus = [json.loads(l) for l in Path(corpus_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.chunk_text = {c["chunk_id"]: c["text"] for c in corpus}
        self.chunk_by_text = {c["text"]: c["chunk_id"] for c in corpus}

        self._question_cursor: dict[str, int] = defaultdict(int)
        self.calls: list[dict] = []

    # ------------------------------------------------------------------
    async def agenerate(self, request) -> dict:
        handler = getattr(self, f"_{request.task}")
        output = handler(**request.inputs)
        self.calls.append({"task": request.task, "inputs": request.inputs, "output": output})
        return output

    # ------------------------------------------------------------------
    def _chunks_in(self, context: str) -> set[str]:
        return {cid for cid, txt in self.chunk_text.items() if txt in context}

    def _supported(self, statement: str, present: set[str]) -> int:
        needed = self.statement_support.get(statement, [])
        return int(bool(needed) and all(c in present for c in needed))

    def _statements_of(self, text: str) -> list[str]:
        entry = self.texts.get(text)
        return [st["s"] for st in entry["statements"]] if entry else []

    def _pair(self, table: list[dict], a: str, b: str):
        ida, idb = self.text_ids.get(a), self.text_ids.get(b)
        for row in table:
            if row["response"] == ida and row["reference"] == idb:
                return row, False
            if row["response"] == idb and row["reference"] == ida:
                return row, True
        return None, False

    # ---- sous-tâches --------------------------------------------------
    def _statement_generator(self, question: str, answer: str) -> dict:
        return {"statements": self._statements_of(answer)}

    def _nli_statement(self, context: str, statements: list[str]) -> dict:
        present = self._chunks_in(context)
        return {"verdicts": [{"statement": s, "verdict": self._supported(s, present)} for s in statements]}

    def _answer_relevance(self, response: str) -> dict:
        entry = self.texts.get(response)
        if not entry or not entry.get("questions"):
            return {"question": "", "noncommittal": 1}
        idx = self._question_cursor[response] % len(entry["questions"])
        self._question_cursor[response] += 1
        q = entry["questions"][idx]
        return {"question": q["q"], "noncommittal": q["noncommittal"]}

    def _context_precision(self, question: str, context: str, answer: str) -> dict:
        item = self.items_by_question.get(question, {})
        chunk_id = self.chunk_by_text.get(context)
        useful = chunk_id in item.get("useful_chunks", []) and answer in self.texts
        return {"verdict": int(useful)}

    def _context_recall(self, question: str, context: str, answer: str) -> dict:
        present = self._chunks_in(context)
        return {"classifications": [
            {"statement": s, "attributed": self._supported(s, present)} for s in self._statements_of(answer)
        ]}

    def _correctness_classifier(self, question: str, answer: list[str], ground_truth: list[str]) -> dict:
        for row in self.correctness:
            r = [st["s"] for st in self.texts_by_id[row["response"]]["statements"]]
            g = [st["s"] for st in self.texts_by_id[row["reference"]]["statements"]]
            if r == answer and g == ground_truth:
                return {"TP": row["TP"], "FP": row["FP"], "FN": row["FN"]}
            if g == answer and r == ground_truth:
                return {"TP": row["TP"], "FP": row["FN"], "FN": row["FP"]}
        return {"TP": [], "FP": answer, "FN": ground_truth}

    def _answer_accuracy(self, query: str, user_answer: str, reference_answer: str, judge: int) -> dict:
        row, reversed_ = self._pair(self.accuracy, user_answer, reference_answer)
        if row is None:
            return {"rating": 0}
        # judge1 = note annotée pour (user_answer=réponse, reference_answer=référence),
        # judge2 = note annotée pour l'ordre inverse, quel que soit le prompt utilisé.
        key = "judge2" if reversed_ else "judge1"
        return {"rating": row[key]}

    def _response_groundedness(self, response: str, context: str, judge: int) -> dict:
        statements = self._statements_of(response)
        if not statements:
            return {"rating": 0}
        present = self._chunks_in(context)
        supported = sum(self._supported(s, present) for s in statements)
        rating = 2 if supported == len(statements) else (1 if supported else 0)
        return {"rating": rating}

    def _context_relevance(self, user_input: str, context: str, judge: int) -> dict:
        item = self.items_by_question.get(user_input, {})
        relevant = item.get("relevant_chunks", [])
        present = self._chunks_in(context)
        hits = sum(1 for c in relevant if c in present)
        rating = 2 if relevant and hits == len(relevant) else (1 if hits else 0)
        return {"rating": rating}

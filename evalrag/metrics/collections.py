"""
Portage allégé des métriques `ragas.metrics.collections` (version épinglée : ragas==0.4.3).

En production, le pipeline importe directement ces classes depuis la librairie ragas.
Pour que le package soit lisible et exécutable hors ligne, elles sont recopiées ici :

- la logique de calcul des scores (formules, cas limites, retries) est reprise de ragas ;
- les prompts du juge sont résumés en une consigne courte (`JudgeRequest.instruction`) ;
- l'appel au juge passe par `llm.agenerate(request)`, qui renvoie un dict structuré
  (en production : sortie structurée d'un LLM via un endpoint compatible OpenAI ;
  ici : le juge stub `evalrag.stubs.llm_judge.StubJudgeLLM`).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List


# ---------------------------------------------------------------------------
# Briques communes
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Résultat d'une métrique : se comporte comme un float."""

    value: float
    reason: str | None = None

    def __float__(self) -> float:
        return float(self.value)


@dataclass
class JudgeRequest:
    """Requête envoyée au juge LLM : une sous-tâche, sa consigne, ses entrées."""

    task: str
    instruction: str
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        parts = [self.instruction, ""]
        for key, value in self.inputs.items():
            parts.append(f"{key}: {value}")
        return "\n".join(parts)


class BaseMetric:
    def __init__(self, name: str, **kwargs):
        self.name = name


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# Consignes (résumées) des prompts ragas
STATEMENT_GENERATOR = (
    "Décompose la réponse à la question en affirmations atomiques, compréhensibles "
    "isolément (sans pronom). Renvoie la liste des affirmations."
)
NLI_STATEMENT = (
    "Pour chaque affirmation, indique si elle peut être directement déduite du contexte "
    "(verdict 1) ou non (verdict 0)."
)
ANSWER_RELEVANCE = (
    "Génère une question à laquelle cette réponse répond. Indique noncommittal=1 si la "
    "réponse est évasive ou vague (ex : 'je ne sais pas'), 0 sinon."
)
CONTEXT_PRECISION = (
    "Étant donné la question, la réponse et le contexte, indique si le contexte a été "
    "utile pour arriver à la réponse (verdict 1) ou non (verdict 0)."
)
CONTEXT_RECALL = (
    "Découpe la réponse en phrases. Pour chaque phrase, indique si elle peut être "
    "attribuée au contexte (attributed 1) ou non (attributed 0)."
)
CORRECTNESS_CLASSIFIER = (
    "Classe les affirmations : TP = affirmations de la réponse soutenues par la vérité "
    "terrain ; FP = affirmations de la réponse non soutenues par la vérité terrain ; "
    "FN = affirmations de la vérité terrain absentes de la réponse."
)
ANSWER_ACCURACY_JUDGE1 = (
    "Note la réponse utilisateur par rapport à la réponse de référence : 0 (inexacte), "
    "2 (partiellement exacte), 4 (exacte). Renvoie uniquement la note."
)
ANSWER_ACCURACY_JUDGE2 = (
    "Tu compares deux réponses à une même question. Note si la réponse de référence est "
    "équivalente à la réponse utilisateur : 0, 2 ou 4. Renvoie uniquement la note."
)
GROUNDEDNESS_JUDGE1 = (
    "La réponse est-elle ancrée dans le contexte ? 0 (pas ancrée), 1 (partiellement), "
    "2 (entièrement). Renvoie uniquement la note."
)
GROUNDEDNESS_JUDGE2 = (
    "Vérifie, affirmation par affirmation, si la réponse est étayée par le contexte. "
    "Note 0, 1 ou 2. Renvoie uniquement la note."
)
CONTEXT_RELEVANCE_JUDGE1 = (
    "Le contexte contient-il les informations nécessaires pour répondre à la question ? "
    "0 (non), 1 (partiellement), 2 (oui). Renvoie uniquement la note."
)
CONTEXT_RELEVANCE_JUDGE2 = (
    "Évalue la pertinence du contexte pour la question posée : 0, 1 ou 2. "
    "Renvoie uniquement la note."
)


def _average_scores(score1: float, score2: float) -> float:
    """Moyenne de deux juges, en ignorant un juge en échec (NaN)."""
    if not math.isnan(score1) and not math.isnan(score2):
        return (score1 + score2) / 2.0
    elif not math.isnan(score1):
        return score1
    elif not math.isnan(score2):
        return score2
    else:
        return float("nan")


async def _get_judge_rating(llm, request: JudgeRequest, allowed: list[int], max_retries: int) -> float:
    """Demande une note au juge ; re-tente si la note est hors de `allowed`."""
    for retry in range(max_retries):
        try:
            result = await llm.agenerate(request)
            rating = result["rating"]
            if rating in allowed:
                return float(rating)
            elif retry < max_retries - 1:
                continue
            else:
                return float("nan")
        except Exception:
            if retry < max_retries - 1:
                continue
            else:
                return float("nan")
    return float("nan")


# ---------------------------------------------------------------------------
# Métriques de génération (réponse vs contexte / vs question)
# ---------------------------------------------------------------------------

class Faithfulness(BaseMetric):
    """Part des affirmations de la réponse qui sont déductibles du contexte récupéré."""

    def __init__(self, llm, name: str = "faithfulness", **kwargs):
        self.llm = llm
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str, retrieved_contexts: List[str]) -> MetricResult:
        if not response:
            raise ValueError("response is missing. Please add response to the test sample.")
        if not user_input:
            raise ValueError("user_input is missing. Please add user_input to the test sample.")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts is missing. Please add retrieved_contexts to the test sample.")

        statements = await self._create_statements(user_input, response)
        if not statements:
            return MetricResult(value=float("nan"))

        context_str = "\n".join(retrieved_contexts)
        verdicts = await self._create_verdicts(statements, context_str)
        score = self._compute_score(verdicts)
        return MetricResult(value=float(score))

    async def _create_statements(self, question: str, response: str) -> List[str]:
        request = JudgeRequest("statement_generator", STATEMENT_GENERATOR, {"question": question, "answer": response})
        result = await self.llm.agenerate(request)
        return result["statements"]

    async def _create_verdicts(self, statements: List[str], context: str) -> List[dict]:
        request = JudgeRequest("nli_statement", NLI_STATEMENT, {"context": context, "statements": statements})
        result = await self.llm.agenerate(request)
        return result["verdicts"]

    def _compute_score(self, verdicts: List[dict]) -> float:
        if not verdicts:
            return float("nan")
        faithful_statements = sum(1 if v["verdict"] else 0 for v in verdicts)
        num_statements = len(verdicts)
        if num_statements > 0:
            score = faithful_statements / num_statements
        else:
            score = float("nan")
        return score


class AnswerRelevancy(BaseMetric):
    """Similarité entre la question posée et des questions régénérées à partir de la réponse."""

    def __init__(self, llm, embeddings, name: str = "answer_relevancy", strictness: int = 3, **kwargs):
        self.llm = llm
        self.embeddings = embeddings
        self.strictness = strictness
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not response:
            raise ValueError("response cannot be empty")

        generated_questions = []
        noncommittal_flags = []
        for _ in range(self.strictness):
            request = JudgeRequest("answer_relevance", ANSWER_RELEVANCE, {"response": response})
            result = await self.llm.agenerate(request)
            if result["question"]:
                generated_questions.append(result["question"])
                noncommittal_flags.append(result["noncommittal"])

        if not generated_questions:
            return MetricResult(value=0.0)

        all_noncommittal = all(noncommittal_flags)

        question_vec = await self.embeddings.aembed_text(user_input)
        gen_question_vecs = await self.embeddings.aembed_texts(generated_questions)
        cosine_sim = [_cosine(v, question_vec) for v in gen_question_vecs]

        score = (sum(cosine_sim) / len(cosine_sim)) * int(not all_noncommittal)
        return MetricResult(value=float(score))


class ResponseGroundedness(BaseMetric):
    """Deux juges notent (0/1/2) l'ancrage de la réponse dans le contexte ; moyenne normalisée."""

    def __init__(self, llm, name: str = "response_groundedness", max_retries: int = 5, **kwargs):
        self.llm = llm
        self.max_retries = max_retries
        super().__init__(name=name, **kwargs)

    async def ascore(self, response: str, retrieved_contexts: List[str]) -> MetricResult:
        if not response:
            raise ValueError("response is missing. Please add response to the test sample.")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts is missing. Please add retrieved_contexts to the test sample.")

        context_str = "\n".join(retrieved_contexts)
        if not response.strip() or not context_str.strip():
            return MetricResult(value=0.0)

        inputs = {"response": response, "context": context_str}
        judge1 = await _get_judge_rating(
            self.llm, JudgeRequest("response_groundedness", GROUNDEDNESS_JUDGE1, {**inputs, "judge": 1}), [0, 1, 2], self.max_retries
        )
        judge2 = await _get_judge_rating(
            self.llm, JudgeRequest("response_groundedness", GROUNDEDNESS_JUDGE2, {**inputs, "judge": 2}), [0, 1, 2], self.max_retries
        )
        score = _average_scores(judge1 / 2.0, judge2 / 2.0)
        return MetricResult(value=float(score))


# ---------------------------------------------------------------------------
# Métriques de retrieval (contexte vs question / vs référence)
# ---------------------------------------------------------------------------

class ContextPrecision(BaseMetric):
    """Précision moyenne pondérée par le rang des contextes jugés utiles pour la référence."""

    def __init__(self, llm, name: str = "context_precision", **kwargs):
        self.llm = llm
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, reference: str, retrieved_contexts: List[str]) -> MetricResult:
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        verdicts = []
        for context in retrieved_contexts:
            request = JudgeRequest(
                "context_precision", CONTEXT_PRECISION,
                {"question": user_input, "context": context, "answer": reference},
            )
            result = await self.llm.agenerate(request)
            verdicts.append(result["verdict"])

        score = self._calculate_average_precision(verdicts)
        return MetricResult(value=float(score))

    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        verdict_list = verdicts
        denominator = len(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        if math.isnan(score):
            logging.warning("Invalid response format. Expected a list of dictionaries with keys 'verdict'")
        return score


class ContextRecall(BaseMetric):
    """Part des phrases de la réponse de référence attribuables au contexte récupéré."""

    def __init__(self, llm, name: str = "context_recall", **kwargs):
        self.llm = llm
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, retrieved_contexts: List[str], reference: str) -> MetricResult:
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        context = "\n".join(retrieved_contexts) if retrieved_contexts else ""
        request = JudgeRequest(
            "context_recall", CONTEXT_RECALL,
            {"question": user_input, "context": context, "answer": reference},
        )
        result = await self.llm.agenerate(request)

        if not result["classifications"]:
            return MetricResult(value=float("nan"))

        attributions = [c["attributed"] for c in result["classifications"]]
        score = sum(attributions) / len(attributions) if attributions else float("nan")
        return MetricResult(value=float(score))


class ContextRelevance(BaseMetric):
    """Deux juges notent (0/1/2) la pertinence du contexte pour la question ; moyenne normalisée."""

    def __init__(self, llm, name: str = "context_relevance", max_retries: int = 5, **kwargs):
        self.llm = llm
        self.max_retries = max_retries
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, retrieved_contexts: List[str]) -> MetricResult:
        if not user_input:
            raise ValueError("user_input is missing. Please add user_input to the test sample.")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts is missing. Please add retrieved_contexts to the test sample.")

        context_str = "\n".join(retrieved_contexts)
        if not user_input.strip() or not context_str.strip():
            return MetricResult(value=0.0)
        if user_input.strip() == context_str.strip():
            return MetricResult(value=0.0)
        if context_str.strip() in user_input.strip():
            return MetricResult(value=0.0)

        inputs = {"user_input": user_input, "context": context_str}
        judge1 = await _get_judge_rating(
            self.llm, JudgeRequest("context_relevance", CONTEXT_RELEVANCE_JUDGE1, {**inputs, "judge": 1}), [0, 1, 2], self.max_retries
        )
        judge2 = await _get_judge_rating(
            self.llm, JudgeRequest("context_relevance", CONTEXT_RELEVANCE_JUDGE2, {**inputs, "judge": 2}), [0, 1, 2], self.max_retries
        )
        score = _average_scores(judge1 / 2.0, judge2 / 2.0)
        return MetricResult(value=float(score))


# ---------------------------------------------------------------------------
# Métriques bout-en-bout (réponse vs référence)
# ---------------------------------------------------------------------------

class AnswerCorrectness(BaseMetric):
    """0.75 x F1 factuel (affirmations TP/FP/FN) + 0.25 x similarité sémantique."""

    def __init__(self, llm, embeddings=None, name: str = "answer_correctness",
                 weights: List[float] = [0.75, 0.25], beta: float = 1.0, **kwargs):
        self.llm = llm
        self.embeddings = embeddings
        self.weights = weights
        self.beta = beta

        if len(weights) != 2:
            raise ValueError("Expects a list of two weights. First for factuality, second for semantic similarity")
        if all([w == 0 for w in weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in weights]):
            raise ValueError("Weights must be non-negative")
        if weights[1] > 0 and embeddings is None:
            raise ValueError("Embeddings are required for semantic similarity scoring.")
        if not isinstance(beta, float):
            raise ValueError("Beta must be a float.")
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str, reference: str) -> MetricResult:
        response_statements = await self._generate_statements(user_input, response)
        reference_statements = await self._generate_statements(user_input, reference)

        if response_statements and reference_statements:
            classification = await self._classify_statements(user_input, response_statements, reference_statements)
            factuality_score = self._compute_f1_score(classification)
        else:
            factuality_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            similarity_score = await self._calculate_similarity(response, reference)

        final_score = (
            (factuality_score * self.weights[0] + similarity_score * self.weights[1])
            / (self.weights[0] + self.weights[1])
        )
        return MetricResult(value=float(final_score))

    async def _generate_statements(self, question: str, text: str) -> List[str]:
        request = JudgeRequest("statement_generator", STATEMENT_GENERATOR, {"question": question, "answer": text})
        result = await self.llm.agenerate(request)
        return result["statements"]

    async def _classify_statements(self, question: str, answer_statements: List[str],
                                   ground_truth_statements: List[str]) -> dict:
        request = JudgeRequest(
            "correctness_classifier", CORRECTNESS_CLASSIFIER,
            {"question": question, "answer": answer_statements, "ground_truth": ground_truth_statements},
        )
        return await self.llm.agenerate(request)

    def _compute_f1_score(self, classification: dict) -> float:
        tp = len(classification["TP"])
        fp = len(classification["FP"])
        fn = len(classification["FN"])

        if tp + fp == 0:
            precision = 1.0 if fn == 0 else 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1.0 if fp == 0 else 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            return 0.0

        beta_squared = self.beta ** 2
        f_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
        return float(f_score)

    async def _calculate_similarity(self, response: str, reference: str) -> float:
        if self.embeddings is None:
            raise RuntimeError("Embeddings required for similarity calculation")
        response_embedding = await self.embeddings.aembed_text(response)
        reference_embedding = await self.embeddings.aembed_text(reference)
        return float(_cosine(response_embedding, reference_embedding))


class AnswerAccuracy(BaseMetric):
    """Deux juges notent (0/2/4) l'accord réponse / référence, dans les deux sens ; moyenne /4."""

    def __init__(self, llm, name: str = "answer_accuracy", max_retries: int = 5, **kwargs):
        self.llm = llm
        self.max_retries = max_retries
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str, reference: str) -> MetricResult:
        if not user_input:
            raise ValueError("user_input is missing. Please add user_input to the test sample.")
        if not response:
            raise ValueError("response is missing. Please add response to the test sample.")
        if not reference:
            raise ValueError("reference is missing. Please add reference to the test sample.")

        judge1 = await _get_judge_rating(
            self.llm,
            JudgeRequest("answer_accuracy", ANSWER_ACCURACY_JUDGE1,
                         {"query": user_input, "user_answer": response, "reference_answer": reference, "judge": 1}),
            [0, 2, 4], self.max_retries,
        )
        judge2 = await _get_judge_rating(
            self.llm,
            JudgeRequest("answer_accuracy", ANSWER_ACCURACY_JUDGE2,
                         {"query": user_input, "user_answer": reference, "reference_answer": response, "judge": 2}),
            [0, 2, 4], self.max_retries,
        )
        score = _average_scores(judge1 / 4.0, judge2 / 4.0)
        return MetricResult(value=float(score))

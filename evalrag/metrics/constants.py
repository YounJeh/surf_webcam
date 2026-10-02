from typing import Literal

from evalrag.metrics.collections import (
    AnswerAccuracy,
    AnswerCorrectness,
    ContextRelevance,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    ResponseGroundedness,
    AnswerRelevancy,
)

# Noms typés des métriques RAGAS utilisées dans le projet
RagasMetricName = Literal[
    "Faithfulness",  # Mesure les hallucinations (fidélité au contexte)
    "AnswerRelevancy",  # Mesure la compréhension de la question (empêche les réponses "à côté")
    "ContextPrecision",  # Mesure la qualité du retrieval (permet de savoir s'il y a trop de bruit)
    "ContextRecall",  # Indique si le retriever manque d'informations (règles absentes)
    "AnswerCorrectness",  # Compare la réponse du modèle à la réponse de référence (tolère les reformulations)
    "AnswerAccuracy",  # Plus strict que AnswerCorrectness (chiffres, dates, conditions)
    "ResponseGroundedness",  # Proche de Faithfulness mais plus fin (détecte les extrapolations)
    "ContextRelevance",  # Permet d'isoler les problèmes : retriever vs générateur
]

# Mapping entre le nom logique de la métrique et la classe RAGAS correspondante
RAGAS_METRIC_FACTORIES: dict[RagasMetricName, type] = {
    "Faithfulness": Faithfulness,
    "AnswerRelevancy": AnswerRelevancy,
    "ContextPrecision": ContextPrecision,
    "ContextRecall": ContextRecall,
    "AnswerCorrectness": AnswerCorrectness,
    "AnswerAccuracy": AnswerAccuracy,
    "ResponseGroundedness": ResponseGroundedness,
    "ContextRelevance": ContextRelevance,
}

# Sélection des métriques par mode
# C'est ici qu'on choisit quelles métriques sont actives.

FAST_METRICS: list[RagasMetricName] = [
    "ContextRelevance",
    "ContextRecall",
    "AnswerAccuracy",
    "ResponseGroundedness",
]

FULL_METRICS: list[RagasMetricName] = [
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    "AnswerCorrectness",
    "AnswerAccuracy",
    "ResponseGroundedness",
    "ContextRelevance",
]

# Point d'entrée unique pour build_metrics()
METRICS_BY_MODE: dict[str, list[RagasMetricName]] = {
    "fast": FAST_METRICS,
    "full": FULL_METRICS,
}

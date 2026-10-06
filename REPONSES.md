## A. Compréhension

### A1. Flux d'un run

| # | Étape | Où | Entrée → sortie |
|---|---|---|---|
| 1 | Import | `dataset_import.py:70` `import_rows` | ligne du golden dataset → item (`input`, `expected_output`) ; `expected_retrieved_context` n'est pas repris |
| 2 | Lancement | `run_eval.py:135` `run_rag_with_ragas` | config + mode → services, métriques, items traités par batchs de 4 |
| 3 | Retrieval | `run_eval.py:59` | question → chunks (distance ≤ `MAX_SEMANTIC_DISTANCE`, au plus `NB_RESULTS`) |
| 4 | Génération | `run_eval.py:64` | question + chunks → `response` |
| 5 | Scoring | `run_eval.py:72` → `ragas_metrics.py:95` | échantillon → `{métrique: score}` |
| 6 | Envoi | `run_eval.py:77` `score_trace`, `:235` `flush` | scores → trace Langfuse de l'item |
| 7 | Rapport | `run_report.py:260` | traces des runs → moyenne / min / max par run, items qui diffèrent (Markdown) |

### A2. Métriques du mode `fast`

| Métrique | Champs | Juge | Score |
|---|---|---|---|
| ContextRelevance | `user_input`, `retrieved_contexts` | 2 juges, note {0, 1, 2} | moyenne de `j1/2` et `j2/2` |
| ContextRecall | `user_input`, `retrieved_contexts`, `reference` | 1 juge, 1/0 par phrase de la référence attribuable au contexte | phrases attribuées / total |
| AnswerAccuracy | `user_input`, `response`, `reference` | 2 juges, note {0, 2, 4}, le juge 2 inverse réponse et référence | moyenne de `j1/4` et `j2/4` |
| ResponseGroundedness | `response`, `retrieved_contexts` | 2 juges, note {0, 1, 2} | moyenne de `j1/2` et `j2/2` |

Si un juge échoue, on garde l'autre ; si les deux échouent, `nan` (`collections.py:117`).

### A3. Familles

- **Retrieval** : ContextPrecision, ContextRecall, ContextRelevance.
- **Génération** : Faithfulness, ResponseGroundedness, AnswerRelevancy.
- **Bout-en-bout** : AnswerCorrectness, AnswerAccuracy.

Besoin d'une référence : ContextPrecision, ContextRecall, AnswerCorrectness, AnswerAccuracy.

### A4. Faithfulness vs answer_relevancy

- **faithfulness** compare la réponse au **contexte** : part des affirmations déductibles des chunks. Elle mesure les hallucinations.
- **answer_relevancy** compare la réponse à la **question** : similarité entre la question et 3 questions régénérées depuis la réponse (0 si la réponse est évasive). Elle mesure les réponses à côté.

Exemples : Q04 répond bien à la question mais invente le montant → bonne relevancy, mauvaise faithfulness. Répondre à Q01 (PRN) par « l'ALES est réservée aux moins de 26 ans », tiré d'ALES#1 présent dans le contexte → bonne faithfulness, mauvaise relevancy.

---

## B. Lecture critique

### B1. Problèmes relevés

**1. `response` et `reference` inversés**


`run_eval.py:72` appelle `score_with_ragas(..., expected_response, response)` alors que la signature (`ragas_metrics.py:95`) attend `(..., response, reference)`. Toutes les métriques sauf ContextRelevance reçoivent le mauvais texte : la faithfulness note la référence, le recall juge le contexte par rapport à la réponse générée. Q04 : faithfulness = 1.000 sur une réponse hallucinée (≈ 0.33 attendu, cf. C1).

**2. ContextPrecision ne suit pas ragas**


`collections.py:300` divise par le nombre de chunks au lieu du nombre de chunks pertinents. Q04, verdicts [1, 0, 0, 0] : 0.25 au lieu de 1.0, alors que le seul chunk utile est classé premier. Le score baisse dès qu'on récupère plus de chunks, ce qui biaise tout réglage du retriever.

**3. `nan` et référence vide mal gérés.**
- `run_eval.py:76` ne filtre que `None`, et `run_report.py:178` moyenne les `nan` : la moyenne de context_recall vaut `nan` sur les 3 runs du sweep (à cause de Q06).
- `run_eval.py:41` fait `str(item.expected_output)` : une référence vide devient le texte `"None"`, qui passe le garde-fou `if not reference`. Q08 est noté contre "None" au lieu d'être exclu.

**4. `max_semantic_distance` lu avec `get_int`**
- `max_semantic_distance` est lu avec `get_int` (`utils.py`) : il vaut 0 dans les métadonnées des 3 runs du sweep .

### B2. Confiance dans le sweep

Non.
- **Scores faux** : à cause de l'inversion, Q03, le seul cas où la distance compte vraiment (MAJO#0 à 0.91), ne bouge pas entre les runs.
- **Trop peu d'items concernés** : entre 0.85 et 0.95, seuls Q03 et Q05 changent de contexte. La décision repose sur 2 items sur 8, et un item pèse 0.125 dans la moyenne.
- **Bruit non contrôlé** : `TEMPERATURE 0.2` sans seed (`run_sweep.py:44`). Q02 tire au hasard l'une de ses deux variantes : un écart peut venir de la variante, pas de la distance. Un seul run par config, pas d'écart-type.
- **Mode `fast` en dur** (`run_sweep.py:167`) : ni faithfulness ni precision.

### B3. Lancement via l'API

`config_path` et `config_overrides` viennent du client sans validation ni authentification (`api_models.py:13-14`, `evaluation_router.py:18`). Un appelant peut rediriger `[Response] SERVER` vers son propre serveur : il reçoit les chunks de la documentation interne et la clé d'API, et peut renvoyer de fausses notes.
Correction : liste blanche des paramètres modifiables (`[Retriever]`), URL et modèles non surchargeables.

---

## C. Raisonnement sur les données

### C1. Q04

On retrouve les scores enregistrés : faithfulness = 3/3 = **1.0** ; context_precision = (1/1 × 1) / 4 = **0.25**.

Ils sont faux, pour deux raisons différentes :
- **faithfulness** : le juge a découpé la référence (64,20 €) à cause de l'inversion. Sur la vraie réponse, seule « 22 jours/mois » est soutenue ; 68,50 € et le cumul avec la PRN ne le sont pas → **0.33**.
- **context_precision** : ici c'est la formule. Le seul chunk utile est en tête → **1.0**.

Les scores disent « bonne génération, mauvais retrieval » ; c'est l'inverse.

### C2. Q05 et Q06

| Métrique | Q05 (légitime) | Q06 (illégitime) |
|---|---|---|
| faithfulness | 0 | nan (aucune affirmation) |
| answer_relevancy | 0 | 0 |
| context_recall | 0 | 1 |
| context_precision | 0 | 0.2 (1.0 avec ragas) |
| answer_accuracy | 1 | 0 |
| response_groundedness | 0 | 0 |
| context_relevance | 0 | 1 |

- **Distinguent les deux cas** : answer_accuracy, la seule qui juge bien les deux, et les métriques de contexte (qui disent seulement s'il y avait de quoi répondre).
- **Pénalisent Q05 à tort** : faithfulness, answer_relevancy, response_groundedness.
- **Conclusion** : les abstentions légitimes tirent les moyennes de génération vers le bas, et Q06 (`nan`) disparaît de la moyenne de faithfulness. Il faut taguer les questions hors périmètre et mesurer l'abstention à part.

### C3. Q03

- **À 0.90**, MAJO#0 (0.91) est coupé. Réponse : 310 €, sans majoration. Faithfulness = **1.0** (le générateur a bien utilisé son contexte), context_recall = 2/4 = **0.5** (la majoration et les 403 € manquent au contexte).
- **À 0.95**, MAJO#0 entre (avec ASA#1, hors sujet). Réponse : 403 €. Recall = **1.0**.
- **context_recall** montre que le problème vient du retriever. Mais avec l'inversion, il est calculé sur la réponse générée et vaut 1.0 aux deux seuils.

---

## D. Améliorations

### D1. Évolutions

| # | Évolution | Problème | Effort | Risque |
|---|---|---|---|---|
| 1 | Corriger l'inversion, ContextPrecision, `nan` / `"None"` + tests | scores faux | < 1 jour | historique incomparable : nouvelle baseline |
| 2 | Recall@k sur `expected_retrieved_context`, sans juge | retrieval jamais comparé à l'attendu | 1-2 jours | ids de chunks instables si on re-découpe |
| 3 | Abstention mesurée à part (`source_tag = hors_perimetre`) | moyennes faussées (C2) | 2-3 jours | détection d'abstention imparfaite |
| 4 | Juge séparé + dataset de 150-300 items stratifiés | auto-évaluation, 8 items | semaines | temps des experts |

### D2. Évolution n°1

```python
# run_eval.py:41
expected_response = item.expected_output or None          # plus de "None" en texte

# run_eval.py:72 : arguments nommés, l'inversion devient impossible
scores = await score_with_ragas(metrics, query=prompt, chunks=ctx_texts,
                                response=response, reference=expected_response,
                                max_concurrency=metrics_concurrency)

# ragas_metrics.py, score_with_ragas
REFERENCE_METRICS = {"context_precision", "context_recall", "answer_correctness", "answer_accuracy"}
async def run_metric(m):
    if m.name in REFERENCE_METRICS and not sample.reference:
        return None                                       # non applicable
    async with sem:
        return await call_ascore(m, sample)

value = float(r) if r is not None else None
scores[m.name] = None if value is None or math.isnan(value) else value

# collections.py:300 : formule ragas
denominator = sum(verdict_list) + 1e-10

def test_context_precision():
    assert round(ContextPrecision(llm=None)._calculate_average_precision([1, 0, 0, 0]), 3) == 1.0
```

### D3. Retours utilisateurs

Ils recoupent nos constats : SITU#1 hors sujet (Q07), « le bon montant est là, c'est la réponse qui est fausse » (Q04), « il manque la circulaire majoration » (Q03).

- **Usage** : retrouver la question via `message_id`, faire juger le même chunk par le juge, mesurer l'accord ; transformer les retours négatifs en nouveaux items du golden dataset.
- **Difficultés** : étiquettes non normalisées (`non_pertinent`, `NON PERTINENT`, `bof`), doublon (lignes 6 et 7), `manquant` qui n'est pas un avis sur un chunk cité, biais de sélection (seuls les chunks cités sont notés, surtout quand ça va mal), volume trop faible.

### D4. Intégration

- **CI (merge request touchant au RAG)** : tests unitaires des métriques, recall@k sans juge sur tout le dataset, puis 30-50 items stratifiés en mode `fast`, température 0. Seuils relatifs à la baseline de `main`, calibrés sur le bruit mesuré ; bloquant si `nb_error > 0` ou si un item critique régresse.
- **Évaluation planifiée (hebdomadaire, et à chaque ré-indexation ou changement de modèle)** : tout le dataset en mode `full`, juge séparé, plusieurs répétitions, résultats par type de question. C'est là qu'on fait les sweeps.

---

## Usage de l'IA

Claude  Code m'a aidé à la compréhension du code, vérification des calculs à partir des traces, reformulation et mise en forme des réponses.

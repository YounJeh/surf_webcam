A1. Décrivez le flux complet d'un run d'évaluation, de la ligne du golden dataset jusqu'au score visible dans le rapport de comparaison. Pour chaque étape : fichier, fonction, données en entrée et en sortie.

| # | étape | où | entrée | sortie |
|---|---|---|---|---|
| 1 | import du dataset | `dataset_import.py:70` `import_rows` | ligne du golden dataset | item Langfuse : `input` = question, `expected_output` = réponse attendue (`expected_retrieved_context` n'est pas repris) |
| 2 | init du run | `run_eval.py:135` `run_rag_with_ragas` | config .ini, mode `fast`/`full` | items, retriever, générateur, métriques (`build_metrics`), lancement des items par batchs de 4 |
| 3 | retrieval | `run_eval.py:59` `retrieve_chunks` | question | chunks filtrés (distance ≤ `MAX_SEMANTIC_DISTANCE`, max `NB_RESULTS`) puis `ctx_texts` |
| 4 | génération | `run_eval.py:64` `generate_response` | question + chunks | `response` |
| 5 | scoring | `run_eval.py:72` → `ragas_metrics.py:95` `score_with_ragas` | appel : `(prompt, ctx_texts, expected_response, response)` ; signature : `(query, chunks, response, reference)` | `{métrique: score}`, `None` si la métrique plante |
| 6 | envoi des scores | `run_eval.py:77` `score_trace` | scores | scores attachés à la trace de l'item, + `nb_error` |
| 7 | persistance | `run_eval.py:235` `langfuse.flush()` | runs + traces | JSON dans `out/langfuse/` |
| 8 | rapport | `run_report.py:260` `main` | noms des runs + des scores | relit les traces (`_load_run_results`), garde les items communs, moyenne/min/max par run, pires items, items qui diffèrent : Markdown dans `out/reports/` |

Deux niveaux de concurrence : les items (sémaphore + batchs, `run_eval.py:168-201`) et les métriques d'un même item (sémaphore, `ragas_metrics.py:101`).


A2. Pour chacune des 4 métriques du mode fast, indiquez : les champs de l'échantillon qu'elle utilise (user_input, retrieved_contexts, response, reference), ce que le juge LLM doit produire, et la formule qui transforme ces jugements en score.

| métrique | champs | ce que produit le juge | formule |
|---|---|---|---|
| ContextRelevance (`collections.py:341`) | `user_input`, `retrieved_contexts` | 2 juges, une note parmi {0, 1, 2} : le contexte permet-il de répondre à la question ? | moyenne de `j1/2` et `j2/2` |
| ContextRecall (`collections.py:311`) | `user_input`, `retrieved_contexts`, `reference` | 1 juge : découpe la référence en phrases, 1/0 par phrase selon qu'elle est attribuable au contexte | phrases attribuées / nb de phrases |
| AnswerAccuracy (`collections.py:464`) | `user_input`, `response`, `reference` | 2 juges, une note parmi {0, 2, 4} ; le juge 2 inverse réponse et référence | moyenne de `j1/4` et `j2/4` |
| ResponseGroundedness (`collections.py:236`) | `response`, `retrieved_contexts` | 2 juges, une note parmi {0, 1, 2} : la réponse est-elle ancrée dans le contexte ? | moyenne de `j1/2` et `j2/2` |

Tous les scores sont entre 0 et 1. Les deux juges d'une même métrique sont le même LLM avec deux formulations de prompt. Si un juge échoue (note hors liste 5 fois de suite), on garde l'autre ; si les deux échouent, `nan` (`_average_scores`, `collections.py:117`). ContextRecall renvoie aussi `nan` si le juge ne classe aucune phrase.

A3. Classez les 8 métriques en trois familles : qualité du retrieval, qualité de la génération, qualité bout-en-bout. Lesquelles ne peuvent pas être calculées sans réponse de référence ?

| famille | métriques |
|---|---|
| retrieval | ContextPrecision, ContextRecall, ContextRelevance |
| génération | Faithfulness, ResponseGroundedness, AnswerRelevancy |
| bout-en-bout | AnswerCorrectness, AnswerAccuracy |

Besoin d'une référence (`reference` dans `ascore`) : ContextPrecision, ContextRecall, AnswerCorrectness, AnswerAccuracy. Les 4 autres peuvent tourner sans golden dataset, donc aussi en production.

À noter : ContextPrecision et ContextRecall jugent le retrieval par rapport à la réponse de référence (texte), pas par rapport aux chunks attendus (`expected_retrieved_context`), qui ne sont jamais importés (`dataset_import.py:94`).


A4. Quelle est la différence entre faithfulness et answer_relevancy ? Donnez un exemple de réponse qui a un bon score sur l'une et un mauvais sur l'autre.

- **faithfulness** (`collections.py:153`) : réponse vs **contexte**. La réponse est découpée en affirmations, chacune jugée déductible du contexte (1) ou non (0). Score = affirmations déductibles / total. Mesure les hallucinations.
- **answer_relevancy** (`collections.py:199`) : réponse vs **question**. Le juge génère 3 questions à partir de la réponse ; score = similarité cosinus moyenne avec la vraie question, et 0 si la réponse est évasive. Mesure les réponses à côté.

Aucune des deux n'utilise la référence : elles ne disent pas si la réponse est juste.

- Bonne relevancy, mauvaise faithfulness : **Q04**. La réponse traite bien la question (montant journalier ASA) mais donne 68,50 € au lieu de 64,20 € et invente le cumul avec la PRN.
- Bonne faithfulness, mauvaise relevancy : sur Q01, répondre « l'ALES est réservée aux moins de 26 ans » (ALES#1 est dans le contexte). Tout est fidèle au contexte mais ne répond pas à la question sur la PRN. Version atténuée dans les données : **Q07** et son paragraphe hors sujet sur le déménagement.


B1. Relevez les problèmes que vous voyez dans ce code : bugs, limites, choix discutables. Pour chacun : emplacement, impact concret sur les chiffres produits, gravité (bloquant, important, mineur). Classez-les du plus grave au moins grave.

**Bloquants**

1. **`response` et `reference` inversés** (`run_eval.py:72` vs signature `ragas_metrics.py:95`). Toutes les métriques sauf ContextRelevance reçoivent le mauvais texte : faithfulness / groundedness / relevancy notent la réponse de référence, recall / precision jugent le contexte par rapport à la réponse générée. Q04 : faithfulness = 1.000 alors que la réponse est hallucinée (attendu ≈ 0.33, cf. C1). C'est aussi pour ça que le sweep ne voit aucune différence sur Q03.
2. **Référence vide → chaîne `"None"`** (`run_eval.py:41`, `str(item.expected_output)` ; `dataset_import.py:97` met `None`). Le garde-fou `if not reference` ne se déclenche pas : Q08 est noté contre le texte "None" au lieu d'être exclu (answer_accuracy = 0.0 dans mon run) et tire les moyennes vers le bas.
3. **ContextPrecision ne suit pas ragas** (`collections.py:299`). On divise par le nombre de chunks au lieu du nombre de chunks pertinents. Q04, verdicts [1, 0, 0, 0] : 0.25 au lieu de 1.0. Le chunk utile est pourtant classé premier.

**Importants**

4. **Les `nan` ne sont pas filtrés** (`run_eval.py:76` teste seulement `None`, `run_report.py:76` et `:178`). Un seul `nan` rend la moyenne `nan` : context_recall = nan sur les 3 runs du sweep à cause de Q06. À l'inverse, une métrique qui plante (`None`) sort l'item de la moyenne sans le signaler, et les runs ne sont plus comparés sur les mêmes items. Par exemple,si aucune affirmation n'est contenue dans la réponse, ContextRecall vaut alors nan.
5. **Juge = LLM de génération** (`ragas_metrics.py:33-34` lit la section `[Response]`). Le modèle note ses propres réponses : biais d'auto-évaluation, scores optimistes.
6. **Sweep non contrôlé** (`run_sweep.py:44`, `TEMPERATURE 0.2` sans seed + `rag_services.py:119`). Q02 tire une variante au hasard à chaque run : un écart entre deux runs peut venir du hasard et pas de la distance. En plus, le mode `fast` est en dur (`run_sweep.py:167`) : pas de faithfulness, alors que le README la demande dans le rapport. On lance run_sweep sur des données générées pour ne pas relancer le modèle. Or des fois il y a plusieurs variantes et si on ne fixe pas une variante, la différence de performance entre chaque run n'est pas lié à TEMPERATURE mais peut être lié à l'exemple tiré qui est plus ou moins bon.
7. **Métadonnées de run fausses** (`utils.py`, `get_rag_run_config`). `max_semantic_distance` est lu avec `get_int` : `"0.9"` → exception → 0. Les 3 runs du sweep affichent 0 dans Langfuse, on ne peut pas savoir lequel est lequel. Les deux clés `glossaire_path` s'écrasent aussi.

**Mineurs**

8. **`expected_retrieved_context` exigé puis jeté** (`dataset_import.py:15` vs `:94`). Les chunks attendus par l'expert ne servent jamais : pas de vrai rappel du retrieval.
9. **Batchs + sémaphore** (`run_eval.py:197-201`). Chaque batch attend son item le plus lent : plus long, mais sans effet sur les scores.

B2. run_sweep.py sert à choisir la valeur de MAX_SEMANTIC_DISTANCE. En l'état, feriez-vous confiance à la comparaison produite par run_report.py pour trancher ? Justifiez.

Non, en l'état je ne trancherais pas avec ce rapport. Il y a très peu d'exemple impactés par les changement de MAX_SEMANTIC_DISTANCE. La variance est trop importante. Il faudrait ajouter plus d'exemple représentatifs.

- **Les scores sont faux** (B1 n°1). Avec l'inversion, recall et precision jugent le contexte par rapport à la réponse générée. Q03, le cas où la distance compte vraiment (MAJO#0 à 0.91), ne bouge pas entre les runs.
- **Peu d'items concernés.** D'après `data/README.md`, passer de 0.85 à 0.95 ne change le contexte que de Q03 et Q05. La décision repose sur 2 items sur 8, et un item = 0.125 de moyenne.
- **Du bruit non contrôlé.** `TEMPERATURE 0.2` sans seed (`run_sweep.py:44`) : Q02 tire sa variante au hasard. Un seul run par config, pas d'écart-type : impossible de séparer l'effet de la distance du hasard.
- **Mode `fast` en dur** (`run_sweep.py:167`) : pas de faithfulness ni de precision, alors que c'est le compromis bruit / rappel qu'on cherche à mesurer. Et si on passe en `full`, ContextPrecision baisse mécaniquement quand il y a plus de chunks (B1 n°3) : biais vers le seuil le plus strict.
- **Rapport fragile** : une moyenne à `nan` (Q06) et la distance enregistrée à 0 pour les 3 runs (B1 n°7).

Pour trancher : corriger les bugs, `TEMPERATURE=0`, mode `full`, mesurer le retrieval directement avec `expected_retrieved_context` (rappel des chunks attendus, sans juge), et regarder Q03 / Q05 au cas par cas plutôt que la moyenne. 

B3. Un point sur le lancement d'une évaluation via l'API (evalrag/api/, evalrag/services/eval_launcher.py) qui vous gênerait avant une mise en production.


**Le client contrôle la config** (`api_models.py:13-14`, `eval_launcher.py:47`). `config_path` et `config_overrides` sont pris tels quels, sans authentification sur la route (`evaluation_router.py:18`). Un appelant peut rediriger `[Response] SERVER` vers son propre serveur : il reçoit les chunks de la doc interne et la clé d'API (envoyée par le client LLM dans `Authorization`), et peut renvoyer de fausses notes (scores falsifiés). Il peut aussi changer `PG_DATABASE` pour viser une autre base.
Correction : liste blanche des clés modifiables (`[Retriever]` uniquement), config de base choisie côté serveur par un identifiant, URL / modèles / base jamais surchargeables, authentification (jeton SSO + rôle) et API limitée au réseau interne.

C1. Lisez traces/trace_01_Q04_hallucination.md. La réponse générée contient un montant faux et une règle inventée.

Recalculez à la main faithfulness et context_precision à partir des valeurs intermédiaires de la trace. Obtenez-vous les scores enregistrés ?
Ces scores décrivent-ils correctement la qualité de cette réponse ? Sinon, expliquez pourquoi, et donnez les valeurs que vous attendriez.

**Recalcul** : oui, on retrouve les scores enregistrés.
- faithfulness : 3 affirmations, 3 verdicts à 1 → 3/3 = **1.000**.
- context_precision : verdicts [1, 0, 0, 0] → (1/1 × 1) / 4 chunks = **0.250**.

**Mais les scores sont faux**, pour deux raisons différentes :
- **faithfulness** : à cause de l'inversion (`run_eval.py:72`), le juge a découpé la *référence* (64,20 €), pas la réponse générée. Sur la vraie réponse : « 68,50 €/jour » non soutenu, « 22 jours/mois » soutenu par ASA#0, « cumulable avec la PRN » non soutenu (PRN#1 dit même l'inverse) → **1/3 ≈ 0.33** au lieu de 1.0.
- **context_precision** : ici l'inversion ne change rien (seul ASA#0 est utile dans les deux cas), c'est la formule. On divise par le nombre de chunks au lieu du nombre de chunks utiles (`collections.py:300`). Le seul chunk utile est classé premier → **1.0** au lieu de 0.25.

Lecture correcte : retrieval parfait (le bon chunk en tête), génération qui hallucine. Les scores enregistrés disent exactement l'inverse.

C2. Lisez traces/trace_02_Q05_Q06_abstentions.md. Q05 est une abstention légitime (question hors périmètre), Q06 une abstention illégitime (la réponse est dans le contexte).

Complétez les deux tableaux de scores à partir des jugements élémentaires.
Quelles métriques distinguent les deux cas ? Lesquelles pénalisent à tort l'abstention légitime ? Qu'en concluez-vous pour l'interprétation des moyennes ?

| métrique | Q05 (légitime) | Q06 (illégitime) |
|---|---|---|
| faithfulness | 0 (0/2) | nan (aucune affirmation) |
| answer_relevancy | 0 (tout noncommittal) | 0 (tout noncommittal) |
| context_recall | 0 (0/2) | 1 (3/3) |
| context_precision | 0 | 0.2 avec le code (1/5), 1.0 avec ragas |
| answer_accuracy | 1 ((4/4 + 4/4)/2) | 0 |
| response_groundedness | 0 | 0 |
| context_relevance | 0 | 1 ((2/2 + 2/2)/2) |

- **Ce qui distingue les deux cas** : answer_accuracy (1 vs 0), la seule qui juge bien les deux abstentions, car la référence de Q05 dit de s'abstenir. Les métriques de contexte distinguent aussi (0 vs 1), mais elles disent seulement s'il y avait de quoi répondre.
- **Ce qui pénalise Q05 à tort** : faithfulness, answer_relevancy et response_groundedness valent 0 pour un comportement correct. Toute abstention a 0 en relevancy (`collections.py:232`).
- **Pour les moyennes** : les abstentions légitimes tirent les métriques de génération vers le bas, et l'abstention illégitime de Q06 donne `nan` en faithfulness : filtrée, elle disparaît de la moyenne. Il faut étiqueter les questions hors périmètre dans le golden dataset, les sortir des moyennes de génération et mesurer l'abstention à part (s'abstient-il quand il faut, et seulement quand il faut ?).

C3. Q03 (voir data/README.md) demande de combiner deux documents. Que se passe-t-il avec MAX_SEMANTIC_DISTANCE=0.90, puis 0.95 ? Quelle métrique vous permettrait de dire que le problème vient du retriever et non du générateur ?
- **À 0.90** : MAJO#0 (0.91) est coupé. Contexte = CGPE#0, CGPE#1, SITU#0. Le générateur répond 310 €, sans la majoration. faithfulness = 2/2 = **1.0** : il a bien utilisé ce qu'il avait. context_recall = 2/4 = **0.5** : les phrases « majoration de 30 % » et « 403 € » de la référence ne sont pas dans le contexte.
- **À 0.95** : MAJO#0 entre, mais aussi ASA#1 (0.94, hors sujet). Le générateur répond 403 €. faithfulness = 1.0, context_recall = 4/4 = **1.0**. context_precision baisse avec la formule du code (verdicts [1, 0, 0, 1, 0] → 1.5/5 = 0.3 ; 0.75 avec ragas).
- **La métrique qui tranche : context_recall.** Faithfulness à 1 + recall à 0.5 = le générateur a fait son travail, c'est le retriever qui n'a pas ramené l'info. Mais avec l'inversion (B1 n°1), le recall est calculé sur la réponse générée → 1.0 aux deux seuils : le problème est invisible.
- Plus simple et sans juge : comparer aux chunks attendus `expected_retrieved_context = CGPE#0;MAJO#0` (MAJO#0 absent à 0.90), mais ils ne sont pas importés.

D1. Évolutions, de la plus simple à la plus ambitieuse

| # | évolution | problème résolu | effort | risque |
|---|---|---|---|---|
| 1 | Corriger les bugs bloquants (inversion, `str(None)`, `nan`, ContextPrecision, `get_int`) + tests unitaires | scores faux (B1 n°1-4) | < 1 jour | historique incomparable : nouvelle baseline, versionner les métriques |
| 2 | Métriques de retrieval sans juge sur `expected_retrieved_context` (hit@k, recall@k, MRR) | retrieval jamais comparé aux chunks de l'expert. Q03 : recall@5 = 0.5 à 0.90, 1.0 à 0.95 | 1-2 jours | les ids de chunks changent si on re-découpe le corpus |
| 3 | Traiter les abstentions à part (`source_tag = hors_perimetre`) : taux de refus à tort, taux de réponse hors périmètre, métriques de génération seulement sur les réponses | abstentions qui faussent les moyennes (C2) | 2-3 jours | détection d'abstention imparfaite, peu d'items tagués |
| 4 | Sweep fiable : température 0, comparaison item par item, intervalle de confiance (bootstrap), ids d'items dans le rapport | impossible de trancher (B2) | 3-5 jours | coût des répétitions |
| 5 | Juge séparé et calibré sur des annotations humaines + golden dataset de 150-300 items stratifiés par type | juge = générateur, 8 items | plusieurs semaines | temps des experts, maintenance quand la réglementation change |

D2. Évolution n°1

```python
# run_eval.py:41 : plus de "None" en texte
expected_response = item.expected_output or None

# run_eval.py:72 : arguments nommés, l'inversion devient impossible
scores = await score_with_ragas(metrics, query=prompt, chunks=ctx_texts,
                                response=response, reference=expected_response,
                                max_concurrency=metrics_concurrency)

# ragas_metrics.py, score_with_ragas
REFERENCE_METRICS = {"context_precision", "context_recall", "answer_correctness", "answer_accuracy"}

async def run_metric(m):
    if m.name in REFERENCE_METRICS and not sample.reference:
        return None                       # pas de référence : non applicable
    async with sem:
        return await call_ascore(m, sample)

value = float(r) if r is not None else None
if value is not None and math.isnan(value):
    value = None                          # compté en "manquant" dans le rapport, pas dans la moyenne
scores[m.name] = value

# collections.py:300 : formule ragas
denominator = sum(verdict_list) + 1e-10

# test
def test_context_precision():
    assert round(ContextPrecision(llm=None)._calculate_average_precision([1, 0, 0, 0]), 3) == 1.0
```

D3. Retours utilisateurs

Les retours confirment ce qu'on a trouvé : SITU#1 jugé hors sujet pour une reprise d'emploi (Q07), « le bon montant est là, c'est la réponse qui est fausse » sur ASA#0 (Q04), « il manque la circulaire majoration » (Q03).

Exploitation :
- **accord juge / humains** : retrouver question et réponse via `message_id`, faire juger le même chunk par le juge (verdict utile / pas utile de ContextPrecision), comparer et calculer un kappa de Cohen ;
- **enrichir le golden dataset** avec les cas notés négativement (le retour « manquant » donne directement un chunk attendu) ;
- **garder séparés** l'avis sur le chunk (retrieval) et l'avis sur la réponse (génération) : le retour n°5 montre qu'un bon chunk peut aller avec une mauvaise réponse.

Difficultés :
- étiquettes libres : `pertinent`, `non_pertinent`, `NON PERTINENT`, `bof`, `manquant` (`eval` en VARCHAR, `document_evaluation.py:45`) ;
- doublon (lignes 6 et 7) ;
- `manquant` est un autre signal (chunk absent, pas de `chunk_index`) ;
- pas de question dans la ligne, jointure obligatoire avec les traces ;
- biais : seuls les chunks cités sont notés, et on note surtout quand ça ne va pas ;
- 8 retours, il en faut 100-200 pour un kappa fiable ;
- `chunk_index` change si on re-découpe ;
- commentaires libres qui peuvent contenir des données personnelles (RGPD).

D4. Intégration

Coût : environ 7 appels au juge par item en `fast`, 20 en `full`.

**(a) CI sur merge request**, seulement si la MR touche au RAG (retriever, prompts, config, modèle) :
1. tests unitaires des métriques (gratuit, aurait bloqué l'inversion) ;
2. retrieval sans juge sur tout le dataset (gratuit) ;
3. 30-50 items stratifiés par type, mode `fast`, température 0 (~10 min, quelques centaines d'appels).

Seuils relatifs à la baseline de `main`, calibrés sur le bruit (baseline lancée 5 fois, blocage au-delà de 2 écarts-types). Bloquant dans tous les cas : `nb_error > 0`, baisse du recall des chunks attendus, item critique qui passe de réussi à échoué. Le rapport de la MR liste les items qui ont changé, pas seulement les moyennes.

**(b) Évaluation planifiée**, chaque semaine et à chaque ré-indexation ou changement de modèle : tout le dataset, mode `full`, juge séparé, 3 répétitions, intervalles de confiance, résultats par type de question et matrice d'abstention. C'est là qu'on fait les sweeps. Suivi dans Langfuse avec alerte, et audit mensuel de 20-30 jugements par un expert pour détecter une dérive du juge.

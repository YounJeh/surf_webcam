# Partie D : brouillon détaillé (à relire, puis à raccourcir pour REPONSES.md)

---

## D1. Évolutions du pipeline, de la plus simple à la plus ambitieuse

L'idée directrice : **d'abord rendre les chiffres justes, ensuite les rendre fiables, enfin les rendre utiles.** Inutile d'agrandir le dataset ou de brancher la CI tant que les scores sont faux.

### 1. Corriger les bugs bloquants et les verrouiller par des tests

**Problème résolu** : les scores actuels sont faux (B1 n°1 à n°4).
- inversion `response` / `reference` (`run_eval.py:72`) ;
- `str(None)` qui transforme une référence vide en texte "None" (`run_eval.py:41`) ;
- `nan` non filtrés, qui cassent les moyennes (`run_eval.py:76`, `run_report.py:171-178`) ;
- dénominateur de ContextPrecision différent de ragas (`collections.py:300`) ;
- au passage : `get_int` → `get_float` pour `max_semantic_distance` dans `utils.py`.

**Exemple d'impact** : Q04 passe de faithfulness 1.0 à 0.33 et de context_precision 0.25 à 1.0. Q03 se met enfin à bouger entre 0.90 et 0.95 (recall 0.5 → 1.0).

**Effort** : moins d'une journée. Quelques lignes de code + 5-6 tests unitaires sur les formules (ex. `average_precision([1,0,0,0]) == 1.0`, et un test qui vérifie que la `reference` reçue par la métrique est bien le texte du golden dataset).

**Risque** : faible techniquement, mais **tous les scores historiques deviennent incomparables** avec les nouveaux. Il faut le dire à l'équipe, relancer une baseline, et versionner les métriques (ex. `metrics_version=2` dans les métadonnées du run) pour ne jamais mélanger avant/après dans un rapport.

### 2. Mesurer le retrieval sans juge, avec `expected_retrieved_context`

**Problème résolu** : aujourd'hui, le retrieval n'est jugé que par un LLM et par rapport à un texte de référence. Les chunks désignés par l'expert sont exigés à l'import puis jetés (`dataset_import.py:15` vs `:94`).

**Ce qu'on ajoute** : les garder dans les métadonnées de l'item, puis calculer des métriques classiques de recherche d'information, **déterministes et gratuites** :
- **hit@k** : au moins un chunk attendu dans les k premiers ?
- **recall@k** : part des chunks attendus récupérés ;
- **MRR** : 1 / rang du premier chunk attendu.

**Exemple** : Q03, attendu `CGPE#0;MAJO#0`.
- à 0.90 : récupérés CGPE#0, CGPE#1, SITU#0 → recall@5 = 1/2 = **0.5**, MRR = 1.0 ;
- à 0.95 : MAJO#0 arrive au rang 4 → recall@5 = **1.0**.

On voit immédiatement que le seuil change quelque chose, sans aucun appel LLM, sans bruit, en quelques millisecondes.

**Effort** : 1 à 2 jours (import + 3 fonctions + colonnes dans le rapport).

**Risque** : les identifiants de chunks dépendent du découpage (`CHUNK_SIZE=2000`, `CHUNK_OVERLAP=300`). Si on re-découpe le corpus, `MAJO#0` ne désigne plus le même texte et les attendus deviennent faux. Parade : stocker l'attendu au niveau **document** (`document_id`) en plus du chunk, ou un extrait de texte qu'on recherche dans le chunk.

### 3. Traiter les abstentions comme un cas à part

**Problème résolu** (C2) : une abstention légitime (Q05) prend 0 sur faithfulness, answer_relevancy et groundedness ; une abstention illégitime (Q06) prend `nan` en faithfulness et disparaît de la moyenne. Les moyennes punissent la prudence et cachent le pire cas.

**Ce qu'on ajoute** :
- le golden dataset a déjà la bonne info : `source_tag = hors_perimetre` pour Q05. On s'en sert ;
- on détecte si la réponse est une abstention (le flag `noncommittal` d'AnswerRelevancy le fait déjà, ou une petite classification par le juge) ;
- on calcule une **matrice d'abstention** :

| | le modèle s'abstient | le modèle répond |
|---|---|---|
| **hors périmètre** (doit s'abstenir) | ✅ bon refus (Q05) | ❌ hallucination hors sujet |
| **dans le périmètre** (doit répondre) | ❌ refus à tort (Q06) | ✅ → métriques habituelles |

- on en tire deux taux : **taux de refus à tort** (Q06) et **taux de réponse hors périmètre** ;
- les métriques de génération (faithfulness, relevancy, groundedness) ne sont calculées que sur la case « dans le périmètre + répond ».

**Effort** : 2 à 3 jours.

**Risque** : la détection d'abstention peut se tromper (« Je ne sais pas exactement, mais en général… » est-ce une abstention ?). Et il faut que les experts taguent proprement les questions hors périmètre (il n'y en a qu'une sur 8).

### 4. Rendre le sweep et le rapport fiables

**Problème résolu** (B2) : on ne peut pas trancher sur `MAX_SEMANTIC_DISTANCE` aujourd'hui.

**Ce qu'on change** :
- `TEMPERATURE=0` (ou seed fixe) pendant l'évaluation : Q02 ne tire plus sa variante au hasard ;
- mode `full` ou au minimum les métriques de retrieval du point 2 ;
- **comparaisons appariées** : on compare chaque item à lui-même entre deux runs, pas deux moyennes ;
- **intervalle de confiance** par bootstrap sur la différence (on ré-échantillonne les items 1000 fois et on regarde si l'écart reste du même signe) ;
- plusieurs répétitions par config pour mesurer le bruit du juge ;
- un identifiant d'item (`request_id`, Q03) dans le rapport au lieu de la question tronquée ;
- les vraies valeurs de config dans les métadonnées (bug `get_int`) ;
- le nombre d'items qui **changent** entre runs, affiché en premier.

**Exemple** : avec 8 items, un écart de moyenne de 0.125 = un seul item. Le rapport devrait afficher « 1 item sur 8 a changé (Q03), IC 95 % de la différence : [-0.05 ; +0.30] → non concluant ».

**Effort** : 3 à 5 jours.

**Risque** : le coût. Répéter 3 fois en mode full multiplie les appels LLM par 3. Il faut un budget et peut-être réserver les répétitions aux décisions importantes.

### 5. Fiabiliser le juge et agrandir le dataset (le plus ambitieux)

**Problème résolu** : un juge non validé (même modèle que le générateur, B1 n°5) et un dataset de 8 items, dont 2 seulement réagissent au réglage testé.

**Ce qu'on fait** :
- **juge séparé** : section `[Judge]` dans la config, modèle différent et plus puissant que le générateur ;
- **calibration du juge** : sur 100-200 jugements, comparer le juge à des annotations humaines (les retours utilisateurs de D3 + un échantillon annoté par les experts). Mesurer l'accord (kappa de Cohen). Si l'accord est faible sur une métrique, on ne la suit plus ;
- **agrandir le golden dataset** à 150-300 items, **stratifiés** par type de question (`source_tag` : montant, éligibilité, délai, démarche, multi-documents, hors périmètre), avec des cas tirés de la prod (les messages notés négativement) ;
- **intégration CI** (voir D4).

**Effort** : plusieurs semaines, dont beaucoup de temps d'experts métier (le facteur limitant).

**Risque** : disponibilité des experts, coût des annotations, et dérive : le dataset doit être maintenu quand la réglementation change (un montant 2026 devient faux en 2027).

---

## D2. Code de l'évolution n°1

Environ 20 lignes, réparties dans 3 fichiers.

```python
# --- run_eval.py:41 -------------------------------------------------------
# Avant : str(item.expected_output) -> "None" si la cellule est vide
expected_response = item.expected_output or None

# --- run_eval.py:72 -------------------------------------------------------
# Arguments nommés : l'inversion response/reference devient impossible
scores = await score_with_ragas(
    metrics,
    query=prompt,
    chunks=ctx_texts,
    response=response,
    reference=expected_response,
    max_concurrency=metrics_concurrency,
)

# --- ragas_metrics.py, dans score_with_ragas ------------------------------
REFERENCE_METRICS = {"context_precision", "context_recall",
                     "answer_correctness", "answer_accuracy"}

async def run_metric(m):
    if m.name in REFERENCE_METRICS and not sample.reference:
        return None                      # pas de référence : non applicable
    async with sem:
        return await call_ascore(m, sample)

# ... puis, à la place de la conversion actuelle :
value = float(r) if r is not None else None
if value is not None and math.isnan(value):
    log_step(f"Métrique '{m.name}' : nan", "⚠️")
    value = None                         # compté dans "manquants" du rapport
scores[m.name] = value

# --- collections.py:300, ContextPrecision ---------------------------------
denominator = sum(verdict_list) + 1e-10  # nb de chunks utiles (formule ragas)
```

Et deux tests qui auraient attrapé les bugs :

```python
def test_context_precision_bon_chunk_en_premier():
    cp = ContextPrecision(llm=None)
    assert round(cp._calculate_average_precision([1, 0, 0, 0]), 3) == 1.0

async def test_reference_bien_transmise():
    spy = SpyMetric()  # métrique factice qui enregistre ce qu'elle reçoit
    await score_with_ragas([spy], query="q", chunks=["c"], response="R", reference="REF")
    assert spy.received["reference"] == "REF"
    assert spy.received["response"] == "R"
```

**Points à savoir défendre** :
- pourquoi des arguments nommés plutôt que remettre les positions dans le bon ordre : parce que deux `str` côte à côte, c'est exactement le genre d'erreur que personne ne voit en relecture. Les noms rendent l'appel lisible et résistant à un changement de signature ;
- pourquoi `nan → None` et pas `nan → 0` : 0 serait un vrai score (« la réponse est mauvaise »), alors que `nan` veut dire « pas mesurable ». Le rapport compte déjà les `None` dans la colonne « manquants », donc rien n'est caché ;
- pourquoi sauter les métriques à référence quand il n'y en a pas (Q08) : noter contre le texte "None" produit un chiffre qui n'a aucun sens et tire la moyenne vers le bas.

---

## D3. Exploiter les retours utilisateurs (`document_evaluations.jsonl`)

### Ce que contiennent ces données

Chaque ligne = un agent qui note un **chunk cité** dans une réponse du chatbot : pouce haut / bas, avec un commentaire optionnel. On a le document (`document_id` + `chunk_index`), la conversation (`chat_id`, `message_id`) et la note (`eval`).

En traduisant les `document_id` avec `corpus.jsonl` (101 = PRN, 103 = CGPE, 104 = MAJO, 105 = ASA, 106 = DELAI, 107 = SITU) :

| id | chunk | note | ce que ça raconte |
|---|---|---|---|
| 1 | SITU#0 | pertinent | |
| 2 | SITU#1 | non_pertinent | « parle de déménagement, rien à voir avec ma reprise d'emploi » → c'est exactement Q07 |
| 3 | SITU#1 | NON PERTINENT | même avis, autre conversation |
| 4 | DELAI#1 | bof | « utile mais pas pour cette question » |
| 5 | ASA#0 | pertinent | « le bon montant est bien là, c'est la réponse qui est fausse » → exactement Q04 |
| 6 | CGPE#0 | pertinent | |
| 7 | CGPE#0 | pertinent | doublon exact de la ligne 6 |
| 8 | doc 104 (MAJO), pas de chunk | manquant | « il manque la circulaire sur la majoration parent isolé » → exactement Q03 |

On voit que ces retours **confirment les trois problèmes trouvés dans l'évaluation** (Q03, Q04, Q07). C'est un bon argument pour les brancher sur le pipeline.

### Comment les exploiter

**1. Mesurer l'accord juge / humains (le plus direct).**
Le juge rend déjà un verdict « chunk utile ou non » (ContextPrecision, un verdict par chunk). Pour chaque retour :
- retrouver la question et la réponse via `message_id` (dans les traces Langfuse de prod) ;
- faire juger le même chunk pour la même question par le juge ;
- comparer : humain pertinent / non pertinent vs juge 1 / 0.

On obtient une matrice de confusion, et un **kappa de Cohen** (accord corrigé du hasard ; > 0.6 = accord correct). Si le juge dit « utile » là où les agents disent « non pertinent » (SITU#1 pour une reprise d'emploi), on sait que ses scores de retrieval sont trop optimistes.

**2. Alimenter le golden dataset.**
Chaque retour négatif est un cas réel où le RAG a mal fonctionné. On le transforme en item : la question (via `message_id`), et un expert écrit la réponse attendue et les chunks attendus. Le retour n°8 fournit même directement un `expected_retrieved_context` (MAJO).

**3. Séparer retrieval et génération.**
Le retour n°5 montre qu'un chunk pertinent peut accompagner une réponse fausse. Un « pertinent » sur le chunk ne veut donc pas dire « bonne réponse ». Il faut garder les deux signaux séparés : retours sur les chunks → qualité du retrieval ; feedback sur la réponse (pouce sur le message) → qualité de la génération.

**4. Suivre un indicateur en production.**
Taux de chunks cités jugés non pertinents par semaine, par type de question : un indicateur de dérive du retrieval qui ne coûte aucun appel LLM.

### Les difficultés dans ces données

- **Étiquettes non normalisées** : `pertinent`, `non_pertinent`, `NON PERTINENT`, `bof`, `manquant`. Le champ `eval` est un `VARCHAR(64)` libre (`document_evaluation.py:45`). Il faut une énumération côté API, et un mapping pour l'historique (`bof` = ? partiellement pertinent ?).
- **`manquant` n'est pas le même type de signal** : ce n'est pas l'avis sur un chunk cité, c'est un chunk **absent** (un signal de recall, pas de precision). Il n'a d'ailleurs pas de `chunk_index`. Il faut le traiter à part.
- **Doublons** : les lignes 6 et 7 sont identiques (même chunk, même message, même note, à un jour d'écart). Double clic ? Il faut dédoublonner sur (`message_id`, `document_id`, `chunk_index`), sinon on donne deux fois plus de poids à certains avis.
- **Pas de question ni de réponse dans la ligne** : il faut la jointure via `message_id` avec les traces de conversation. Si les traces de conversation sont purgées avant qu'on fasse la jointure, on perd le contexte : il faut une durée de rétention compatible.
- **Biais de sélection** :
  - seuls les chunks **cités** peuvent être notés : les chunks non récupérés ne reçoivent jamais d'avis (sauf `manquant`) ;
  - les agents notent surtout quand ça ne va pas : les retours surreprésentent les échecs ;
  - quelques agents très actifs peuvent peser lourd.
- **Volume** : 8 retours ici. Pour un kappa fiable, il en faut au moins 100 à 200, répartis sur les types de questions.
- **Avis subjectifs et contradictoires** : `bof` + « utile mais pas pour cette question » : deux agents peuvent noter différemment le même chunk. Il faudrait mesurer l'accord **entre humains** d'abord : si les humains ne sont pas d'accord entre eux, on ne peut pas demander au juge de l'être.
- **Stabilité des identifiants** : `chunk_index` dépend du découpage. Si on re-découpe le corpus, les anciens retours pointent vers d'autres textes. Il faut dater les retours par version d'indexation.
- **Données personnelles** : les commentaires sont du texte libre écrit par des agents d'un organisme social. Ils peuvent contenir des informations sur des usagers. Avant de les réutiliser (dans un dataset, ou envoyés à un juge LLM), il faut les filtrer (RGPD).

---

## D4. Intégration : CI et évaluation planifiée

### Ordre de grandeur du coût

Appels au juge par item, d'après `collections.py` :
- mode `fast` : ContextRelevance 2 + ContextRecall 1 + AnswerAccuracy 2 + Groundedness 2 = **~7 appels** ;
- mode `full` : + Faithfulness 2 + AnswerRelevancy 3 (+ embeddings) + ContextPrecision 1 par chunk (~5) + AnswerCorrectness 3 = **~20 appels**.

Plus 1 appel de génération par item. Donc :
- 50 items en `fast` ≈ 400 appels ;
- 200 items en `full` ≈ 4 000 appels, × 3 répétitions = 12 000 appels.

### (a) En CI, garde-fou à chaque merge request

**Quand** : pas sur toutes les MR. Seulement celles qui touchent au RAG : retriever, prompts, config `[Retriever]` / `[Response]`, modèle, pipeline d'indexation. Les autres MR ne lancent que les tests unitaires.

**Trois étages, du moins cher au plus cher** :
1. **Tests unitaires des métriques** (toujours, gratuit, quelques secondes) : formules, transmission des champs (les tests de D2). Ils auraient bloqué l'inversion dès le premier jour.
2. **Métriques de retrieval sans juge** (D1 n°2) sur tout le golden dataset : hit@k, recall@k. Aucun appel LLM, juste le retriever. Quelques secondes à quelques minutes.
3. **Évaluation LLM sur un sous-ensemble** : ~30 à 50 items, stratifiés (au moins un item de chaque `source_tag`, dont les cas hors périmètre et multi-documents), mode `fast`, `TEMPERATURE=0`. Une dizaine de minutes, quelques centaines d'appels.

**Seuils** : relatifs, pas absolus.
- On compare à une **baseline** : le dernier run de `main`, stocké.
- Seuil de tolérance basé sur le **bruit mesuré** : on lance la baseline 5 fois, on mesure l'écart-type de chaque métrique, et on bloque si la baisse dépasse 2 écarts-types (ou un seuil fixe du genre -0.05, à calibrer).
- **Garde-fous durs** (bloquants quoi qu'il arrive) : `nb_error = 0`, pas de `nan` inexpliqué, recall@5 des chunks attendus qui ne baisse pas, et les items « critiques » (montants, hors périmètre) ne passent pas de réussi à échoué.
- En dessous des seuils : simple **avertissement**, pas de blocage. Sinon l'équipe finit par ignorer la CI.

**Gestion du bruit** :
- `TEMPERATURE=0` pour la génération et le juge ;
- comparaison **appariée** item par item avec la baseline ;
- le rapport affiché dans la MR liste les items qui ont **changé** (ex. « Q03 : recall 1.0 → 0.5 »), pas seulement les moyennes ;
- cache : si ni le retriever ni le générateur n'ont changé pour un item, on réutilise les scores de la baseline.

### (b) Évaluation à froid planifiée

**Quand** : chaque nuit ou chaque semaine, et aussi **à chaque événement** qui change le système sans MR : ré-indexation du corpus, nouvelle circulaire ajoutée, nouvelle version du modèle côté fournisseur.

**Quoi** :
- **tout** le golden dataset (150-300 items), mode `full`, juge séparé et plus puissant ;
- **3 répétitions** pour mesurer la variance ;
- **intervalles de confiance** (bootstrap) sur chaque métrique ;
- résultats par **type de question** (`source_tag`) et pas seulement en moyenne globale : une baisse sur « multi_documents » peut être noyée dans la moyenne ;
- matrice d'abstention (D1 n°3) ;
- les sweeps de paramètres (distance, `NB_RESULTS`, reranker) se font ici, pas en CI.

**Suivi dans le temps** :
- tableau de bord Langfuse : courbe de chaque métrique par semaine, avec la version du modèle, du corpus et des métriques ;
- alerte si une métrique sort de son intervalle habituel ;
- **audit mensuel du juge** : un expert relit 20-30 jugements tirés au hasard, pour détecter une dérive du juge (surtout si le fournisseur met à jour le modèle) ;
- intégration des retours utilisateurs (D3) : accord juge / humains suivi dans le temps, nouveaux items ajoutés au dataset.

**Coût** : ~12 000 appels par semaine en mode complet avec répétitions. À chiffrer selon le prix du modèle juge ; si c'est trop cher, on garde 1 répétition en routine et 3 seulement avant une mise en production.

### Résumé CI vs planifiée

| | CI (merge request) | Planifiée |
|---|---|---|
| déclencheur | MR qui touche au RAG | nuit / semaine / ré-indexation / nouveau modèle |
| items | 30-50, stratifiés | tout le dataset (150-300) |
| métriques | tests unitaires + retrieval sans juge + `fast` | `full` + abstention + par type |
| répétitions | 1 (température 0) | 3 |
| seuils | relatifs à la baseline, calibrés sur le bruit | intervalles de confiance, tendances |
| but | bloquer une régression | suivre la qualité, choisir des réglages |
| durée | ~10 min | ~1 h |

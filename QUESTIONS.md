# Questions

Répondez dans `REPONSES.md`. Réponses courtes acceptées (listes, tableaux, pseudo-code).
Citez le code sous la forme `fichier:ligne`. Temps indicatif entre parenthèses.

## A. Compréhension (10 min)

**A1.** Décrivez le flux complet d'un run d'évaluation, de la ligne du golden dataset
jusqu'au score visible dans le rapport de comparaison. Pour chaque étape : fichier, fonction,
données en entrée et en sortie.

**A2.** Pour chacune des 4 métriques du mode `fast`, indiquez : les champs de l'échantillon
qu'elle utilise (`user_input`, `retrieved_contexts`, `response`, `reference`), ce que le juge
LLM doit produire, et la formule qui transforme ces jugements en score.

**A3.** Classez les 8 métriques en trois familles : qualité du **retrieval**, qualité de la
**génération**, qualité **bout-en-bout**. Lesquelles ne peuvent pas être calculées sans
réponse de référence ?

**A4.** Quelle est la différence entre `faithfulness` et `answer_relevancy` ? Donnez un
exemple de réponse qui a un bon score sur l'une et un mauvais sur l'autre.

## B. Lecture critique (10 min)

**B1.** Relevez les problèmes que vous voyez dans ce code : bugs, limites, choix discutables.
Pour chacun : emplacement, impact concret sur les chiffres produits, gravité (bloquant,
important, mineur). Classez-les du plus grave au moins grave.

**B2.** `run_sweep.py` sert à choisir la valeur de `MAX_SEMANTIC_DISTANCE`. En l'état,
feriez-vous confiance à la comparaison produite par `run_report.py` pour trancher ?
Justifiez.

**B3.** Un point sur le lancement d'une évaluation via l'API (`evalrag/api/`,
`evalrag/services/eval_launcher.py`) qui vous gênerait avant une mise en production.

## C. Raisonnement sur les données (10 à 15 min)

**C1.** Lisez [traces/trace_01_Q04_hallucination.md](traces/trace_01_Q04_hallucination.md).
La réponse générée contient un montant faux et une règle inventée.
- Recalculez à la main `faithfulness` et `context_precision` à partir des valeurs
  intermédiaires de la trace. Obtenez-vous les scores enregistrés ?
- Ces scores décrivent-ils correctement la qualité de cette réponse ? Sinon, expliquez
  pourquoi, et donnez les valeurs que vous attendriez.

**C2.** Lisez [traces/trace_02_Q05_Q06_abstentions.md](traces/trace_02_Q05_Q06_abstentions.md).
Q05 est une abstention **légitime** (question hors périmètre), Q06 une abstention
**illégitime** (la réponse est dans le contexte).
- Complétez les deux tableaux de scores à partir des jugements élémentaires.
- Quelles métriques distinguent les deux cas ? Lesquelles pénalisent à tort l'abstention
  légitime ? Qu'en concluez-vous pour l'interprétation des moyennes ?

**C3.** Q03 (voir `data/README.md`) demande de combiner deux documents. Que se passe-t-il
avec `MAX_SEMANTIC_DISTANCE=0.90`, puis `0.95` ? Quelle métrique vous permettrait de dire
que le problème vient du retriever et non du générateur ?

## D. Améliorations (5 à 10 min)

**D1.** Proposez 3 à 5 évolutions du pipeline d'évaluation, **classées de la plus simple à
la plus ambitieuse**, avec pour chacune : le problème résolu, l'effort estimé, le risque.

**D2.** Écrivez (code ou pseudo-code, une vingtaine de lignes) l'évolution que vous placez
en premier.

**D3.** Les utilisateurs notent en production les chunks cités (`data/document_evaluations.jsonl`).
Comment exploiteriez-vous ces retours dans l'évaluation (par exemple pour mesurer l'accord
entre le juge LLM et les humains) ? Quelles difficultés voyez-vous dans ces données ?

**D4.** Comment intégreriez-vous ce pipeline (a) en CI, comme garde-fou de non-régression à
chaque merge request, et (b) en évaluation à froid planifiée ? Seuils, taille du jeu, coût,
gestion du bruit.

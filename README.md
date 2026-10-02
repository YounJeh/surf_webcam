# Exercice AI Engineer : évaluation d'un RAG

## Contexte

Un organisme public de prestations sociales (anonymisé) opère un assistant conversationnel
RAG pour ses agents : l'agent pose une question en langage naturel, le système retrouve des
extraits de la documentation réglementaire interne, puis un LLM rédige la réponse.

Pour piloter la qualité, l'équipe a construit un **pipeline d'évaluation** :

1. des experts métier remplissent un **golden dataset** (question, réponse attendue, chunks attendus) ;
2. ce dataset est importé dans l'outil d'observabilité (Langfuse) ;
3. pour chaque question, le pipeline exécute le RAG réel (retrieval puis génération) ;
4. un **LLM juge** calcule des métriques de type RAGAS ;
5. les scores sont attachés à chaque trace, puis un rapport compare plusieurs runs
   (par exemple plusieurs réglages du retriever).

Ce dossier contient **une copie allégée et fidèle de ce code**. Seule l'infrastructure a été
remplacée par des **stubs locaux** (dossier `evalrag/stubs/`, chaque fichier marqué `[STUB]`) :
LLM juge, embeddings, base vectorielle, LLM de génération, Langfuse. La logique d'évaluation
(runner, calcul des métriques, agrégation) est celle du code en production. Toutes les données sont **fictives**.

## Ce qu'on attend de vous

Répondez aux questions de [QUESTIONS.md](QUESTIONS.md) dans un fichier `REPONSES.md`
(Markdown, 2 à 4 pages). Vous pouvez joindre un patch ou du pseudo-code.

- **Budget : 30 à 45 minutes.** Ne cherchez pas à tout couvrir : on préfère trois constats
  précis, localisés (`fichier:ligne`) et chiffrés, plutôt que dix constats survolés.
- **Vous n'avez pas besoin d'exécuter le code.** Tout est raisonnable en lecture, avec les
  traces statiques de `traces/` et la vue d'ensemble de `data/README.md`.
- Vos réponses serviront de base à un **entretien technique d'une heure**. Vous y
  défendrez vos choix, en citant le code.

## Ordre de lecture conseillé

| priorité | fichier | quoi |
|---|---|---|
| 1 | `evalrag/runners/run_eval.py` | le runner : un item = retrieval, génération, scoring, envoi des scores |
| 1 | `evalrag/metrics/ragas_metrics.py` | construction des métriques et appel de scoring |
| 1 | `evalrag/metrics/constants.py` | quelles métriques, quels modes (`fast`, `full`) |
| 2 | `evalrag/metrics/collections.py` | formules des 8 métriques (portage de `ragas==0.4.3`) |
| 2 | `traces/`, `data/README.md` | un exemple déroulé, les données |
| 3 | `evalrag/runners/run_report.py` | agrégation et comparaison de runs |
| 3 | `evalrag/runners/run_sweep.py` | balayage de configurations (plusieurs runs en parallèle) |
| 3 | `evalrag/services/eval_launcher.py`, `evalrag/api/` | lancement d'un run via l'API interne |
| 3 | `evalrag/dataset_import.py`, `evalrag/utils.py`, `evalrag/tracing/` | import du dataset, config de run, traçage |
| optionnel | `evalrag/stubs/` | infrastructure simulée : pas nécessaire pour répondre |

## Politique d'usage de l'IA

L'usage d'assistants IA (ChatGPT, Claude, Copilot...) est **autorisé** pour cette partie
asynchrone. En contrepartie :

- indiquez en fin de `REPONSES.md` quels outils vous avez utilisés et pour quoi ;
- en entretien, vous devrez **expliquer et défendre chaque affirmation**, sans outil,
  en citant le code. Une réponse que vous ne savez pas justifier ne comptera pas.

## (Optionnel) Exécution locale

Python 3.10 ou plus, **aucune dépendance** à installer, aucun accès réseau.

```bash
cd candidate
python -m evalrag.runners.run_eval --mode full --run-name essai
python -m evalrag.runners.run_sweep            # 3 runs, distances 0.85 / 0.90 / 0.95
python -m evalrag.runners.run_report --dataset_name golden_set_demo \
    --runs <nom_run_1> <nom_run_2> --score_names faithfulness context_recall
```

Les noms de run sont affichés en fin d'exécution (`[stub-langfuse] run enregistré : ...`).
Les sorties sont écrites dans `out/` (traces et scores au format JSON, rapports Markdown).
Les fichiers `evalrag/api/*.py` ne sont pas exécutables sans FastAPI et pydantic : ils sont
fournis pour lecture.

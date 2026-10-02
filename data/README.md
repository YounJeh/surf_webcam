# Données (100 % fictives)

Les prestations (PRN, ALES, CGPE, ASA), montants, circulaires, utilisateurs et identifiants
sont **inventés**. Ils ressemblent à de la documentation de prestations sociales, sans en
reproduire aucune.

| fichier | rôle | lu par |
|---|---|---|
| `golden_dataset.jsonl` | Golden dataset : export du fichier Excel rempli par les experts métier | `dataset_import.py` |
| `corpus.jsonl` | Base documentaire découpée en chunks (`chunk_id`, texte) | stubs |
| `retrieval_fixture.json` | Pour chaque question : chunks candidats et distance cosinus, avant filtrage | stub retriever |
| `generation_fixture.json` | Pour chaque question : réponse(s) que le LLM de génération a produites | stub générateur |
| `judge_annotations.json` | Jugements rejoués par le juge LLM stub (voir `evalrag/stubs/llm_judge.py`) | stub juge |
| `document_evaluations.jsonl` | Retours utilisateurs en production sur les chunks cités (pouce haut / bas) | rien pour l'instant |

## Vue d'ensemble des 8 items

Configuration par défaut : `MAX_SEMANTIC_DISTANCE=0.9`, `NB_RESULTS=5`. Un chunk entre
crochets est écarté par le filtre de distance.

| id | question (résumé) | réponse attendue (résumé) | chunks récupérés (distance) | réponse générée (résumé) | feedback humain |
|---|---|---|---|---|---|
| Q01 | Montant PRN au collège | 210 €, versée fin août | PRN#0 (.18), PRN#1 (.31), ALES#1 (.66), CGPE#1 (.74), DELAI#0 (.79) ; ASA#1 (.88) coupé par NB_RESULTS | 210 €, versée fin août aux familles déjà bénéficiaires | |
| Q02 | Étudiant de 27 ans, droit à l'ALES ? | Non (< 26 ans), sauf handicap (30 ans) | ALES#0 (.20), ALES#1 (.29), PRN#1 (.63), SITU#1 (.72), DELAI#0 (.85) | **deux variantes** : avec ou sans l'exception handicap | mitigé |
| Q03 | Parent isolé, 1 900 €/mois, montant CGPE ? | Tranche 2 : 310 €, majoré de 30 % : 403 € | CGPE#0 (.22), CGPE#1 (.35), SITU#0 (.58) ; [MAJO#0 (.91)], [ASA#1 (.94)], [PRN#0 (.96)] | 310 € (sans la majoration) ; 403 € si MAJO#0 est dans le contexte | |
| Q04 | Montant journalier ASA | 64,20 €/jour, 32,10 €/demi-journée, 22 jours/mois max | ASA#0 (.19), ASA#1 (.27), DELAI#0 (.70), PRN#1 (.81) | **68,50 €/jour**, 22 jours, « cumulable avec la PRN » | négatif |
| Q05 | Déclarer des revenus fonciers (impôts) | Hors périmètre : ne pas répondre, orienter vers l'administration fiscale | SITU#0 (.83), DELAI#0 (.87), ALES#1 (.89) ; [PRN#1 (.93)] | « Je ne dispose pas d'information... contactez l'administration fiscale » | positif |
| Q06 | Délai de versement après changement de situation | Mois suivant, au plus tard sous 30 jours | DELAI#1 (.16), DELAI#0 (.28), SITU#0 (.40), SITU#1 (.55), CGPE#1 (.77) | « Je ne sais pas » | négatif |
| Q07 | Reprise d'emploi à temps partiel, à déclarer ? | Oui, dans le mois, espace personnel, impact sur les aides | SITU#0 (.21), SITU#1 (.33), DELAI#1 (.45), DELAI#0 (.61), ASA#1 (.80) | Oui (correct) + paragraphe sur le déménagement | négatif |
| Q08 | Plafond de ressources ALES | *(cellule vide dans l'Excel, à compléter par le métier)* | ALES#1 (.17), ALES#0 (.30), PRN#1 (.52), CGPE#0 (.81) | 18 000 € par an | |

Colonne `expected_retrieved_context` du golden dataset : chunks qui, selon l'expert métier,
contiennent la réponse (ex. Q03 : `CGPE#0;MAJO#0`).

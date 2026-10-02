# Trace 02 : deux abstentions, Q05 et Q06

Fiche d'analyse statique. Pour chaque item : l'entrée, le contexte récupéré, la réponse générée, la référence, puis les **jugements élémentaires** rendus par le juge LLM à chaque sous-tâche. Les **scores finaux sont masqués** : c'est à vous de les déduire (question C2). Configuration par défaut (`MAX_SEMANTIC_DISTANCE=0.9`, `NB_RESULTS=5`). Données fictives.

## Q05 : question hors périmètre

- **Question** : Comment déclarer mes revenus fonciers sur ma déclaration d'impôt ?
- **Référence (réponse attendue)** : Cette question ne relève pas des prestations couvertes par la documentation. L'assistant doit indiquer qu'il ne peut pas répondre et orienter vers l'administration fiscale.
- **Chunks récupérés (ordre)** : SITU#0, DELAI#0, ALES#1

  - SITU#0 : [FICTIF] Déclarer un changement de situation. Tout changement de situation professionnelle (reprise d'emploi, y compris à temps partiel, perte d'emploi, changement de revenus) doit être déclaré dans le mois qui suit, depuis l'espace personnel en ligne. Il peut modifier le montant des aides.
  - DELAI#0 : [FICTIF] Délais de traitement. Une demande complète est traitée en 15 jours ouvrés en moyenne. Toute pièce manquante suspend le délai jusqu'à sa réception.
  - ALES#1 : [FICTIF] ALES : ressources et montant. Les ressources annuelles de l'étudiant ne doivent pas dépasser 18 000 €. Le montant mensuel est de 95 € en résidence universitaire et de 130 € dans le parc locatif privé.

- **Réponse générée** : Je ne dispose pas d'information sur la déclaration des revenus fonciers dans la documentation à ma disposition. Pour cette question, je vous invite à contacter l'administration fiscale.

Jugements élémentaires :

1. Décomposition de la **réponse générée** en affirmations : 
   - La documentation ne contient pas d'information sur la déclaration des revenus fonciers. (déductible du contexte : 0)
   - Pour la déclaration des revenus fonciers, il faut contacter l'administration fiscale. (déductible du contexte : 0)
2. Phrases de la **référence** et attribution au contexte :
   - La question ne relève pas des prestations couvertes par la documentation. (attribuable : 0)
   - L'assistant doit orienter vers l'administration fiscale. (attribuable : 0)
3. Questions régénérées depuis la **réponse générée** (noncommittal) :
   - Où trouver de l'information sur la déclaration des revenus fonciers ? (1)
   - Qui contacter pour déclarer des revenus fonciers ? (1)
   - La documentation traite-t-elle des revenus fonciers ? (1)
4. Utilité de chaque chunk pour obtenir la référence : SITU#0=0, DELAI#0=0, ALES#1=0
5. Notes AnswerAccuracy (0/2/4) : juge 1 = 4, juge 2 = 4
6. Notes ResponseGroundedness (0/1/2) : 0, 0
7. Notes ContextRelevance (0/1/2) : 0, 0

| métrique | score à prédire |
|---|---|
| faithfulness | ? |
| answer_relevancy | ? |
| context_recall | ? |
| context_precision | ? |
| answer_accuracy | ? |
| response_groundedness | ? |
| context_relevance | ? |

## Q06 : information présente dans la documentation

- **Question** : Après avoir déclaré un changement de situation, dans quel délai le nouveau montant est-il versé ?
- **Référence (réponse attendue)** : Après la déclaration d'un changement de situation, les droits sont recalculés et le nouveau montant est versé le mois suivant, au plus tard sous 30 jours.
- **Chunks récupérés (ordre)** : DELAI#1, DELAI#0, SITU#0, SITU#1, CGPE#1

  - DELAI#1 : [FICTIF] Versement après changement de situation. Après la déclaration d'un changement de situation, les droits sont recalculés et le nouveau montant est versé le mois suivant, au plus tard sous 30 jours.
  - DELAI#0 : [FICTIF] Délais de traitement. Une demande complète est traitée en 15 jours ouvrés en moyenne. Toute pièce manquante suspend le délai jusqu'à sa réception.
  - SITU#0 : [FICTIF] Déclarer un changement de situation. Tout changement de situation professionnelle (reprise d'emploi, y compris à temps partiel, perte d'emploi, changement de revenus) doit être déclaré dans le mois qui suit, depuis l'espace personnel en ligne. Il peut modifier le montant des aides.
  - SITU#1 : [FICTIF] Déclarer un déménagement. En cas de déménagement, la nouvelle adresse doit être déclarée dans le mois, avec la date d'entrée dans le logement et, le cas échéant, le nouveau bail.
  - CGPE#1 : [FICTIF] CGPE : conditions générales. Le complément est versé à partir du mois suivant le début de la garde, sur présentation du contrat de travail de l'assistant maternel. Il n'est pas versé pendant les périodes de congé parental à temps plein.

- **Réponse générée** : Je ne sais pas. Je n'ai pas trouvé cette information dans les documents à ma disposition.

Jugements élémentaires :

1. Décomposition de la **réponse générée** en affirmations : aucune affirmation extraite.
2. Phrases de la **référence** et attribution au contexte :
   - Après une déclaration de changement de situation, les droits sont recalculés. (attribuable : 1)
   - Le nouveau montant est versé le mois suivant la déclaration. (attribuable : 1)
   - Le nouveau montant est versé au plus tard sous 30 jours. (attribuable : 1)
3. Questions régénérées depuis la **réponse générée** (noncommittal) :
   - Quelle information l'assistant n'a-t-il pas trouvée ? (1)
   - L'assistant connaît-il la réponse ? (1)
   - Que disent les documents ? (1)
4. Utilité de chaque chunk pour obtenir la référence : DELAI#1=1, DELAI#0=0, SITU#0=0, SITU#1=0, CGPE#1=0
5. Notes AnswerAccuracy (0/2/4) : juge 1 = 0, juge 2 = 0
6. Notes ResponseGroundedness (0/1/2) : 0, 0
7. Notes ContextRelevance (0/1/2) : 2, 2

| métrique | score à prédire |
|---|---|
| faithfulness | ? |
| answer_relevancy | ? |
| context_recall | ? |
| context_precision | ? |
| answer_accuracy | ? |
| response_groundedness | ? |
| context_relevance | ? |

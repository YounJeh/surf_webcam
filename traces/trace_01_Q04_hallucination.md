# Trace 01 : Q04, montant journalier de l'Allocation Soutien Aidant

Trace statique d'une exécution du code fourni, **tel quel**, en mode `full`, configuration par défaut (`MAX_SEMANTIC_DISTANCE=0.9`, `NB_RESULTS=5`). Les valeurs ci-dessous ont été relevées pendant l'exécution (journal du juge stub). Toutes les données sont fictives.

## 1. Entrée (item du golden dataset)

- `item.input` : Quel est le montant journalier de l'Allocation Soutien Aidant ?
- `item.expected_output` : L'ASA est de 64,20 € par journée d'aide (32,10 € par demi-journée), dans la limite de 22 jours par mois.

## 2. Retrieval

Candidats renvoyés par la recherche, puis filtre `distance <= 0.9` et coupe à 5 :

| rang | chunk | distance | conservé |
|---|---|---|---|
| 1 | ASA#0 | 0.19 | oui |
| 2 | ASA#1 | 0.27 | oui |
| 3 | DELAI#0 | 0.70 | oui |
| 4 | PRN#1 | 0.81 | oui |

Texte des chunks conservés (`ctx_texts`, dans l'ordre) :

- **ASA#0** : [FICTIF] Allocation Soutien Aidant (ASA). Montant : 64,20 € par journée d'aide et 32,10 € par demi-journée, dans la limite de 22 jours par mois.
- **ASA#1** : [FICTIF] ASA : bénéficiaires. L'ASA est ouverte aux personnes qui réduisent ou cessent temporairement leur activité pour aider un proche en perte d'autonomie. La demande est à renouveler tous les 12 mois.
- **DELAI#0** : [FICTIF] Délais de traitement. Une demande complète est traitée en 15 jours ouvrés en moyenne. Toute pièce manquante suspend le délai jusqu'à sa réception.
- **PRN#1** : [FICTIF] PRN : conditions. L'enfant doit être âgé de 6 à 18 ans au 31 décembre de l'année de rentrée. Les ressources du foyer ne doivent pas dépasser 32 000 € par an pour un enfant, plafond majoré de 20 % par enfant supplémentaire. La PRN n'est pas cumulable avec l'Allocation Soutien Aidant perçue au titre du même enfant.

## 3. Génération

Réponse renvoyée par `responder.generate_response(...)` :

> L'Allocation Soutien Aidant est de 68,50 € par jour, dans la limite de 22 jours par mois. Elle est cumulable avec la Prime de Rentrée Numérique.

## 4. Échantillon construit dans `score_with_ragas`

```text
user_input         = Quel est le montant journalier de l'Allocation Soutien Aidant ?
retrieved_contexts = [ASA#0, ASA#1, DELAI#0, PRN#1]
response           = L'ASA est de 64,20 € par journée d'aide (32,10 € par demi-journée), dans la limite de 22 jours par mois.
reference          = L'Allocation Soutien Aidant est de 68,50 € par jour, dans la limite de 22 jours par mois. Elle est cumulable avec la Prime de Rentrée Numérique.
```

## 5. Appels au juge et valeurs intermédiaires

### faithfulness

Étape 1, `statement_generator(question, answer=sample.response)` :

- L'ASA est de 64,20 € par journée d'aide.
- L'ASA est de 32,10 € par demi-journée.
- L'ASA est versée dans la limite de 22 jours par mois.

Étape 2, `nli_statement(context=ctx_texts joints, statements)` :

| affirmation | verdict |
|---|---|
| L'ASA est de 64,20 € par journée d'aide. | 1 |
| L'ASA est de 32,10 € par demi-journée. | 1 |
| L'ASA est versée dans la limite de 22 jours par mois. | 1 |

Score enregistré : **faithfulness = 1.000**

### answer_relevancy

3 appels `answer_relevance(response=sample.response)` (strictness=3) :

| question régénérée | noncommittal |
|---|---|
| Quel est le montant de l'ASA par journée d'aide ? | 0 |
| Combien vaut l'ASA pour une demi-journée ? | 0 |
| Combien de jours d'ASA peut-on percevoir par mois ? | 0 |

Cosinus (embeddings) de chaque question régénérée avec `user_input`, puis moyenne x `int(not all_noncommittal)`.

Score enregistré : **answer_relevancy = 0.250**

### context_precision

Un appel `context_precision(question, context=chunk, answer=sample.reference)` par chunk, dans l'ordre du retrieval :

| rang | chunk | verdict |
|---|---|---|
| 1 | ASA#0 | 1 |
| 2 | ASA#1 | 0 |
| 3 | DELAI#0 | 0 |
| 4 | PRN#1 | 0 |

Score enregistré : **context_precision = 0.250**

### context_recall

`context_recall(question, context=ctx_texts joints, answer=sample.reference)` :

| phrase | attributed |
|---|---|
| L'Allocation Soutien Aidant est de 68,50 € par jour. | 0 |
| L'Allocation Soutien Aidant est versée dans la limite de 22 jours par mois. | 1 |
| L'Allocation Soutien Aidant est cumulable avec la Prime de Rentrée Numérique. | 0 |

Score enregistré : **context_recall = 0.333**

### answer_correctness

Affirmations de `sample.response` et de `sample.reference` (2 appels `statement_generator`), puis classification :

- TP (1) : L'Allocation Soutien Aidant est versée dans la limite de 22 jours par mois.
- FP (2) : L'ASA est de 64,20 € par journée d'aide. ; L'ASA est de 32,10 € par demi-journée.
- FN (2) : L'Allocation Soutien Aidant est de 68,50 € par jour. ; L'Allocation Soutien Aidant est cumulable avec la Prime de Rentrée Numérique.

Poids : 0.75 x F1 factuel + 0.25 x cosinus(embedding(response), embedding(reference)).

Score enregistré : **answer_correctness = 0.359**

### answer_accuracy

| juge | user_answer | reference_answer | note |
|---|---|---|---|
| 1 | L'ASA est de 64,20 € par journée d'aide (32,10 ... | L'Allocation Soutien Aidant est de 68,50 € par ... | 2 |
| 2 | L'Allocation Soutien Aidant est de 68,50 € par ... | L'ASA est de 64,20 € par journée d'aide (32,10 ... | 0 |

Score enregistré : **answer_accuracy = 0.250**

### response_groundedness

`response_groundedness(response=sample.response, context)`, deux juges : notes 2, 2 (sur 2).

Score enregistré : **response_groundedness = 1.000**

### context_relevance

`context_relevance(user_input, context)`, deux juges : notes 2, 2 (sur 2).

Score enregistré : **context_relevance = 1.000**

## 6. Scores envoyés à Langfuse pour cette trace

| score | valeur |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.250 |
| context_precision | 0.250 |
| context_recall | 0.333 |
| answer_correctness | 0.359 |
| answer_accuracy | 0.250 |
| response_groundedness | 1.000 |
| context_relevance | 1.000 |
| nb_error | 0 |

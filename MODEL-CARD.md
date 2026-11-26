# Model Card

This repo ships two trained artifacts from one pipeline.

## 1. `models/bbc-topic-model.joblib` (the working model)
- **Task:** classify a news article's text into one of 5 topics (business, entertainment,
  politics, sport, tech).
- **Model:** TF-IDF (1–2 grams, 30k features) + `LinearSVC`. Artifact ≈ 2.3 MB.
- **Data:** BBC News, 2,225 articles, CC0.
- **Measured performance:** test accuracy 0.989, macro-F1 0.989; 5-fold CV macro-F1
  0.981 ± 0.005.
- **Intended use:** topic tagging of English news-style text; a reference for small,
  CPU-only text classification.
- **Limitations:** 5 fixed topics; 2004–2005 BBC style; not a general-purpose classifier.

## 2. `models/tickets-baseline-model.joblib` (the negative-result model)
- **Task:** predict support-ticket type (5) and priority (4) from ticket text.
- **Model:** same pipeline as above.
- **Measured performance:** at chance (Type 0.217 vs floor 0.207; Priority 0.245 vs
  floor 0.259).
- **Intended use:** **none in production.** It is included as evidence of the negative
  result — the public dataset's labels are synthetic/random and carry no learnable signal.
- **How to get a useful ticket model:** retrain the same pipeline on real historical tickets
  (text + the type/priority agents actually assigned).

## Shared details
- **Inputs:** raw text only. No demographic or PII fields are used as features.
- **Metric philosophy:** every score is reported against a chance floor; macro-F1 alongside
  accuracy.
- **Training compute:** CPU, seconds to a couple of minutes. No GPU required.

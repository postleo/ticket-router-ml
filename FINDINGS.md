# Findings (summary)

**Question:** can a standard text classifier predict support-ticket type and priority from
ticket text — and how does the same method behave on data with real signal?

**Setup:** one pipeline — TF-IDF (word 1–2 grams, up to 30k features, sublinear term
frequency) + LinearSVC, one classifier per target column, class-weight balanced — run on both
datasets with a single config change (`DATASET=tickets` vs `DATASET=bbc`). Validation is
stratified 5-fold cross-validation on the training split plus a held-out 20% test set, with
the vectorizer fit inside each fold so nothing leaks from test to train. Every score is read
against a chance floor: the larger of the most-frequent-class rate and 1 / number-of-classes.

**Data:**

- Customer Support Tickets — 8,469 rows; input is subject + description; targets are type
  (5 classes) and priority (4 classes). The classes are near-evenly split (type 1,634–1,752
  per class; priority 2,063–2,192 per class), and all 8,469 descriptions contain the literal
  placeholder `{product_purchased}`.
- BBC News — 2,225 articles in 5 topics (sport 511, business 510, politics 417, tech 401,
  entertainment 386); input is the article text; target is the topic.

**Result:**

| Dataset (same pipeline) | Target | Classes | Chance floor | Model accuracy | Macro-F1 | Verdict |
|-------------------------|--------|--------:|-------------:|---------------:|---------:|---------|
| Customer support tickets | Type | 5 | 0.207 | 0.217 | 0.217 | at chance |
| Customer support tickets | Priority | 4 | 0.259 | 0.245 | 0.245 | at/below chance |
| BBC News | Topic | 5 | 0.229 | **0.989** | 0.989 | strong signal |

**Detail:**

- BBC News: 5-fold cross-validation gives macro-F1 0.981 ± 0.005, so the 0.989 held-out
  result holds across folds rather than coming from one split. Per-class F1 is high across all
  five topics, and the confusion that remains is small and sits between related topics. Full
  per-class report and confusion matrix: [`results/RESULTS-bbc.md`](results/RESULTS-bbc.md).
- Tickets: both targets land within about 0.01–0.02 of the chance floor, with priority a
  touch below it. Per-class F1 is flat near 1 / number-of-classes (about 0.20 for type, 0.25
  for priority), and the confusion matrix is an even scatter — the pattern of a model with
  nothing in the text to key on. Full report:
  [`results/RESULTS-tickets.md`](results/RESULTS-tickets.md).

**Takeaways:**

1. The ticket model is at chance because that dataset's labels are near-uniformly random and
   its descriptions are templated (every row contains `{product_purchased}`). The text does
   not encode the labels, so nothing can predict them. This is a data property, not a model
   fault — a larger or deeper model would not move the number.
2. The identical pipeline reaches 98.9% on BBC News (CV 0.981 ± 0.005), proving the method
   works when the data carries signal.
3. Practical lesson: a ticket router is feasible, but only on real historical tickets — the
   text plus the type and priority agents actually assigned. The same pipeline applies with
   one config entry, and the BBC run shows what it produces once the data carries signal.
4. Always read a score against its chance floor: 25% accuracy means nothing until you know the
   floor is 26%. That is why the chance floor and macro-F1 are reported next to accuracy.

# Results (measured)

All numbers below are measured in the sandbox on CPU. Each model score is shown next to a **chance baseline** (`most_frequent`). A model is only meaningful if it clearly beats that baseline.


## Ticket Type  (5 classes)

| metric | chance (most_frequent) | TF-IDF + LinearSVC |
|--------|------------------------|-----------------|
| accuracy | 0.207 | 0.217 |
| macro-F1 | 0.069 | 0.217 |

Cross-val (train, 5-fold): macro-F1 0.194 ± 0.011, accuracy 0.195

```
                      precision    recall  f1-score   support

     Billing inquiry       0.21      0.21      0.21       327
Cancellation request       0.22      0.22      0.22       339
     Product inquiry       0.21      0.21      0.21       328
      Refund request       0.21      0.21      0.21       351
     Technical issue       0.24      0.23      0.23       349

            accuracy                           0.22      1694
           macro avg       0.22      0.22      0.22      1694
        weighted avg       0.22      0.22      0.22      1694

confusion matrix:
[[69 67 67 64 60]
 [74 75 66 68 56]
 [58 73 70 60 67]
 [67 68 70 74 72]
 [66 61 63 79 80]]
```

## Ticket Priority  (4 classes)

| metric | chance (most_frequent) | TF-IDF + LinearSVC |
|--------|------------------------|-----------------|
| accuracy | 0.259 | 0.245 |
| macro-F1 | 0.103 | 0.245 |

Cross-val (train, 5-fold): macro-F1 0.250 ± 0.010, accuracy 0.250

```
              precision    recall  f1-score   support

    Critical       0.23      0.24      0.23       426
        High       0.26      0.25      0.25       417
         Low       0.22      0.23      0.23       413
      Medium       0.27      0.27      0.27       438

    accuracy                           0.24      1694
   macro avg       0.25      0.24      0.24      1694
weighted avg       0.25      0.24      0.25      1694

confusion matrix:
[[101  99 108 118]
 [100 103 118  96]
 [112  99  94 108]
 [122  98 101 117]]
```

## Reading these numbers

- **Ticket Type**: test accuracy 0.217 vs chance floor 0.207 (lift +0.010) → at/near chance (no learnable signal in this public data).
- **Ticket Priority**: test accuracy 0.245 vs chance floor 0.259 (lift -0.014) → at/near chance (no learnable signal in this public data).

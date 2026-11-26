# Results (measured)

All numbers below are measured in the sandbox on CPU. Each model score is shown next to a **chance baseline** (`most_frequent`). A model is only meaningful if it clearly beats that baseline.


## labels  (5 classes)

| metric | chance (most_frequent) | TF-IDF + LinearSVC |
|--------|------------------------|-----------------|
| accuracy | 0.229 | 0.989 |
| macro-F1 | 0.075 | 0.989 |

Cross-val (train, 5-fold): macro-F1 0.981 ± 0.005, accuracy 0.981

```
               precision    recall  f1-score   support

     business       1.00      0.97      0.99       102
entertainment       0.97      1.00      0.99        77
     politics       0.99      0.98      0.98        84
        sport       0.99      1.00      1.00       102
         tech       0.99      1.00      0.99        80

     accuracy                           0.99       445
    macro avg       0.99      0.99      0.99       445
 weighted avg       0.99      0.99      0.99       445

confusion matrix:
[[ 99   1   1   1   0]
 [  0  77   0   0   0]
 [  0   1  82   0   1]
 [  0   0   0 102   0]
 [  0   0   0   0  80]]
```

## Reading these numbers

- **labels**: test accuracy 0.989 vs chance floor 0.229 (lift +0.760) → beats chance — real signal.

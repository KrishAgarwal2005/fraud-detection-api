| Model | Val PR-AUC | Test PR-AUC | Threshold | Precision | Recall | F1 | Frauds caught | False alarms |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (class weights) | 0.7513 | 0.7387 | 0.999 | 0.6548 | 0.7333 | 0.6918 | 55 of 75 | 29 |
| Logistic Regression (SMOTE) | 0.7681 | 0.7496 | 0.999 | 0.6629 | 0.7867 | 0.7195 | 59 of 75 | 30 |
| Random Forest (class weights) | 0.7627 | 0.8138 | 0.26 | 0.9194 | 0.76 | 0.8321 | 57 of 75 | 5 |
| XGBoost (scale_pos_weight) | 0.7748 | 0.7921 | 0.78 | 0.8507 | 0.76 | 0.8028 | 57 of 75 | 10 |

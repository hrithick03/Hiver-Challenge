# Headline Benchmark Summary

Evaluated on the 200-sample hand-labelled Golden Set:

| System                     | Intent Acc   |   Intent Macro F1 |   Escalation F1 |   Escalation Recall |   Cost Penalty |   ROUGE-L |   Judge Grounded |   Judge Voice |   Judge Privacy |   Judge Overall |
|----------------------------|--------------|-------------------|-----------------|---------------------|----------------|-----------|------------------|---------------|-----------------|-----------------|
| Baseline 1 (Trivial)       | 17.5%        |             0.05  |           0     |               0     |           1.48 |      11.1 |             4    |          5    |            4.12 |            4.28 |
| Baseline 2 (Simple)        | 58.0%        |             0.584 |           0.469 |               0.39  |           0.98 |       9.2 |             4.38 |          5    |            4.12 |            4.12 |
| Apple Support Agent (Ours) | 61.5%        |             0.635 |           0.578 |               0.441 |           0.85 |      14.1 |             4.66 |          4.71 |            4.46 |            4.48 |

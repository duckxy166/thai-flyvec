# GloVe baseline (dim=300, epochs=50, lr=0.05, x_max=100.0, alpha=0.75)

Seeds: 0, 1, 2. Evaluated on the same common pairs (WordSim 212, SimLex 743) and splits.

| model | w | TH-WordSim-353 ρ | TH-SimLex-999 ρ | wisesight accuracy | wisesight macro-F1 | train_minutes | size_mb |
|---|---|---|---|---|---|---|---|
| GloVe | 11 | 0.496 ± 0.009 | 0.442 ± 0.003 | 0.620 ± 0.003 | 0.413 ± 0.005 | 2.031 ± 0.026 | 22.89 |

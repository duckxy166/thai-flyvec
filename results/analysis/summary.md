# What FlyVec is good at (issue #58)

Mean ± std over seeds, w = 11. Written by `scripts/run_analysis.py`.

## H1 rare words: th-wordsim-353 ρ by frequency of the rarer word

rare 71 pairs (count 28-191), mid 70 pairs (count 191-446), frequent 71 pairs (count 446-8708)

| model | rare | mid | frequent | oov |
|---|---|---|---|---|
| FlyVec k = 8 | 0.312 ± 0.063 | 0.439 ± 0.071 | 0.545 ± 0.030 |  |
| FlyVec k = 16 | 0.406 ± 0.095 | 0.482 ± 0.021 | 0.396 ± 0.075 |  |
| FlyVec k = 32 | 0.397 ± 0.038 | 0.522 ± 0.059 | 0.377 ± 0.068 |  |
| word2vec binary k = 8 | 0.220 ± 0.111 | 0.293 ± 0.078 | 0.288 ± 0.014 |  |
| word2vec binary k = 16 | 0.299 ± 0.095 | 0.419 ± 0.071 | 0.387 ± 0.088 |  |
| word2vec binary k = 32 | 0.393 ± 0.138 | 0.510 ± 0.035 | 0.485 ± 0.070 |  |
| word2vec | 0.507 ± 0.004 | 0.578 ± 0.011 | 0.716 ± 0.020 |  |
| fastText | 0.645 ± 0.012 | 0.596 ± 0.009 | 0.723 ± 0.007 |  |
| fastText + subword OOV | 0.645 ± 0.012 | 0.596 ± 0.009 | 0.723 ± 0.007 | 0.512 ± 0.006 |

## H1 rare words: th-simlex-999 ρ by frequency of the rarer word

rare 248 pairs (count 27-129), mid 249 pairs (count 129-417), frequent 246 pairs (count 417-32863)

| model | rare | mid | frequent | oov |
|---|---|---|---|---|
| FlyVec k = 8 | 0.293 ± 0.027 | 0.343 ± 0.052 | 0.560 ± 0.041 |  |
| FlyVec k = 16 | 0.298 ± 0.015 | 0.277 ± 0.014 | 0.573 ± 0.036 |  |
| FlyVec k = 32 | 0.281 ± 0.072 | 0.307 ± 0.031 | 0.504 ± 0.021 |  |
| word2vec binary k = 8 | 0.284 ± 0.069 | 0.306 ± 0.018 | 0.405 ± 0.091 |  |
| word2vec binary k = 16 | 0.391 ± 0.012 | 0.419 ± 0.049 | 0.478 ± 0.036 |  |
| word2vec binary k = 32 | 0.408 ± 0.014 | 0.469 ± 0.015 | 0.542 ± 0.040 |  |
| word2vec | 0.425 ± 0.003 | 0.555 ± 0.003 | 0.623 ± 0.004 |  |
| fastText | 0.481 ± 0.002 | 0.574 ± 0.005 | 0.631 ± 0.002 |  |
| fastText + subword OOV | 0.481 ± 0.002 | 0.574 ± 0.005 | 0.631 ± 0.002 | 0.505 ± 0.007 |

## H2 OOV-heavy text: wisesight test macro-F1 by share of OOV tokens

0% 740 texts, 0-10% 524 texts, 10-20% 737 texts, >20% 660 texts, all 2671 texts

| model | 0% | 0-10% | 10-20% | >20% | all |
|---|---|---|---|---|---|
| FlyVec k = 8 | 0.427 ± 0.020 | 0.354 ± 0.010 | 0.382 ± 0.023 | 0.457 ± 0.003 | 0.417 ± 0.010 |
| FlyVec k = 16 | 0.441 ± 0.019 | 0.341 ± 0.009 | 0.386 ± 0.026 | 0.471 ± 0.015 | 0.424 ± 0.013 |
| FlyVec k = 32 | 0.436 ± 0.022 | 0.351 ± 0.015 | 0.365 ± 0.018 | 0.495 ± 0.022 | 0.423 ± 0.009 |
| word2vec binary k = 8 | 0.414 ± 0.032 | 0.325 ± 0.012 | 0.376 ± 0.010 | 0.476 ± 0.024 | 0.411 ± 0.017 |
| word2vec binary k = 16 | 0.421 ± 0.029 | 0.335 ± 0.011 | 0.375 ± 0.017 | 0.461 ± 0.033 | 0.411 ± 0.018 |
| word2vec binary k = 32 | 0.423 ± 0.005 | 0.346 ± 0.020 | 0.377 ± 0.028 | 0.489 ± 0.018 | 0.421 ± 0.010 |
| word2vec | 0.435 ± 0.014 | 0.376 ± 0.006 | 0.381 ± 0.004 | 0.481 ± 0.004 | 0.426 ± 0.005 |
| fastText | 0.437 ± 0.010 | 0.360 ± 0.012 | 0.377 ± 0.002 | 0.484 ± 0.008 | 0.423 ± 0.007 |
| fastText + subword OOV | 0.429 ± 0.010 | 0.389 ± 0.006 | 0.397 ± 0.015 | 0.438 ± 0.012 | 0.423 ± 0.007 |

## H3 topic classification: prachathai-67k test (12 topics, multi-label)

| model | macro_f1 | micro_f1 |
|---|---|---|
| FlyVec k = 8 | 0.477 ± 0.002 | 0.592 ± 0.002 |
| FlyVec k = 16 | 0.482 ± 0.004 | 0.597 ± 0.004 |
| FlyVec k = 32 | 0.484 ± 0.004 | 0.597 ± 0.002 |
| word2vec binary k = 8 | 0.454 ± 0.004 | 0.580 ± 0.001 |
| word2vec binary k = 16 | 0.469 ± 0.005 | 0.587 ± 0.002 |
| word2vec binary k = 32 | 0.475 ± 0.004 | 0.590 ± 0.002 |
| word2vec | 0.480 ± 0.000 | 0.594 ± 0.001 |
| fastText | 0.483 ± 0.002 | 0.596 ± 0.001 |

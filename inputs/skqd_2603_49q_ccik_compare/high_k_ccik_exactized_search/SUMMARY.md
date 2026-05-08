# High-K Standard CCIK + Exactized Routine Search

Target absolute error: `1.0e-08`

Control standard seed:
- dimension `17`, Kprop `1000`, Kstore `1000`, nroot `5`, ncycles `16`
- projected determinant count `2051`
- absolute error `1.978149257308e-01`

Best exactized branch:
- branch `frontier_seed01_top128` from `frontier` seed `1`
- seed settings: dimension `17`, Kprop `1000`, Kstore `1000`, nroot `5`, ncycles `16`, refine_topn `128`
- best stage `exactized_pool2`
- best determinant count `37483`
- best absolute error `3.732694153769e-09`
- stop reason `target_reached`

First sub-1e-8 result:
- determinant count `37483`
- stage `exactized_pool2`

Paper comparison:
- candidate determinant count `37483` vs paper `244296`
- candidate error `3.732694153769e-09` vs paper `2.426651666758e-01`
- candidate does beat the paper at the same-or-larger determinant count

Comparison against the existing local high-accuracy run:
- existing local reference: pool `130013`, error `3.732438358384e-09`
- campaign best determinant count is `smaller by `3.469x`` relative to pool `130013`

Artifacts:
- `data/standard_ccik_runs.csv`
- `data/standard_ccik_frontier.csv`
- `data/exactized_routine_runs.csv`
- `data/winning_strategy.json`
- `ccik_exactized_high_k_strategy.png`
- `ccik_exactized_high_k_strategy.pdf`

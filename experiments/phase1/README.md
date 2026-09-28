# Phase 1 scripts (archived)

These scripts produced `results/phase1_original_code/`: one seed per configuration, run on the source as it was
before the Phase 2 fixes (random dev split, L2 divided by the mini-batch size, near-inactive LR decay, float64).
They are kept so that the Phase 1 numbers in the report can be traced. They expect the original source, the old
`models/<id>` layout, and `results/logs/` paths. Running them against the current source will not reproduce
Phase 1 exactly.

# Pinpad collapse study (2026-09-30 – 2026-10-02)

Plotting and analysis scripts behind the `EXPERIMENTS.md` entries on the
`director_og` pinpad find-then-lose collapse (e1031–e1067). They read archived
runs from the bucket (or `/work` for the reset study) and write PNGs next to
themselves; set `OUT=<dir>` where supported. Rendered figures are in
`/work/DoyaU/vasilache/work/collapse_figs/`.

| script | what |
|---|---|
| `director_og_pinpad_analysis.py` | metrics around find / peak / collapse for the 18 benchmark runs (e920–e954) |
| `plot_freeze.py` | freeze study (e1031–e1053): score and worker goal reward per arm |
| `plot_wkrcode.py`, `plot_wkrcode_diag.py` | worker conditioned on the goal code (e1054–e1059) vs control |
| `plot_deter_scale.py` | world-model state size / goal-AE fit / dimension reshuffle (`tools/deter_scale.py` output) |
| `plot_reset.py`, `reset_status.py` | reset study (e1060–e1067) |

Measurement tools live one level up: `tools/code_drift.py`, `tools/deter_scale.py`,
`tools/splice_reset.py`.

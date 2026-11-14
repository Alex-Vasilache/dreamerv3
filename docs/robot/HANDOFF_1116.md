# Handoff, Mon 11-16 noon CET (weekend 11-14 to 11-16)

Draft, updated as results land. Progress page (plain language, figures):
claude.ai artifact "Emotion Research Progress". Task list: `TASKS.md` (top).

## Needs you first

- **/apps disk is nearly full** (50 GB quota; it hit 100 % on 11-15). The
  Ligneul dataset download (`wget -r ... DSC_3017049.01_905_v1`, started in
  your tmux) is still pulling the fMRI part (80,597 files) into
  `apps/clipboard/`. We only need behaviour and stress, which are now on
  /work (`/work/DoyaU/vasilache/work/data_ligneul`, checked against the
  dataset's SHA256 manifest). Stopping that wget was blocked for me; please
  stop it (Ctrl-C in the tmux pane) and remove the clipboard copy if you
  don't need the fMRI data.
- Phone app copies of `emotion.py` / `policy_runner.py` are older than the
  repo; sync before the next live run (robot default arousal is kept at `kl`
  until then).

## Results so far

See `TASKS.md` and the progress page: pleasure validated on Rutledge's app
data; arousal = Pearce-Hall associability including the current error (best
in sim, consistent with human pupil data); dominance does its job in
contact; round LE1 (three learning mechanisms) null; LE2 running.

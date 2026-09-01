e506_dmc_hopper_stand_director_s0_BIG_j4676887  impl=director  task=dmc_hopper_stand  hard_metric=hamming

e506_director_hopper_hard.png -- panel A, HARD code, full range.
  One column per target code-similarity, left to right 0.2..1.0. Each
  pair-block (stacked within its column) is two goal IMAGES decoded
  from two DIFFERENT codes whose hard-code similarity falls in that
  column. Label under each pair is the OPPOSING number only -- the
  true goal-state similarity (column position already says the code
  similarity).
  Metric: plain Hamming equality (same id per block or not) -- the
  right metric for Director's unordered argmax labels.
  If images still look near-identical in a low column (e.g. 0.2), that
  is code collapse (many dissimilar codes -> one goal), not subtlety.

e506_director_hopper_soft.png -- panel B, SOFT code, narrow range.
  Pairs are restricted to states whose GOAL-SPACE (deter) cosine-max
  similarity is already in the column's band (0.7/0.8/0.9/1.0, left to
  right), i.e. states that were already close. Each pair-block decodes
  the two states' SOFT (pre-sample) codes to images. Label under each
  pair is the OPPOSING number only -- the soft-code similarity (column
  position already says the goal-state similarity). If code sim tracks
  goal sim smoothly within this narrow high band and the images show
  correspondingly small, real visual differences, that supports "codes
  track subtle changes" over "codes collapse".

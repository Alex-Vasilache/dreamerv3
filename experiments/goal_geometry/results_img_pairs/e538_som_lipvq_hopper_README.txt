e538_dmc_hopper_stand_som_lipvq_line_prod_s0_BIG_j4677171  impl=vq  task=dmc_hopper_stand

e538_som_lipvq_hopper_hard.png -- panel A, HARD code, full range.
  One row per target code-similarity (left margin: 1.0..0.2). Each
  pair-block is two goal IMAGES decoded from two DIFFERENT codes whose
  hard cosine-max similarity is (nearest realized level to) that row.
  Label under each pair: "<actual code sim> <true goal-state sim>".
  If images still look near-identical at a low row (e.g. 0.2), that is
  code collapse (many dissimilar codes -> one goal), not subtlety.

e538_som_lipvq_hopper_soft.png -- panel B, SOFT code, narrow range.
  Pairs are restricted to states whose GOAL-SPACE (deter) cosine-max
  similarity is already in the row's band (0.7/0.8/0.9/1.0), i.e.
  states that were already close. Each pair-block decodes the two
  states' SOFT (pre-sample) codes to images. Label: "<soft code sim>
  <true goal-state sim>". If code sim tracks goal sim smoothly within
  this narrow high band and the images show correspondingly small,
  real visual differences, that supports "codes track subtle changes"
  over "codes collapse".

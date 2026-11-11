#!/bin/bash
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
cd /work/DoyaU/vasilache/work/meetings/tce_investigation; mkdir -p probes
while read name run step; do
  [ -f probes/$name.npz ] && continue
  python -u probe.py $run $step probes/$name > probes/$name.log 2>&1; echo "$name exit $?"
done <<LIST
sqrt_s0_2M e9502_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s0_j4781346_2 2000000
sqrt_s1_2M e9508_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s1_j4781346_8 2000000
crit_s1_2M h0322_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-fromto_dir-tce_hash-tce_critic-pubmatch_s1_j4809330_8 2000000
sqrt_s0_4M e9502_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s0_j4781346_2 4000000
ent_s1_2M e1885_pinpad_six_director-director_og-wkr_ent_fixed-pubmatch_s1_j4767814_5 2000000
ent_s0_4M e1882_pinpad_six_director-director_og-wkr_ent_fixed-pubmatch_s0_j4767814_2 4000000
sqrt_s2_2M e9514_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s2_j4781346_14 2000000
crit_s0_4M h0316_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-fromto_dir-tce_hash-tce_critic-pubmatch_s0_j4809330_2 4000000
LIST

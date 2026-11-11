#!/bin/bash
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
cd /work/DoyaU/vasilache/work/meetings/tce_investigation; mkdir -p kl
while read name run step; do
  python -u probe_kl.py $run $step kl/$name > kl/$name.log 2>&1; echo "$name exit $?"
done <<LIST
sqrt_s1_2M e9508_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s1_j4781346_8 2000000
sqrt_s0_2M e9502_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s0_j4781346_2 2000000
sqrt_s0_4M e9502_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s0_j4781346_2 4000000
crit_s1_2M h0322_pinpad_six_director-director_og-wkr_ent_fixed-wkr_v3-fromto_dir-tce_hash-tce_critic-pubmatch_s1_j4809330_8 2000000
LIST

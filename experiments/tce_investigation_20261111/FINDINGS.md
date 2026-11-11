# TCE pin pad investigation (11-11, offline only: logged metrics, numpy sim, checkpoint rollouts on CPU)
Files: seeds.py/seeds.json (per-seed timing), fromto_traj.py (TCE metrics), pinsim.py (sim), probe.py + probes/ (rollouts),
analyze.py -> analysis.txt, literature.md.

1. Pin pad six is all-or-nothing: a seed either finds the first reward (then sustained-50 ~0.5M later, late ~205-220)
   or stays at 0. Solved/finished: OG 2/5, DV3 0/5, OG+fixed ent 3/5, V 1/5, US-SOM ~4/13, TCE sqrt 4/12, TCE-KL ~5/12, TCE critic 2/3.
2. TCE barely acts: lean controller (target 1.0) holds E_pi[b] ~ E_unif[b]; weight falls to 0.05-0.2 by 1-2M on every
   six seed; it only rises after a seed has solved. Novelty ~0.8 almost everywhere (most of the 256 TO regions per row are unseen).
3. Failed seeds at 2M: few pad presses (2-20 per 1000 steps vs 106-136 in solved runs), goals mostly off the pads
   (3-31% on a pad vs 60-67%), sqrt s2 sits in ~6 cells (H(pos)=2.7 bits). They never get past prefix 2.
   ent s1 (failed) is different: goals on pads (60%) but blurry and the worker reaches only 16%.
4. Representation is fine: deter -> last pad / sequence prefix linear probe 95-100% on solved runs; hash cell -> last pad 89-95%.
5. Worker: reaches its goal within the 8-step hold 18-25% of holds in solved runs (~50% in failed runs, whose goals are near);
   52-70% of pad presses are accidental (not the goal pad), i.e. solved runs press pads by passing through.
6. Timing (sim, oracle that knows the sequence, K-step holds): K=1/2/4/8/16 -> four 415/392/368/357/296, five 353/338/324/313/249,
   six 291/283/262/220/173. Solved six seeds already score ~205-220 = the K=8 ceiling. DV3 four 382 > K=8 ceiling 357.
7. Exploration (sim, no learning, K=8): first reward on six after median 505k steps for uniform random goals,
   25k for "pick a pad not in the current window" (episodic memory). Real agents: 1-2M or never -> worse than random goals.
8. Decay: mean count lifetime 0.72M steps (half-life 0.5M); not the binding constraint since failed seeds barely press pads.
9. US-SOM: only matters where TO = goal-code halves (direct TCE sqrt); with hash->hash (TCE critic) it is unused; V vs US-SOM n.s.

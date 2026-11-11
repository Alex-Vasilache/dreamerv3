# TCE: related work on exploration

Background: TCE counts real transitions n(from, to). FROM is a 16-bit SimHash of the
1024-d deterministic world-model state. TO is either the goal region reached or the low 8
bits of the same hash. The bonus is 1/sqrt(n+1), and counts decay with a 0.5M-step
half-life. It is used in two ways: directly in the manager loss, or as an intrinsic reward
with its own critic. Pin pad six is solved on about 4 of 11 seeds.

The question for this review is why some seeds fail. The candidate causes are:
1. too many (from, to) pairs to try;
2. decay forgetting early tries;
3. goals that don't map to pads;
4. worker timing;
5. irrelevant state bits diluting the counts.

---

## 1. Hash- and count-based exploration

**Tang et al. 2017, "#Exploration: A Study of Count-Based Exploration for Deep RL", NeurIPS.**
- They map each state to k sign bits of a fixed random projection (SimHash) and count the hash codes, with bonus β/sqrt(n).
- They also learn the hash: an autoencoder with a sigmoid code layer that is pushed to binary values and then rounded.
- Their main finding is that the hash must keep the distinctions that matter for the task. On hard Atari games (Montezuma, Venture), a learned or hand-made code beat raw-pixel SimHash, and k had to be tuned: too few bits merges important states, too many leaves every count near zero.
- For us: our FROM hash is exactly their SimHash, but on a learned latent. Their results say the 16 random directions may mostly encode features that have nothing to do with the pads.

**Bellemare et al. 2016, "Unifying Count-Based Exploration and Intrinsic Motivation", NeurIPS.**
- They derive a pseudo-count from how much a density model's probability of a state rises after one more update on it.
- This gives counts for states that are never exactly repeated, and it made the first real progress on Montezuma's Revenge.
- For us: the count does not have to be a table. It could be a density over the WM latent, which generalizes smoothly instead of using hard hash buckets.

**Ostrovski et al. 2017, "Count-Based Exploration with Neural Density Models", ICML.**
- They replace the CTS density model with PixelCNN and mix the bonus into mixed Monte-Carlo returns.
- The paper stresses that the bonus has to propagate quickly. The return mixing mattered about as much as the density model did.
- For us: this supports the critic variant of TCE, where the bonus is carried across steps by a value function, rather than a per-decision bonus alone.

**Machado, Bellemare & Bowling 2020, "Count-Based Exploration with the Successor Representation", AAAI.**
- The norm of a successor representation that is still being learned acts as an implicit inverse count (the "substochastic SR").
- No density model is needed, and the counts live in the agent's own learned features.
- For us: the count can be read from temporally extended features, which is close to counting where you can get to next rather than where you are.

**Classical background: MBIE-EB (Strehl & Littman 2008, JCSS) and UCRL2 (Jaksch, Ortner & Auer 2010, JMLR).**
- Both count (s, a) or (s, a, s′) and give a bonus proportional to 1/sqrt(n).
- They come with guarantees, but only for small tabular MDPs with no decay. The cost of exploring grows with the number of transitions to try (|S|²|A|-type terms), which is concern 1.

## 2. Novelty from prediction error

**Pathak et al. 2017, "Curiosity-driven Exploration by Self-supervised Prediction" (ICM), ICML.**
- The reward is the error of a forward model that works in a feature space trained by inverse dynamics, i.e. predicting the action from (s, s′).
- Inverse dynamics keeps only the features the agent can affect, so distractors such as noise or leaves blowing in the wind are ignored.

**Burda et al. 2019, "Exploration by Random Network Distillation" (RND), ICLR.**
- The reward is the error of a network trained to predict the output of a fixed, randomly initialized network.
- The bonus shrinks with visitation like a count, but it generalizes smoothly between states.
- RND kept a non-episodic (lifelong) intrinsic return with its own value head, separate from the extrinsic one. This is the same design as our critic variant.

**Sekar et al. 2020, "Planning to Explore via Self-Supervised World Models" (Plan2Explore), ICML.**
- The reward is the disagreement of an ensemble of latent one-step models, computed and optimized entirely in imagination on Dreamer's latent.
- This is the closest relative of TCE-critic: novelty is scored inside imagined rollouts, and the world model is trained only on real data.
- Unlike counts, disagreement is a learned model of novelty, so it can be queried on imagined states the agent has never visited.
- For us: imagined steps that land in a hash bucket never seen in real data get the maximum bonus. That is optimistic, but it is only as reliable as the hash.

## 3. Episodic novelty ("which pads have I already tried?")

**Savinov et al. 2019, "Episodic Curiosity through Reachability", ICLR.**
- The agent keeps an episodic memory of embeddings and gives a reward when the current state is more than k steps away from everything in memory. Reachability is judged by a trained comparator network.
- Novelty is reset each episode, so this kind of memory cannot permanently go stale.

**Badia et al. 2020, "Never Give Up: Learning Directed Exploration Strategies" (NGU), ICLR.**
- The reward is r = r_episodic · min(max(α, 1), L).
- r_episodic is an inverse kernel count over the k nearest neighbours in an episodic memory of inverse-dynamics embeddings.
- α is an RND-based lifelong multiplier.
- The episodic term makes the agent avoid revisiting within an episode. The lifelong term slowly turns that pressure down in places that have been explored across all of training.
- They learn a family of policies with different exploration weights β and discount factors.

**Badia et al. 2020, "Agent57: Outperforming the Atari Human Benchmark", ICML.**
- Agent57 is NGU plus a meta-controller (a bandit) that chooses the exploration/exploitation policy each episode, and separate value heads for the intrinsic and extrinsic rewards.
- The split value heads were needed for training to be stable.

**Raileanu & Rocktäschel 2020, "RIDE: Rewarding Impact-Driven Exploration", ICLR.**
- The reward is the change in an ICM-style embedding, ‖φ(s′) − φ(s)‖, divided by sqrt of the episodic count of s′.
- So it rewards transitions that change the controllable state.

**Zhang et al. 2021, "NovelD: A Simple yet Effective Exploration Criterion", NeurIPS.**
- The reward is max(N(s′) − α·N(s), 0), where N is RND novelty, and it is paid only on the first visit to s′ in the episode (an episodic count = 1 gate).
- This rewards crossing a frontier between familiar and unfamiliar states, which is a transition-based criterion like ours.

**Flet-Berliac et al. 2021, "Adversarially Guided Actor-Critic" (AGAC), ICLR.**
- An adversary predicts the actor's actions, and the actor is rewarded for being hard to predict.
- It is combined with an episodic state-count bonus on MiniGrid.

**Henaff et al. 2022, "Exploration via Elliptical Episodic Bonuses" (E3B), NeurIPS.**
- The authors show that RIDE, AGAC and NovelD depend mostly on their episodic-count term, and that the term breaks down when states rarely repeat exactly.
- E3B replaces it with an elliptical bonus φᵀC⁻¹φ, where C is the covariance of the episode's inverse-dynamics embeddings and is reset each episode.
- For us: the episodic part, not the lifelong part, is what does the work on procedurally varied tasks.

**Episodic vs lifelong.**
- A lifelong count (TCE, RND, #Exploration) says "this transition is rare over all of training". It fades as training goes on, and with decay it can come back.
- An episodic count says "I have not done this yet in this episode". It never fades, because it resets every episode, so it supports systematic search within an episode.
- NGU multiplies the two: the episodic term gives the direction to search, and the lifelong term turns the pressure down where the world is already well known.
- Pin pad has the structure of a search within one episode. The pad order is fixed, and after a wrong press the agent should try a different next pad, which is exactly what an episodic count expresses. TCE currently has only the lifelong part.

## 4. Go-Explore

**Ecoffet et al. 2019, "Go-Explore: a New Approach for Hard-Exploration Problems", arXiv:1901.10995; Ecoffet et al. 2021, "First return, then explore", Nature.**
- The agent keeps an archive of "cells" (downscaled frames, or domain features such as room and keys).
- Cells are chosen with weights ∝ 1/sqrt(count chosen / visited / led-to-new-cell). The agent returns to the chosen cell (by reset, or with a goal-conditioned policy) and then explores from there.
- Two failures are named:
  - detachment: the agent forgets frontiers that it once found promising;
  - derailment: exploratory noise on the way back stops the agent from reaching the frontier.
- Cell design decides whether the method works. Domain-knowledge cells were much stronger than pixel cells. Gallouédec & Dellandréa 2023, "Cell-Free Latent Go-Explore", ICML, learns the cells from a latent.
- For us:
  - Decaying counts can cause detachment: early partial sequences fade out before they are extended.
  - Selection weights based on whether choosing a cell led to new cells say which FROM states are worth exploring from, not only which (from, to) pairs are rare.
  - The manager-picks-goal / worker-reaches-goal structure already is "return then explore". If the worker cannot reliably reach the goal within K=8 steps, that is derailment (concern 4).

## 5. Hierarchical and goal-conditioned exploration

**Vezhnevets et al. 2017, "FeUdal Networks for Hierarchical RL", ICML.** The manager sets directional goals in a learned latent space and the worker is rewarded for moving in that direction. It is the ancestor of Director's split into goal choice and goal reaching.

**Nachum et al. 2018, "Data-Efficient Hierarchical RL" (HIRO), NeurIPS; Levy et al. 2019, "Learning Multi-Level Hierarchies with Hindsight" (HAC), ICLR.** Both are off-policy HRL with goals in state space. Both relabel the high-level goal with the state that was actually reached (off-policy correction / hindsight), so the manager learns from what the worker really did. Our "TO = goal region of the reached state" is the same move applied to counting.

**Hafner et al. 2022, "Deep Hierarchical Planning from Pixels" (Director), NeurIPS.**
- The manager picks a discrete code from a goal VAE every K=8 steps, and the worker maximizes cosine-max similarity to the decoded goal.
- Exploration reward: the goal-autoencoder's reconstruction error on the WM state, given to the manager.
- Pin pad is introduced here, and Director solves it while flat Dreamer does not. Their exploration bonus is novelty of states, not of transitions.

**Mendonca et al. 2021, "Discovering and Achieving Goals via World Models" (LEXA), NeurIPS.** An explorer trained in imagination with Plan2Explore disagreement is paired with an achiever trained to reach goals sampled from the replay buffer, using a temporal-distance reward. Exploration and goal reaching are learned as separate policies.

**Mazzaglia et al. 2023, "Choreographer: Learning and Adapting Skills in Imagination", ICLR.** Skills are learned as VQ codes over the world-model latent, separately from an exploration policy. A meta-controller then picks among the codes in imagination. This is close to our SOM/VQ goal codebook.

**Pong et al. 2020, "Skew-Fit: State-Covering Self-Supervised RL", ICML; Pitis et al. 2020, "Maximum Entropy Gain Exploration for Long Horizon Multi-goal RL" (MEGA), ICML.**
- Skew-Fit samples goals with weight p(s)^α for α < 0 so that goal coverage becomes uniform.
- MEGA picks goals from low-density, reachable regions of the achieved-goal distribution.
- Both count goals rather than states, which is what our TO = goal-region choice already does.

**Eysenbach et al. 2019, "Diversity is All You Need" (DIAYN), ICLR; Sharma et al. 2020, "Dynamics-Aware Unsupervised Discovery of Skills" (DADS), ICLR.**
- DIAYN learns skills that maximize I(skill; state).
- DADS learns skills that maximize I(skill; s′ | s), so its skills are predictable transitions.
- DADS's objective is a transition-level counterpart to our (from, to) counting.

**Campos et al. 2020, "Explore, Discover and Learn: Unsupervised Discovery of State-Covering Skills", ICML.** They show that mutual-information skill methods such as DIAYN cover only the states that are already visited. EDL first explores (with state-marginal matching), then learns skills on that data. The lesson for us: a goal codebook learned only from visited data cannot represent pads that have not been reached (concern 3).

**Counting over abstract states or features.**
- Martin et al. 2017, "Count-Based Exploration in Feature Space for RL", IJCAI: the φ-pseudocount is the product of per-feature visit densities over the agent's linear features.
- Choi et al. 2019, "Contingency-Aware Exploration in RL", ICLR: an attentive inverse-dynamics model finds the location the agent controls, and they count (room, controllable position, ...) cells. This gave large gains on Montezuma.
- We did not find an established method that counts transitions between learned option or goal codes in a Director-style hierarchy, and that seems to be TCE's new contribution.

## 6. Controllability and relevance filters

- **Inverse dynamics** (ICM; NGU's embedding; RIDE; E3B): train φ to predict a from (φ(s), φ(s′)). Features the agent cannot change are dropped, and E3B's ablations show this matters a lot in noisy environments.
- **Contingency awareness** (Choi et al. 2019): attention over the controllable region. Counts are taken only over agent-relevant coordinates, which is an explicit relevance filter on what gets hashed.
- **Empowerment** (Klyubin, Polani & Nehaniv 2005, IEEE CEC; Mohamed & Rezende 2015, NeurIPS): the reward is the channel capacity I(a; s′ | s). It favours states from which the agent has many distinct, controllable futures, so it measures which features the agent can influence, not just novelty.
- **Learned hashes** (Tang et al. 2017 AE hash; Gallouédec & Dellandréa 2023): learn the discretization instead of using random projections.
- Taken together: a random-projection hash of the full deterministic RSSM state mixes in features the agent cannot affect (for example the background and the pad colours shown on screen) and features it can (agent position, sequence progress). In pin pad only position and sequence progress matter, so random bits that track anything else split each meaningful count across buckets (concern 5).

---

## Implications for TCE

- **Add an episodic count and multiply it with the lifelong one, as in NGU (Badia et al. 2020).**
  - Keep a per-episode table of (from-region, to-region) transitions already taken, and use r = r_episodic · clip(lifelong).
  - This is the "I already tried the blue pad after red this episode" signal. E3B (Henaff et al. 2022) shows that this term carries most of the benefit in similar methods.
- **Hash controllable features, not the raw deter state (ICM/NGU inverse-dynamics embeddings; Choi et al. 2019).**
  - Project an inverse-dynamics or goal-code embedding before the SimHash, or learn the hash as in Tang et al.'s AE hash.
  - This should cut the effective (from, to) space and stop irrelevant bits from diluting counts. In pin pad the useful state is roughly agent position and pads pressed so far.
- **Remove or slow the count decay (Go-Explore's detachment; Strehl & Littman / UCRL2 use permanent counts).**
  - Halving every 0.5M steps can bring back the bonus on transitions already tried and hide frontiers found early.
  - Try no decay, or a decay tied to world-model drift (rehash when the WM changes) rather than to wall-clock steps.
- **Weight FROM states the Go-Explore way (Ecoffet et al. 2021; MEGA, Pitis et al. 2020).**
  - Favour (from, to) pairs whose FROM bucket recently led to new buckets, i.e. score frontier productivity rather than rarity alone.
  - This targets the "too many transitions to try" failure by putting effort on the edge of the explored region.
- **Check that goals map to pads before blaming the bonus (EDL, Campos et al. 2020; LEXA, Mendonca et al. 2021).**
  - A codebook trained only on visited data may have no code for unvisited pads.
  - Measure, per seed, how goal codes are distributed over pad positions, and how often the worker actually reaches its goal within K=8 (Go-Explore's derailment). Failed seeds may be failures to reach goals, not failures to explore.
- **Keep the critic variant separate from the extrinsic critic, with its own value head and no decay inside imagination (RND; Agent57; Plan2Explore).** This is already our design. The literature suggests the bonus scale should be adapted per seed, for example with Agent57's bandit over β, rather than fixed.

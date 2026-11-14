# Emotion: pleasure, arousal and dominance from the internal model

The target model since 2026-11-12. It replaces the 11-04 to 11-10 model
(kept in [history/EMOTION_2026-11-06.md](history/EMOTION_2026-11-06.md)).
The face mapping is in [FACE_MATHS.md](FACE_MATHS.md); the literature in
[EMOTION_LITERATURE.md](EMOTION_LITERATURE.md). Status of each part is in
the table at the end: parts are being built behind switches, the old model
stays selectable until the sim round has run.

## Rules

1. Every axis is one quantity the DreamerV3 agent's world model and critics
   already predict or evaluate. No event tables, task rules or mixing weights.
2. The only innate inputs come from the task: reward (including intrinsic
   rewards) and pain (impact). Everything else is learned, including where
   danger is.
3. No hand-coded protection: the runaway brake is replaced by a reflex that
   foresees hits with the world model (after the sim test).

## Notation

| symbol | meaning |
|---|---|
| x̄ | running average of x over ~10 s |
| x̃ | fading average, x̃_t = λ x̃_{t-1} + (1 − λ) x_t, λ = exp(−1/(0.5 s × 65 Hz)) ≈ 0.97 |
| z(x) | z-score, (x − x̄) / running SD of x |
| r, c | reward and pain at this step |
| s | the world model's latent state (32 categorical variables) |
| V, T | value = expected future reward; threat = expected future pain (the `agent.fear` critic; "fear" is kept for the emotion, LeDoux 2014) |
| S̄_V, S̄_T | the two critics' running return scales, as the actor uses them |
| w | the actor's pain weight (`agent.fear.weight`, 1.0) |

## Pleasure: what just happened, and how the outlook moved

    o_t = r_t / S̄_V − w c_t / S̄_T          outcome now
    O_t = V_t / S̄_V − w T_t / S̄_T          outlook
    δ_t = o_t + γ O_{t+1} − O_t              TD error over reward and pain
    E_t = (1 − γ) O_{t+1}                    expectation per step
    p_t = δ_t + E_t = o_t + O_{t+1} − O_t    (γ cancels)
    pleasure_t = tanh z(p̃_t)

The outlook sums about 1/(1 − γ) = 130 future steps, so (1 − γ)O is its value
per step, in the units of o. Adding it to the TD error cancels the discount:
pleasure is the outcome now plus how much the outlook just changed, faded
over ~0.5 s. Recent outcomes, expectations and prediction errors that fade
also model human momentary well-being (Rutledge et al. 2014, with fitted
weights). With nothing learned, pleasure is reward minus pain; entering danger
lowers the outlook before contact. A steady dangerous state reads only through
its reward and pain; controllability shows in dominance. Fading is needed
because O_{t+1} − O_t over one 15 ms step is critic jitter and a hit lasts one
or two steps (unfaded TD error: AUC 0.74 for hits; faded: 0.89).

Human data (`../../rutledge_rl`, 11-14). On Rutledge's gamble task, our
terms fit happiness ratings: in two online cohorts (Jangraw et al. 2023,
n = 93 and 1,829) r + γ'(V' − V) with γ' ≈ 0.5-0.6 matches Rutledge's model;
on Rutledge's own app data (n = 42,121) his model fits best in-sample, but the
equal-weight r + V' − V predicts each participant's next play best
(play-to-play r = 0.51 vs 0.38). "What I expected" is the value, not the
advantage; Q(s,a) and V(s) fit alike; a regret (advantage) term adds nothing.
So pleasure keeps equal weights and no regret term.

## Arousal: how badly this moment was predicted

    e_t = (1/32)[−log p̂(s_t)] − log p̂(r_t) − log p̂(c_t)        prediction error
    ê_t = (1/32) H(p̂_s) + H(p̂_r) + H(p̂_c)                      expected error
    arousal_t = σ(z(e_t))   or   σ(z(e_t − ê_t))

p̂ is the prediction made before the sensor reading (the prior; reward and
pain heads read on the prior state). Dividing the state term by 32 makes the
32 state variables count as much as one reward or pain variable; all terms
are in nats. Being tested: with novelty (total error, noradrenaline plus
acetylcholine in Yu & Dayan's terms) or without (unexpected error,
noradrenaline only).

Update 11-14: an offline bake-off of ten definitions on 48 sim runs
(`LIT_AROUSAL_YU.md`) picked the Pearce-Hall associability, a running
average of the absolute TD error, α_t = η|δ_{t-1}| + (1-η)α_{t-1} with
η ≈ 0.05-0.1 per step, arousal = σ(z(α)) over ~60 s: AUC 0.81 at a hit, 0.75
0.1-0.3 s before, 0.69 after a push, and the only definition that adds
information beyond pleasure (0.63). It is the arousal for the next sim round;
the surprise variants and Yu's base-rate version stay as switches.

## Dominance: can my actions make the end state different?

    m_s = (1/32) Σ_k s_k,  m_V = (1/32) Σ_k V_k,  m_T = (1/32) Σ_k T_k     average end point
    d_k = (1/32) KL(s_k ‖ m_s) + KL(V_k ‖ m_V) + KL(T_k ‖ m_T)            how far command k lands from it
    C_t = (1/32) Σ_k d_k                                                  empowerment
    dominance_t = tanh z(C_t)

s_k, V_k, T_k: predicted latent state, value and threat after holding wheel
command k for H steps in the world model; k: 32 commands drawn uniformly
each time; H = 16 or 32 (0.25 or 0.5 s), being tested. Recomputed every 3 s in
slices on the phone. On the recorded sessions 32 random commands at 16 steps
picked out hits best per session (AUC 0.74 ± 0.07, 87 ms per evaluation);
deviations around the policy and the old 9-command grid did worse
(SAMPLE_EFF_STUDY.md, empowerment comparison).

Named emotions are regions of the PAD space, not extra signals: fear is low
pleasure, high arousal, low dominance; helplessness is a bad situation with
low dominance.

## Pain, threat and the catch reflex

    ι_t = |θ̇_t − θ̇_{t−1}|                         impact
    c_t = clip((ι_t − ι0) / (ι1 − ι0), 0, 1)       pain

ι0 is the tolerance (no pain below), ι1 the impact of full pain (from data,
`env.robot.pain_mode impact`). The impact signal picks out bumper hits without
knowing where the bumpers are (AUC 0.81 on the recorded sessions).

Threat is a second critic trained on pain; the actor trades the two
advantages, so reward cannot buy a hard hit:

    L_actor = −E[(A^V / S̄_V − w A^T / S̄_T) log π(a|s)] − η H[π]

In sim (pushes and walls, n = 4) this critic cut bumper crossings from
14.1 ± 3.0 to 4.0 ± 1.0 a minute when trained on bumper-angle pain; impact
pain is untested.

The catch reflex replaces the runaway brake: a learned constraint-manifold
safety filter (after ATACOM, Liu et al. 2025), with the world model in place
of a physics model and threat in place of a hand-written limit. Each step:

    k(a) = T̂(a) − T_max                           how far over the limit
    g    = (k(a + ε) − k(a − ε)) / 2ε              slope along the drive
    a'   = a − max(0, k(a)) / g                    command sent

T̂(a) is the threat predicted after rolling a a few steps ahead. It fires on a
prediction, like catching yourself when you start to slip; it cannot protect
while the model is still poor and carries no guarantee.

## Brain counterparts (closest, with how firm)

| quantity | counterpart | source |
|---|---|---|
| TD error, change in outlook (pleasure) | phasic dopamine; momentary happiness | Schultz et al. 1997; Rutledge et al. 2014 (strong) |
| reward and pain now | hedonic "liking"; pain unpleasantness in anterior cingulate | Berridge & Kringelbach 2015; Rainville et al. 1997 (strong) |
| threat | defensive threat circuits, apart from felt fear; serotonin as dopamine's aversive opponent | LeDoux 2014; Daw, Kakade & Dayan 2002 (debated) |
| prediction error (arousal) | noradrenaline: unexpected uncertainty; acetylcholine: expected uncertainty | Yu & Dayan 2005; Aston-Jones & Cohen 2005; Nassar et al. 2012 (strong) |
| empowerment (dominance) | prefrontal detection of control inhibiting the serotonergic stress response | Maier & Seligman 2016; Amat et al. 2005 (strong in rodents) |
| pain as impact | nociception, a sensation with its own drive | Craig 2003; Seymour 2019 (strong) |

## Status

| part | status |
|---|---|
| Pleasure, arousal, dominance as above | being built as emotion model `v2` (old `v1` kept) |
| Arousal with or without novelty; reward/pain from the prior; H 16 or 32; value and threat in dominance | being tested on the 5 recorded real sessions |
| Impact pain, threat on it, catch reflex | being built (switches, off by default); sim round PF: today's pain + brake vs impact pain + threat vs + reflex |
| Face: three axes only, pain as the negative end of pleasure, dominance as lid, eye height and gaze | built (face.html, 43d0a0e) |
| Time constants: pleasure fading 0.5 s, z-score window ~10 s | open |

Plan page with figures: claude.ai artifact "Robot Emotion Model".

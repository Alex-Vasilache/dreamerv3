# DreamerV3 on the OIST Smartphone Robot

This repo trains [DreamerV3][paper] on the
[OIST smartphone robot](https://github.com/oist/smartphone-robot/wiki): a phone
on a two-wheeled base that learns to balance and to follow joystick commands.
The policy runs on the phone; a learner on a GPU (a laptop or a Slurm cluster)
trains on what the phone records and pushes it new weights every few seconds.

**Start at [ROBOT.md](ROBOT.md)** -- install, training, monitoring, configs and
troubleshooting.

## Repos

The robot setup spans three repos:

| repo | branch | what |
|---|---|---|
| [oist/dreamerv3](https://github.com/oist/dreamerv3) | `main` | learner, robot env, simulator, tools ([ROBOT.md](ROBOT.md)) |
| [Alex-Vasilache/smartphone-robot-android](https://github.com/Alex-Vasilache/smartphone-robot-android/tree/dreamer-bridge) | `dreamer-bridge` | phone app `dreamerBridge` (fork of [oist/smartphone-robot-android](https://github.com/oist/smartphone-robot-android)) |
| [Alex-Vasilache/smartphone-robot-firmware](https://github.com/Alex-Vasilache/smartphone-robot-firmware/tree/telemetry-cache) | `telemetry-cache` | base firmware with a motor watchdog and I2C recovery (fork of [oist/smartphone-robot-firmware](https://github.com/oist/smartphone-robot-firmware)) |

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

On the robot, the observations are the body's tilt and tilt rate, the two
wheel speeds and, for command following, the joystick command; the actions are
the two wheel commands. The world model and actor-critic train on the learner; only the
actor runs on the phone.

The repo also contains a simulated robot (`robot_sim`) for trying things
without hardware; see [ROBOT.md](ROBOT.md#without-hardware).

## Credit

The DreamerV3 code is Danijar Hafner's
[reimplementation](https://github.com/danijar/dreamerv3) ([paper][paper],
[website][website]); the robot hardware, firmware and Android framework are
from the [OIST smartphone robot project](https://github.com/oist/smartphone-robot/wiki).
If you use DreamerV3, please cite:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

# Instructions (general DreamerV3)

For the robot, follow [ROBOT.md](ROBOT.md) instead. The rest of this file is the
original DreamerV3 usage, for other tasks.

# Instructions

The code has been tested on Linux and Mac and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Scalar metrics are also writting as JSONL files.

# Tips

- All config options are listed in `dreamerv3/configs.yaml` and you can
  override them as flags from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag.
- You can use multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter size50m`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as Scope summaries. Other outputs like WandB and
  TensorBoard can be enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions. Try `--batch_size 1` to rule out an out of memory error.
- Many environments are included, some of which require installing additional
  packages. See the `Dockerfile` for reference.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

# Disclaimer

This repository contains a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is unrelated to Google or DeepMind. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3

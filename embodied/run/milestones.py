import elements


def make_saver(cp, logdir, every, keys=('step', 'agent')):
  """Returns a `fn(step)` that writes a permanent checkpoint every `every` env
  steps, on the absolute step grid (500k, 1M, 1.5M, ...).

  The rolling `ckpt/` directory keeps only the latest save, so a finished run
  can only be analyzed at its final parameters. Milestones live in their own
  `ckpt_milestones/<step>/` folders, are never pruned, and are picked up by the
  bucket archiver together with the rest of the logdir.

  Only `step` and `agent` are saved -- replay is deliberately left out, so a
  milestone is an analysis snapshot (loadable with `--run.from_checkpoint`),
  not a resume point. Pass `every=0` to disable.
  """
  every = int(every)
  directory = elements.Path(logdir) / 'ckpt_milestones'

  def save_milestone(step):
    if every <= 0:
      return
    index = int(step) // every
    if index < 1:
      return
    path = directory / f'{index * every:012d}'
    if elements.checkpoint.exists(path):
      return  # Already written, e.g. before a requeue.
    if path.exists():
      # Partial save left behind by a killed job; `save()` refuses to overwrite.
      path.remove(recursive=True)
    cp.save(path, keys=keys)

  return save_milestone

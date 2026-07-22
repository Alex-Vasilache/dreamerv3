"""Isolated verification: does the .values()/.write()-based "freeze module
params for one call" trick zero out the gradient into the module's own
parameters while preserving gradient into its input, under nj.grad?

Mimics the planned agent.py construction:
    g(params, x) = f(sg(params), x)
"""
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


class TinyDecoder(nj.Module):
  def __call__(self, x):
    w = self.value('w', lambda: jnp.array(3.0, f32))
    return w * x


def freeze_call(module, x):
  live = module.values
  for k, v in live.items():
    module.write(k, sg(v))
  out = module(x)
  for k, v in live.items():
    module.write(k, v)
  return out


def run(x0, mode):
  dec = TinyDecoder(name='dec')
  _ = dec(x0)  # ensure dec/w exists before nj.grad introspects targets

  def normal_loss(x):
    y = dec(x)
    return (y * y).sum()

  def frozen_loss(x):
    y = freeze_call(dec, x)
    return (y * y).sum()

  loss_fn = frozen_loss if mode == 'frozen' else normal_loss

  loss_w, _params_w, grads_w = nj.grad(loss_fn, ['dec'])(x0)
  loss_x, _params_x, grads_x = nj.grad(loss_fn, [0])(x0)
  return dict(
      loss=loss_w,
      grad_w=list(grads_w.values())[0],
      grad_x=grads_x[0],
  )


pure_run = nj.pure(run)


def main():
  x0 = jnp.array(2.0, f32)
  state = {}
  # Init pass: create the 'dec/w' state entry.
  state, _ = pure_run(state, x0, 'normal', seed=0, create=True, modify=True)

  state_n, out_n = pure_run(state, x0, 'normal', seed=0, create=False, modify=True)
  state_f, out_f = pure_run(state, x0, 'frozen', seed=0, create=False, modify=True)

  out_n = {k: float(v) for k, v in out_n.items()}
  out_f = {k: float(v) for k, v in out_f.items()}
  print('NORMAL:', out_n)
  print('FROZEN:', out_f)

  ok = True
  if abs(out_n['grad_w']) < 1e-6:
    print('SUSPICIOUS: normal-call grad wrt w is ~zero; test setup is broken '
          '(expected the normal call to have a nonzero param gradient).')
    ok = False
  if abs(out_f['grad_w']) > 1e-6:
    print('FAIL: frozen-call grad wrt decoder param w is NOT zero:', out_f['grad_w'])
    ok = False
  else:
    print('PASS: frozen-call grad wrt decoder param w is ~zero:', out_f['grad_w'])

  if abs(out_f['grad_x']) < 1e-6:
    print('FAIL: frozen-call grad wrt x is zero (should be nonzero)')
    ok = False
  else:
    print('PASS: frozen-call grad wrt x is nonzero:', out_f['grad_x'])

  if abs(out_n['grad_x'] - out_f['grad_x']) > 1e-5:
    print('FAIL: grad wrt x differs between normal and frozen calls '
          f'(normal={out_n["grad_x"]}, frozen={out_f["grad_x"]}); should match '
          'since freezing only affects the param branch, not the input branch.')
    ok = False
  else:
    print('PASS: grad wrt x matches between normal and frozen calls '
          f'({out_n["grad_x"]} vs {out_f["grad_x"]})')

  if abs(out_n['loss'] - out_f['loss']) > 1e-5:
    print('FAIL: forward loss value differs between normal and frozen calls:',
          out_n['loss'], out_f['loss'])
    ok = False
  else:
    print('PASS: forward values match between normal and frozen calls')

  print('ALL_OK' if ok else 'SOME_CHECKS_FAILED')
  if not ok:
    raise SystemExit(1)


if __name__ == '__main__':
  main()

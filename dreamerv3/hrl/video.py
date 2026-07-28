"""Image/video helpers for the report and episode-panel logging paths."""
import math

import jax
import jax.numpy as jnp

def tb_video_grid(video_bthwc, time_stride=1, space_stride=1):
  """(batch, time, H, W, C) uint8 -> (time, H, batch*W, C) for TensorBoard video.

  ``time_stride``/``space_stride`` subsample frames and pixels before the grid
  reshape — both default to 1 (no-op). Encoding cost scales with output
  pixel count, so a 4× time stride drops report video work ~4×.
  """
  if time_stride > 1:
    video_bthwc = video_bthwc[:, ::time_stride]
  if space_stride > 1:
    video_bthwc = video_bthwc[:, :, ::space_stride, ::space_stride]
  rb, t, h, w, c = video_bthwc.shape
  return video_bthwc.transpose(1, 2, 0, 3, 4).reshape(t, h, rb * w, c)


def vec_to_tb_rgb(vec_bt_d):
  """(B, T, D) float -> (B, T, H, W, 3) uint8 via min-max norm on flattened vector."""
  # Tolerate bool/int inputs (e.g. the discrete manager mask) — min-max needs float.
  vec_bt_d = jnp.asarray(vec_bt_d).astype(jnp.float32)
  d = vec_bt_d.shape[-1]
  h = max(1, int(math.floor(math.sqrt(float(d)))))
  cells = int(math.ceil(d / h) * h)
  w = cells // h
  flat = jnp.pad(vec_bt_d, [(0, 0)] * (vec_bt_d.ndim - 1) + [(0, cells - d)])
  lo = flat.min(axis=-1, keepdims=True)
  hi = flat.max(axis=-1, keepdims=True)
  g = (flat - lo) / (hi - lo + 1e-8)
  g = g.reshape(*vec_bt_d.shape[:-1], h, w, 1)
  u8 = (g * 255).astype(jnp.uint8)
  return jnp.repeat(u8, 3, axis=-1)


def resize_frames(frames_bthwc, target_h, target_w):
  """Nearest-neighbor resize (B, T, H, W, C) uint8 to (B, T, target_h, target_w, C)."""
  B, T, H, W, C = frames_bthwc.shape
  if H == target_h and W == target_w:
    return frames_bthwc
  flat = frames_bthwc.reshape(B * T, H, W, C).astype(jnp.float32)
  resized = jax.image.resize(flat, (B * T, target_h, target_w, C), method='nearest')
  return jnp.clip(resized, 0, 255).reshape(B, T, target_h, target_w, C).astype(jnp.uint8)

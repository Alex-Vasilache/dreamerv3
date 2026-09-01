"""Minimal PNG writer (the env has no Pillow, and this needs ~20 lines)."""
import struct
import zlib

import numpy as np


def encode_png(arr):
  """``(H, W, 3)`` uint8 -> PNG bytes."""
  arr = np.ascontiguousarray(arr, np.uint8)
  assert arr.ndim == 3 and arr.shape[2] == 3, arr.shape
  h, w = arr.shape[:2]
  # Each scanline is prefixed with its filter type; 0 means "no filter".
  raw = np.hstack([np.zeros((h, 1), np.uint8), arr.reshape(h, w * 3)])
  def chunk(tag, data):
    out = struct.pack('>I', len(data)) + tag + data
    return out + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
  return b''.join([
      b'\x89PNG\r\n\x1a\n',
      chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)),
      chunk(b'IDAT', zlib.compress(raw.tobytes(), 6)),
      chunk(b'IEND', b''),
  ])


def montage(frames, cols, pad=2, bg=32, scale=1):
  """``(N, H, W, 3)`` -> one tiled ``(H', W', 3)`` image."""
  frames = np.asarray(frames, np.uint8)
  if scale > 1:
    frames = frames.repeat(scale, 1).repeat(scale, 2)
  n, h, w = frames.shape[:3]
  rows = (n + cols - 1) // cols
  out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3),
                bg, np.uint8)
  for i, frame in enumerate(frames):
    r, c = divmod(i, cols)
    y, x = pad + r * (h + pad), pad + c * (w + pad)
    out[y:y + h, x:x + w] = frame
  return out

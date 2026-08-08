"""`embodied/jax/nets.py` normalization and the `hrl/video.py` helpers.

`Norm` sits on every layer of every network (`norm: rms` throughout), and its
float32 upcast/downcast is what keeps bfloat16 training stable. The `video.py`
helpers had no coverage at all; they only affect reporting, but a wrong reshape
there produces plausible-looking garbage panels rather than an error.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied.jax.nets as nn
from dreamerv3.hrl.video import resize_frames, tb_video_grid, vec_to_tb_rgb

f32 = jnp.float32


# `Norm.__call__` starts with `ensure_dtypes(x)`, which requires the compute
# dtype (bfloat16). Tests therefore feed bfloat16 and compare in float32 with
# tolerances suited to its ~3 decimal digits.
CD = nn.COMPUTE_DTYPE


def bf(x):
  return jnp.asarray(x, CD)


def run_norm(impl, x, **kw):
  layer = nn.Norm(impl, name='norm', **kw)
  fn = lambda v: layer(v)
  params = nj.init(fn)({}, x, seed=0)
  return nj.pure(fn)(params, x, seed=0)[1]


class TestNorm:

  def test_none_is_the_identity(self):
    x = bf(np.random.default_rng(0).normal(size=(2, 5)))
    np.testing.assert_allclose(
        np.asarray(run_norm('none', x), np.float32),
        np.asarray(x, np.float32))

  def test_a_float32_input_is_rejected(self):
    """`ensure_dtypes` pins every layer input to the compute dtype."""
    with pytest.raises(AssertionError):
      run_norm('rms', jnp.ones((2, 4), f32))

  def test_rms_gives_unit_root_mean_square(self):
    rng = np.random.default_rng(1)
    x = bf(rng.normal(0, 7.0, size=(4, 64)))
    got = np.asarray(run_norm('rms', x), np.float32)
    rms = np.sqrt((got ** 2).mean(-1))
    np.testing.assert_allclose(rms, 1.0, rtol=2e-2)

  def test_rms_does_not_remove_the_mean(self):
    x = bf(np.full((2, 32), 5.0))
    got = np.asarray(run_norm('rms', x), np.float32)
    assert np.all(got > 0), 'rms norm must not centre the input'

  def test_layer_removes_the_mean_and_gives_unit_variance(self):
    rng = np.random.default_rng(2)
    x = bf(rng.normal(3.0, 7.0, size=(4, 128)))
    got = np.asarray(run_norm('layer', x), np.float32)
    np.testing.assert_allclose(got.mean(-1), 0.0, atol=3e-2)
    np.testing.assert_allclose(got.std(-1), 1.0, rtol=3e-2)

  def test_an_all_zero_input_does_not_produce_nan(self):
    for impl in ('rms', 'layer'):
      got = np.asarray(run_norm(impl, jnp.zeros((2, 16), CD)), np.float32)
      assert np.all(np.isfinite(got)), impl

  def test_a_constant_input_does_not_produce_nan_under_layer_norm(self):
    got = np.asarray(run_norm('layer', bf(np.full((2, 16), 3.0))), np.float32)
    assert np.all(np.isfinite(got))

  @pytest.mark.parametrize('impl', ['none', 'rms', 'layer'])
  def test_input_dtype_is_returned_unchanged(self, impl):
    x = jnp.asarray(np.random.default_rng(3).normal(size=(2, 8)), jnp.bfloat16)
    assert run_norm(impl, x).dtype == jnp.bfloat16

  def test_the_statistic_is_accumulated_in_float32_not_bfloat16(self):
    """`Norm` upcasts before the reduction; a bf16 sum over 4096 terms drifts.

    Compare the layer's own rms against a float32 reference: agreement well
    inside bf16's own representable step size shows the mean-of-squares was not
    accumulated in bf16.
    """
    rng = np.random.default_rng(4)
    raw = rng.normal(0, 100.0, size=(1, 4096))
    got = np.asarray(run_norm('rms', bf(raw)), np.float32)
    ref32 = raw / np.sqrt((raw ** 2).mean(-1, keepdims=True) + 1e-4)
    np.testing.assert_allclose(got, ref32, rtol=5e-2, atol=5e-2)

  def test_shapes_are_preserved(self):
    for shape in [(2, 8), (2, 3, 8), (2, 3, 4, 8)]:
      assert run_norm('rms', jnp.ones(shape, CD)).shape == shape

  def test_the_eps_suffix_is_parsed_from_the_impl_string(self):
    layer = nn.Norm('rms1em6', name='n')
    assert layer.eps == pytest.approx(1e-6)

  def test_an_unknown_impl_is_rejected(self):
    with pytest.raises(NotImplementedError):
      run_norm('batchnorm', jnp.ones((2, 4), CD))

  def test_rms_is_scale_equivariant(self):
    rng = np.random.default_rng(5)
    x = bf(rng.normal(size=(2, 32)))
    a = np.asarray(run_norm('rms', x), np.float32)
    b = np.asarray(run_norm('rms', (x.astype(f32) * 1000.0).astype(CD)),
                   np.float32)
    np.testing.assert_allclose(a, b, rtol=3e-2)

  def test_gradient_flows_through_the_norm(self):
    layer = nn.Norm('rms', name='norm')
    fn = lambda v: layer(v).sum()
    x = bf(np.random.default_rng(6).normal(size=(2, 8)))
    params = nj.init(fn)({}, x, seed=0)
    g = nj.pure(jax.grad(fn))(params, x, seed=0)[1]
    assert np.any(np.asarray(g, np.float32) != 0.0)


class TestTbVideoGrid:

  def test_batch_is_tiled_along_the_width_axis(self):
    v = jnp.zeros((3, 5, 8, 6, 3), jnp.uint8)
    assert tb_video_grid(v).shape == (5, 8, 3 * 6, 3)

  def test_each_batch_element_lands_in_its_own_column_block(self):
    v = np.zeros((2, 1, 2, 2, 1), np.uint8)
    v[0] = 10
    v[1] = 200
    got = np.asarray(tb_video_grid(jnp.asarray(v)))
    assert np.all(got[0, :, :2, 0] == 10)
    assert np.all(got[0, :, 2:, 0] == 200)

  def test_time_stride_subsamples_frames(self):
    v = jnp.zeros((2, 8, 4, 4, 3), jnp.uint8)
    assert tb_video_grid(v, time_stride=4).shape[0] == 2

  def test_space_stride_subsamples_pixels(self):
    v = jnp.zeros((2, 3, 8, 8, 3), jnp.uint8)
    got = tb_video_grid(v, space_stride=2)
    assert got.shape == (3, 4, 2 * 4, 3)

  def test_strides_of_one_are_a_no_op(self):
    v = jnp.asarray(
        np.random.default_rng(0).integers(0, 255, (2, 3, 4, 4, 3)), jnp.uint8)
    np.testing.assert_array_equal(
        np.asarray(tb_video_grid(v, 1, 1)), np.asarray(tb_video_grid(v)))

  def test_dtype_is_preserved(self):
    assert tb_video_grid(jnp.zeros((2, 3, 4, 4, 3), jnp.uint8)).dtype == jnp.uint8


class TestVecToTbRgb:

  def test_output_is_a_square_ish_rgb_image(self):
    got = vec_to_tb_rgb(jnp.zeros((2, 3, 16), f32))
    assert got.shape == (2, 3, 4, 4, 3)
    assert got.dtype == jnp.uint8

  def test_a_non_square_length_is_padded_to_a_full_grid(self):
    got = vec_to_tb_rgb(jnp.zeros((1, 1, 10), f32))
    h, w = got.shape[2], got.shape[3]
    assert h * w >= 10

  def test_min_maps_to_zero_and_max_maps_to_255(self):
    v = jnp.asarray([[[0.0, 1.0, 2.0, 3.0]]], f32)
    got = np.asarray(vec_to_tb_rgb(v))
    assert got.min() == 0
    assert got.max() == 255

  def test_a_constant_vector_does_not_produce_nan(self):
    got = np.asarray(vec_to_tb_rgb(jnp.full((1, 1, 9), 4.0, f32)))
    assert np.all(np.isfinite(got.astype(np.float32)))

  def test_boolean_input_is_accepted(self):
    got = vec_to_tb_rgb(jnp.zeros((1, 2, 8), bool))
    assert got.dtype == jnp.uint8

  def test_integer_input_is_accepted(self):
    got = vec_to_tb_rgb(jnp.arange(8, dtype=jnp.int32)[None, None])
    assert got.dtype == jnp.uint8

  def test_the_three_channels_are_identical_grayscale(self):
    got = np.asarray(vec_to_tb_rgb(
        jnp.asarray(np.random.default_rng(0).normal(size=(1, 1, 16)), f32)))
    np.testing.assert_array_equal(got[..., 0], got[..., 1])
    np.testing.assert_array_equal(got[..., 1], got[..., 2])

  def test_normalisation_is_per_timestep_not_global(self):
    v = jnp.asarray([[[0.0, 1.0], [0.0, 100.0]]], f32)
    got = np.asarray(vec_to_tb_rgb(v))
    assert got[0, 0].max() == 255 and got[0, 1].max() == 255


class TestResizeFrames:

  def test_a_matching_size_is_returned_unchanged(self):
    v = jnp.zeros((2, 3, 8, 8, 3), jnp.uint8)
    assert resize_frames(v, 8, 8) is v

  def test_upscaling_produces_the_requested_shape(self):
    v = jnp.zeros((2, 3, 4, 4, 3), jnp.uint8)
    assert resize_frames(v, 8, 8).shape == (2, 3, 8, 8, 3)

  def test_downscaling_produces_the_requested_shape(self):
    v = jnp.zeros((1, 2, 16, 16, 3), jnp.uint8)
    assert resize_frames(v, 4, 4).shape == (1, 2, 4, 4, 3)

  def test_output_stays_uint8_and_in_range(self):
    v = jnp.asarray(
        np.random.default_rng(0).integers(0, 256, (1, 2, 4, 4, 3)), jnp.uint8)
    got = np.asarray(resize_frames(v, 9, 9))
    assert got.dtype == np.uint8
    assert got.min() >= 0 and got.max() <= 255

  def test_nearest_neighbour_preserves_exact_pixel_values(self):
    v = np.zeros((1, 1, 2, 2, 1), np.uint8)
    v[0, 0, 0, 0, 0] = 200
    got = np.asarray(resize_frames(jnp.asarray(v), 4, 4))
    assert set(np.unique(got)).issubset({0, 200})

  def test_non_square_targets_are_supported(self):
    v = jnp.zeros((1, 1, 4, 4, 3), jnp.uint8)
    assert resize_frames(v, 2, 8).shape == (1, 1, 2, 8, 3)

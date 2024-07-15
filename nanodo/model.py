# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer Decoder-only model."""

# pylint: disable=g-importing-member
# pylint: disable=invalid-name

import dataclasses
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp

from nanodo import fsdp


@dataclasses.dataclass
class DoConfig:
  """Hyper-parameters for Transformer decoder-only."""
  D: int  # model/embed dim  = qkv dim
  H: int  # num attention heads
  L: int  # max context/sequence length (move out of config?)
  N: int  # number of transformer block layers
  V: int  # vocab size
  F: int  # FF inner dimension
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
  embed_init: nn.initializers.Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal', out_axis=0)
  dtype: jnp.dtype = jnp.float32
  fsdp_enabled: bool = True

  # Transformer block rematerialization / gradient checkpointing to save memory.
  remat: bool = False


class TransformerDo(nn.Module):
  """Transformer decoder-only."""
  docfg: DoConfig

  def setup(self):
    cfg = self.docfg
    self.embed = nn.Embed(
        num_embeddings=cfg.V,
        features=cfg.D,
        embedding_init=fsdp.init('embedding', cfg),
    )

    block = nn.remat(TBlock) if cfg.remat else TBlock
    self.blocks = [block(cfg) for _ in range(cfg.N)]
    self.out_ln = nn.RMSNorm(dtype=cfg.dtype, use_bias=False)

  def __call__(self, y_BxL: jax.Array):
    # For training on concatenated examples.
    y_BxLxD = self.embed(y_BxL)
    positions = jnp.arange(0, y_BxL.shape[1])[None, ...]
    for block in self.blocks:
      y_BxLxD = block(y_BxLxD, positions)
    y_BxLxD = self.out_ln(y_BxLxD)
    logits_BxLxV = self.embed.attend(y_BxLxD.astype(jnp.float32))
    return logits_BxLxV


class Mlp(nn.Module):
  """Multilayer perceptron."""
  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg
    linear = partial(
        nn.Dense, kernel_init=fsdp.init('mlp_kernel', cfg), use_bias=False,
        dtype=cfg.dtype
    )
    x_BxLxF = linear(cfg.F * 2)(x_BxLxD)
    gate, proj = x_BxLxF.split(2, axis=-1)
    x_BxLxF = jax.nn.swish(gate) * proj
    x_BxLxD = linear(cfg.D)(x_BxLxF)
    return x_BxLxD


class TBlock(nn.Module):
  """Transformer Block."""
  docfg: DoConfig

  @nn.compact
  def __call__(self, in_BxLxD: jax.Array, positions: jax.Array):
    cfg = self.docfg

    # "pre-layernorm"
    x_BxLxD = nn.RMSNorm(dtype=cfg.dtype, use_bias=False)(in_BxLxD)
    x_BxLxD = CausalAttn(cfg)(x_BxLxD, positions)
    x_BxLxD += in_BxLxD

    z_BxLxD = nn.RMSNorm(dtype=cfg.dtype, use_bias=False)(x_BxLxD)
    z_BxLxD = Mlp(cfg)(z_BxLxD)

    return x_BxLxD + z_BxLxD


def apply_rope(
    inputs: jax.Array,    # [B, L]
    positions: jax.Array, # [B, L]
    head_dim: int,
    max_wavelength: int = 10_000,
) -> jax.Array:
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = max_wavelength**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


class CausalAttn(nn.Module):
  """Causal attention layer."""
  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array, positions: jax.Array):
    cfg = self.cfg

    assert cfg.D % cfg.H == 0, f'D {cfg.D} not divisible by H {cfg.H}'
    Dh = cfg.D // cfg.H

    # Maps D -> (H, Dh)
    multilinear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=fsdp.init('attn_in_proj', cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )

    q_BxLxHxDh, k_BxLxHxDh, v_BxLxHxDh = (
        multilinear(name='query')(x_BxLxD),
        multilinear(name='key')(x_BxLxD),
        multilinear(name='value')(x_BxLxD),
    )
    q_BxLxHxDh = apply_rope(q_BxLxHxDh, positions, Dh)
    k_BxLxHxDh = apply_rope(k_BxLxHxDh, positions, Dh)
    q_BxLxHxDh /= Dh**0.5
    att_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxLxHxDh)
    # cast to fp32 for softmax
    att_BxHxLxL = att_BxHxLxL.astype(jnp.float32)

    # causal attention mask
    L = x_BxLxD.shape[1]
    mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))

    _NEG_INF = jnp.finfo(cfg.dtype).min
    att_BxHxLxL = jnp.where(mask_1x1xLxL, att_BxHxLxL, _NEG_INF)
    att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
    att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)
    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxL, v_BxLxHxDh)
    # Output projection followed by contraction back to original dims
    out_BxLxD = nn.DenseGeneral(
        features=cfg.D,
        name='attn_out_proj',
        axis=(-2, -1),
        kernel_init=fsdp.init('attn_out_proj', cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )(out_BxLxHxDh)
    return out_BxLxD

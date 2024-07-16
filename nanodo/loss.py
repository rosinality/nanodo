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
"""Loss functions."""

# pylint: disable=invalid-name,g-import-not-at-top,g-bare-generic

from typing import Any, Callable, TYPE_CHECKING

from flax.struct import dataclass
import jax
import jax.numpy as jnp
from nanodo import data
from optax import losses
from typing import Tuple

if TYPE_CHECKING:
    import ml_collections


PyTree = Any


@dataclass
class LossAuxData:
    ntokens: jax.Array
    state: PyTree
    log_perplexity: jax.Array
    z_loss: jax.Array


# loss(params) function to be used in `jax.value_and_grad`.
LossFn = Callable[[PyTree], tuple[jax.Array, LossAuxData]]

LossFnFactory = Callable[
    [jax.Array, Callable, "ml_collections.ConfigDict"],
    LossFn,
]


@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes cross entropy loss with stable custom gradient.
    Computes a stabilized-gradient version of:
      -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
    If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
    will be added to the cross entropy loss (z = softmax normalization constant).
    The two uses of z_loss are:
    1. To keep the logits from drifting too far from zero, which can cause
       unacceptable roundoff errors in bfloat16.
    2. To encourage the logits to be normalized log-probabilities.
    Args:
      logits: [batch, length, num_classes] float array.
      targets: categorical one-hot targets [batch, length, num_classes] float
        array.
      z_loss: coefficient for auxiliary z-loss loss term.
    Returns:
      tuple with the total loss and the z_loss, both
      float arrays with shape [batch, length].
    """
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxiliary z-loss term.
    log_z = jnp.squeeze(logits_sum, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss
    return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray],
    Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
]:
    """Forward-mode of `cross_entropy_with_logits`."""
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxiliary z-loss term.
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss
    return (loss, total_z_loss), (
        logits,
        targets,
        z_loss,
        exp_shifted,
        sum_exp,  # pytype: disable=bad-return-type  #jax-ndarray
        log_softmax,
        log_z,
    )


def _cross_entropy_with_logits_bwd(
    res: Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    g: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-mode of `cross_entropy_with_logits`."""
    g = g[0]  # Ignore z_loss component as that is only used for logging.
    logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
    # z-loss term adds the (2 * z_loss * log_z) factor.
    deriv = (
        jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
    )
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
    return (
        jnp.asarray(g_logits, logits.dtype),
        jnp.asarray(g_targets, targets.dtype),
        jnp.array(0.0),
    )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd
)


def get_default_loss_fn(
    in_BxL: jax.Array,
    apply_fn: Callable,
    c: "ml_collections.ConfigDict",
    vocab_size: int,
) -> LossFn:
    """Standard next-token-prediction language modeling loss."""

    def loss_fn(params: PyTree) -> tuple[jax.Array, LossAuxData]:
        x_BxL, y_BxL, weights_BxL = data.get_in_out(in_BxL)

        mutable = ("intermediate_acts",) if c.get("log_internal_metrics", False) else ()
        logits_BxLxV, state = apply_fn(
            {"params": params},
            x_BxL,
            mutable=mutable,
        )

        # losses_BxL = losses.softmax_cross_entropy_with_integer_labels(
        #     logits_BxLxV, y_BxL
        # )
        onehot = jax.nn.one_hot(y_BxL, vocab_size)
        losses_BxL, z_loss_BxL = cross_entropy_with_logits(
            logits_BxLxV, onehot, c.model.z_loss
        )
        ntokens = weights_BxL.sum()
        mean_loss = jnp.sum(losses_BxL * weights_BxL) / ntokens
        mean_z_loss = jnp.sum(z_loss_BxL * weights_BxL) / ntokens

        return mean_loss, LossAuxData(
            ntokens=ntokens,
            state=state,
            log_perplexity=mean_loss - mean_z_loss,
            z_loss=mean_z_loss,
        )

    return loss_fn

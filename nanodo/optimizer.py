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
"""Optimizer."""

# pylint: disable=g-import-not-at-top

import functools
from typing import Iterable, TYPE_CHECKING, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import transform
from jax import tree_util as jtu
from optax._src import base, numerics, wrappers, combine

if TYPE_CHECKING:
    import ml_collections


def get_optimizer(c: "ml_collections.ConfigDict") -> optax.MultiSteps:
    """Get optimizer."""
    optimizer = _get_base_optimizer(c)

    if c.get("layerwise_lr_multiplier", None) is not None:
        scale_dict = dict(c.layerwise_lr_multiplier)
        optimizer = optax.chain(optimizer, _scale_by_dict(scale_dict))

    clip_by_global_norm = c.get("clip_by_global_norm", None)
    if clip_by_global_norm:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_by_global_norm), optimizer
        )

    # Multistep gradient accumulation
    optimizer = optax.MultiSteps(optimizer, c.get("grad_accumulation_steps", 1))

    return optimizer


def get_learning_rate_schedule(
    c: "ml_collections.ConfigDict", multiplier=1, constant_start=False
) -> optax.Schedule:
    """Creates a learning rate schedule based on the config."""

    init_value = c.init_learning_rate * multiplier
    
    if constant_start:
        init_value = c.peak_learning_rate * multiplier

    schedules = [
        optax.linear_schedule(
            init_value=init_value,
            end_value=c.peak_learning_rate * multiplier,
            transition_steps=c.warmup_steps,
        )
    ]

    decay_type = c.get("decay_type", "cosine")

    if decay_type == "rsqrt":
        schedules.append(
            _rsqrt_schedule(
                init_value=c.peak_learning_rate,
                shift=1 + c.warmup_steps,
            )
        )

    elif decay_type == "cosine":
        decay_steps = c.get("decay_steps", c.num_train_steps - c.warmup_steps)
        schedules.append(
            optax.cosine_decay_schedule(
                init_value=c.peak_learning_rate * multiplier,
                decay_steps=decay_steps,
                alpha=c.final_learning_rate / c.peak_learning_rate,
                exponent=1.0,
            )
        )

    elif decay_type == "linear":
        schedules.append(
            optax.linear_schedule(
                init_value=c.peak_learning_rate,
                end_value=c.final_learning_rate,
                transition_steps=c.num_train_steps - c.warmup_steps,
            )
        )

    elif decay_type == "constant_without_warmup":
        return optax.constant_schedule(value=c.peak_learning_rate)

    elif decay_type == "constant":
        schedules.append(optax.constant_schedule(value=c.peak_learning_rate))

    elif decay_type.startswith("constant_linear_decay_"):
        if decay_type.endswith("p"):
            percent_decay = float(decay_type.split("_")[-1].split("p")[0]) / 100
            if percent_decay < 0 or percent_decay > 1:
                raise ValueError(f"Invalid decay % provided in {decay_type}")
            transition_steps = int(c.num_train_steps * percent_decay)
        else:
            decay_steps = int(decay_type.split("_")[-1])
            if decay_steps < 0 or decay_steps > c.num_train_steps:
                raise ValueError(f"Invalid decay steps provided in {decay_type}")
            transition_steps = decay_steps
        schedules += [
            optax.constant_schedule(value=c.peak_learning_rate),
            optax.linear_schedule(
                init_value=c.peak_learning_rate,
                end_value=c.final_learning_rate,
                transition_steps=transition_steps,
            ),
        ]
        return optax.join_schedules(
            schedules, boundaries=[c.warmup_steps, c.num_train_steps - transition_steps]
        )

    else:
        raise NotImplementedError(f"Unsupported decay type: {c.decay_type}")

    return optax.join_schedules(schedules, boundaries=[c.warmup_steps])


def _rsqrt_schedule(*, init_value: float, shift: int) -> optax.Schedule:
    """Constructs a schedule with reciprocal sqrt decay."""

    def schedule(count):
        return init_value * (count + shift) ** -0.5 * shift**0.5

    return schedule


def _params_mask(
    params: optax.Params, exclude_names: Iterable[str] = ("bias", "scale")
) -> optax.Params:
    """Generate boolean mask for params PyTree with `exclude_names` parameters."""

    def _check_key_contain_exclude_names(key_path):
        return any(
            [
                x in "/".join([k.key for k in key_path if hasattr(k, "key")])
                for x in exclude_names
            ]
        )

    # Mask should return True for parameters that does not match patterns inside
    # `exclude_names`.
    return jax.tree_util.tree_map_with_path(
        lambda key_path, _: not _check_key_contain_exclude_names(key_path), params
    )


def add_decayed_weights(weight_decay=0.0, mask=None) -> base.GradientTransformation:
    """Add parameter scaled by `weight_decay`.

    Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the transformation to, and `False` for those you want to skip.

    Returns:
    A `GradientTransformation` object.
    """

    if callable(weight_decay):
        return weight_decay_by_schedule(weight_decay, mask)

    return transform.add_decayed_weights(weight_decay, mask)


class WeightDecayByScheduleState(NamedTuple):
    """Maintains count for scale scheduling."""

    count: chex.Array  # shape=(), dtype=jnp.int32


def weight_decay_by_schedule(weight_decay_fn, mask):
    def init_fn(params):
        del params
        return WeightDecayByScheduleState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        weight_decay = weight_decay_fn(state.count)
        updates = jtu.tree_map(
            lambda g, p: g + jnp.array(weight_decay, dtype=p.dtype) * p, updates, params
        )
        return updates, WeightDecayByScheduleState(
            count=numerics.safe_int32_increment(state.count)
        )

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return wrappers.masked(base.GradientTransformation(init_fn, update_fn), mask)

    return base.GradientTransformation(init_fn, update_fn)


def adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype=None,
    weight_decay: float = 1e-4,
    mask=None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
    return combine.chain(
        transform.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
    )


def _get_base_optimizer(
    c: "ml_collections.ConfigDict",
) -> optax.GradientTransformation:
    """Get base optimizer."""
    learning_rate_fn = get_learning_rate_schedule(c)
    optimizer_type = c.optimizer
    weight_decay_exclusion_names = c.get("weight_decay_exclusion_names", [])
    if c.get("independent_weight_decay", False):
        weight_decay = c.weight_decay / c.peak_learning_rate
    else:
        weight_decay = c.weight_decay

    if c.get("weight_decay_lr_exponent", False):
        constant_start = c.get("weight_decay_constant_start", False)
        weight_decay = get_learning_rate_schedule(c, multiplier=weight_decay, constant_start=constant_start)
        print("using weight_decay_lr_exponent")

    if optimizer_type == "adafactor":
        base_optimizer = optax.adafactor(
            learning_rate_fn,
            multiply_by_parameter_scale=c.get("multiply_by_parameter_scale", True),
            decay_rate=c.get("decay_rate", 0.8),
            momentum=c.get("momentum", None),
            factored=c.get("factored", True),
            eps=c.get("eps", 1e-30),
            weight_decay_rate=c.weight_decay,
            weight_decay_mask=functools.partial(
                _params_mask, exclude_names=weight_decay_exclusion_names
            ),
        )

    elif optimizer_type == "adamw":
        b1 = c.get("b1", 0.9)
        b2 = c.get("b2", 0.98)

        base_optimizer = adamw(
            learning_rate_fn,
            b1=c.get("b1", 0.9),
            b2=c.get("b2", 0.98),
            eps=c.get("eps", 1e-9),
            weight_decay=weight_decay,
            mask=functools.partial(
                _params_mask, exclude_names=weight_decay_exclusion_names
            ),
        )

        print(f"AdamW b1 = {b1} b2 = {b2}")

    elif optimizer_type == "lion":
        base_optimizer = optax.lion(
            learning_rate_fn,
            b1=c.get("b1", 0.9),
            b2=c.get("b2", 0.98),
            weight_decay=weight_decay,
            mask=functools.partial(
                _params_mask, exclude_names=weight_decay_exclusion_names
            ),
        )

    else:
        raise ValueError(optimizer_type)

    return base_optimizer


def _scale_by_dict(scale_dict: dict[str, float]) -> optax.GradientTransformation:
    """Optax transform for performing layerwise learning rate rescaling.

    Args:
      scale_dict: a dictionary that determines which parameters to apply
      learning rate rescaling, e.g., {"kernel": 3.} means using a 3X learning rate
      for all parameters whose name contain "kernel".

    Returns:
      An Optax transform suitable for chaining (should be applied after the
      optimizer).
    """

    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params

        def scale(keys, x):
            # Convert to str "module_name_1/module_name_2/.../kernel"
            str_keys = "/".join([k.key for k in keys if hasattr(k, "key")])
            for which_to_rescale, multiplier in scale_dict.items():
                if which_to_rescale in str_keys:
                    return x * multiplier
            return x

        updates = jax.tree_util.tree_map_with_path(scale, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)

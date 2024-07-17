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
"""Default Hyperparameter configuration.

Usage:
/bin/bash third_party/py/nanodo/run.sh --config=default
"""

import math

from flax import linen as nn
import ml_collections

from nanodo.model import orthogonal_init, flops_per_token, model_params


def get_config() -> ml_collections.ConfigDict:
    """Get the default hyperparameter configuration."""
    cfg = ml_collections.ConfigDict()
    cfg.seed = 42

    # Data
    cfg.batch_size = 56 * 2  # Global batch size. Must be divisible by the #devices.
    cfg.train_epochs = None  # None=>infinite
    cfg.ds_name = "scripts/fileinstructions.json"
    cfg.vocab_path = "tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model"  # set to local-path
    cfg.eval_batch_size = 256

    dim = 288
    n_layer = 8
    seq_len = 1024
    
    cfg.scale = 5
    base_flops = 124611846576537600
    flops = flops_per_token(n_layer, dim, seq_len)
    params = model_params(n_layer, dim, 32101)
    cfg.base_train_steps = base_flops / flops / seq_len / cfg.batch_size
    cfg.flops_multiplier = 1

    # Transformer
    cfg.model = ml_collections.config_dict.create(
        D=dim,  # model/embed dim  = qkv dim
        H=4,  # num attention heads
        L=seq_len,  # max context/sequence length (move out of config?)
        N=n_layer,  # number of transformer block layers
        F=int(dim * 3.5),  # FF inner dimension
        dtype="bfloat16",  # computation dtype.
        fsdp_enabled=True,  # True to shard the model.
        remat=False,  # Transformer block gradient checkpointing to save memory.
        kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
        kernel_init_str="fan_in-1.0",
        # kernel_init=nn.initializers.xavier_uniform(),
        output_kernel_init=nn.initializers.variance_scaling(
            1.0 / n_layer, "fan_in", "normal"
        ),
        output_kernel_init_str="fan_in-1/n_layer",
        # output_kernel_init=nn.initializers.xavier_uniform(),
        head_init=nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
        head_init_str="fan_in-1.0",
        embed_init=nn.initializers.normal(0.01),
        # embed_init=nn.initializers.variance_scaling(
        #     1.0, "fan_in", "normal", out_axis=0
        # ),
        embed_init_str="normal-0.01",
        z_loss=1e-4,
        attn_logit_softcapping=50,
    )

    # Optimizer
    cfg.opt = ml_collections.config_dict.create(
        num_train_steps=math.ceil(cfg.base_train_steps),  # Note: lm1b has 30,301,028 training examples
        peak_learning_rate=3e-3,
        init_learning_rate=0,
        final_learning_rate=3e-4,
        warmup_steps=math.ceil(params / seq_len / cfg.batch_size),
        decay_type="cosine",
        weight_decay=1e-4,
        clip_by_global_norm=1.0,  # 1.0 is common for many well-known LLMs.
        optimizer="adamw",
        independent_weight_decay=True,
        weight_decay_exclusion_names=("bias", "scale"),
        b2=0.99,
        layerwise_lr_multiplier={"kernel": 384 / dim},
    )

    # Checkpointing
    cfg.workdir = "/home/rosinality/results"
    cfg.checkpoint = True
    cfg.checkpoint_every_steps = 2000
    # Path to the checkpoint to be restored. Note than new checkpoints will be
    # saved to the new workdir.
    cfg.checkpoint_restore_dir = None
    cfg.max_to_keep = 100

    # Eval
    cfg.eval_every_steps = 500
    cfg.eval_split = "test"  # 306,688 examples
    cfg.eval_steps = 100  # less if this exceeds 1 epoch
    cfg.eval_max_target_length = 512

    # Logging
    cfg.write_train_metrics_every_steps = 1  # train loss, gradient norms, etc.
    cfg.write_perf_metrics_every_steps = 100  # steps_per_sec, uptime.
    # For Vizier interface, we currently require write_to_xm_measurements=True
    cfg.write_to_xm_measurements = True
    # Option to turn on internal statistics: rms_norm, mean, std of per-layer,
    # module-wise statistics. Due to high-load, when setting this to True consider
    # turning off writing to XM measurements and rely on Datatables.
    cfg.log_internal_metrics = True

    # pygrain
    cfg.pygrain_worker_count = 16  # might increase this if input-bound
    # Buffer size (in unit of batches) for the data loader. Default to 2 so we
    # always prefetch another batch
    cfg.pygrain_worker_buffer_size = 2

    return cfg

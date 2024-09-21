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
"""Data pipeline."""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
from typing import Iterator
import math
import json

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

import sentencepiece as spm

from nanodo.multihost_dataloading import MultiHostDataLoadIterator

PAD_ID = 0
### pure python helpers for use with grain ###


class Preprocess(enum.Enum):
    NOAM_PACKED = 1
    PADDED = 2


@dataclasses.dataclass
class FileInstruction:
    filename: str
    skip: int
    take: int
    examples_in_shard: int


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
    def map(self, features):
        def _parse(example):
            parsed = tf.io.parse_example(
                example, {"text": tf.io.FixedLenFeature(shape=(), dtype=tf.string)}
            )

            return parsed

        return _parse(features)


@dataclasses.dataclass
class DecodeFeatures(grain.MapTransform):
    def map(self, features):
        return {"text": features["text"].numpy().decode("utf-8")}


def py_batched_tfds(
    *,
    fileinstructions: str,
    split: str,
    context_size: int,
    worker_count: int,
    vocab_path: str,
    batch_size: int,
    global_mesh,
    seed: int | None = 1234,
    num_epochs: int | None = None,
    num_records: int | None = None,
    preprocessing: Preprocess = Preprocess.NOAM_PACKED,
    worker_buffer_size: int = 2,
    shuffle: bool = True,
) -> grain.DataLoader:
    """Returns iterator for regularly batched text examples."""
    # datasource = tfds.data_source(tfds_name, split=split)

    with open(fileinstructions, "r") as f:
        fileinstructions = json.load(f)

    instructions = []
    for instruction in fileinstructions:
        shard_size = instruction["examples_in_shard"]
        eval_size = math.ceil(shard_size * 0.0001)
        if split == "train":
            instructions.append(
                FileInstruction(
                    instruction["filename"],
                    eval_size,
                    shard_size - eval_size,
                    shard_size,
                )
            )

        else:
            instructions.append(
                FileInstruction(instruction["filename"], 0, eval_size, shard_size)
            )

    datasource = grain.ArrayRecordDataSource(instructions)

    index_sampler = grain.IndexSampler(
        num_records=num_records if num_records is not None else len(datasource),
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=True,
        ),
        shuffle=shuffle,
        seed=seed,
    )
    spt = _SPTokenizer(vocab_path)

    pad_len = None if preprocessing == Preprocess.NOAM_PACKED else context_size
    pygrain_ops = [
        ParseFeatures(),
        DecodeFeatures(),
        Tokenize(spt, pad_len),
    ]
    if preprocessing == Preprocess.NOAM_PACKED:
        pygrain_ops.append(_NoamPack(context_size=context_size))
    elif preprocessing == Preprocess.PADDED:
        pygrain_ops.append(grain.MapOperation(map_function=np.array))
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    pygrain_ops.append(
        grain.Batch(batch_size=batch_size // jax.process_count(), drop_remainder=True)
    )
    batched_dataloader = grain.DataLoader(
        data_source=datasource,
        operations=pygrain_ops,
        sampler=index_sampler,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )
    return MultiHostDataLoadIterator(batched_dataloader, global_mesh)


def get_py_tokenizer(path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    assert sp.pad_id() == PAD_ID
    assert sp.eos_id() != -1
    assert sp.bos_id() != -1
    return sp


# Need this because we can't pickle SentencePieceProcessor object
class _SPTokenizer:
    """Wrapper class for SentencePiece tokenizer."""

    def __init__(self, vocab_path):
        self._tokenizer = None
        self._vocab_path = vocab_path

    def get_tokenizer(self) -> spm.SentencePieceProcessor:
        if not self._tokenizer:
            self._tokenizer = get_py_tokenizer(self._vocab_path)
        return self._tokenizer


@dataclasses.dataclass
class Tokenize(grain.MapTransform):
    def __init__(self, tokenizer, pad_len):
        self.tokenizer = tokenizer
        self.pad_len = pad_len

    def map(
        self,
        features: Mapping[str, str],
        pad_id: int = PAD_ID,
    ) -> Sequence[int]:
        """Tokenizes text into ids, optionally pads or truncates to pad_len."""
        text = features["text"]
        tokenizer = self.tokenizer.get_tokenizer()
        bos_id = tokenizer.bos_id()
        eos_id = tokenizer.eos_id()
        ids = tokenizer.EncodeAsIds(text)

        ids.insert(0, bos_id)
        ids.append(eos_id)
        if self.pad_len is not None:
            if len(ids) < self.pad_len:
                ids.extend([pad_id] * (self.pad_len - len(ids)))
            elif len(ids) > self.pad_len:
                ids = ids[: self.pad_len]
        return ids


@dataclasses.dataclass
class _NoamPack:
    """Pygrain operation for tokenizing and Noam packing text."""

    context_size: int

    def __call__(
        self, idseq_iterator: Iterator[grain.Record]
    ) -> Iterator[grain.Record]:
        packed_ids = []
        for input_record in idseq_iterator:
            start = 0
            while start < len(input_record.data):
                rem_data = input_record.data[start:]
                if len(packed_ids) + len(rem_data) < self.context_size:
                    packed_ids.extend(rem_data)  # use rest of example, move-on
                    break
                else:
                    take = self.context_size - len(packed_ids)
                    packed_ids.extend(rem_data[:take])
                    last_record_key = input_record.metadata.remove_record_key()
                    yield grain.Record(
                        last_record_key, np.array(packed_ids, dtype=np.int32)
                    )
                    start += take
                    packed_ids = []
                    # Drop remainder for simplicity.
                    # We lose the rest of the example on restore.


# pylint: disable=invalid-name


def get_in_out(
    in_BxL: jax.Array,
    pad_id: int = PAD_ID,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns input, output, and weights for a batch of examples."""
    # Assumes input of the form <BOS> <IDs> <EOS> for eval.
    x_BxL = in_BxL
    y_BxL = jnp.pad(
        in_BxL[:, 1:],
        ((0, 0), (0, 1)),
        mode="constant",
        constant_values=pad_id,
    )
    weights_BxL = jnp.where(y_BxL != pad_id, 1, 0).astype(jnp.float32)

    return x_BxL, y_BxL, weights_BxL

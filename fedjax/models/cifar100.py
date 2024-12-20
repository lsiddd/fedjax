# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CIFAR-100 models."""

import jax
from fedjax.core import metrics
from fedjax.core import models

import haiku as hk
import numpy as np

# Defines the expected structure of input batches to the model. This is used to
# determine the model parameter shapes.
_HAIKU_SAMPLE_BATCH = {
    'x': np.zeros((1, 32, 32, 3), dtype=np.float32),
    'y': np.zeros(1, dtype=np.int32)
}
_TRAIN_LOSS = lambda batch, params: metrics.unreduced_cross_entropy_loss(
    batch['y'], params)
_EVAL_METRICS = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy(),
}


def create_logistic_model() -> models.Model:
  """Creates CIFAR-100 logistic model with haiku."""
  num_classes = 100

  def forward_pass(batch):
    network = hk.Sequential([
        hk.Flatten(),
        hk.Linear(num_classes),
    ])
    return network(batch['x'])

  transformed_forward_pass = hk.transform(forward_pass)
  return models.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=_HAIKU_SAMPLE_BATCH,
      train_loss=_TRAIN_LOSS,
      eval_metrics=_EVAL_METRICS)



def preprocess_batch(batch):
    """Preprocesses the input batch (e.g., normalization)."""
    batch['x'] = (batch['x'] - 127.5) / 127.5  # Normalize pixel values to [-1, 1]
    return batch

_TRAIN_LOSS = lambda batch, params: metrics.unreduced_cross_entropy_loss(
    batch['y'], params)
_EVAL_METRICS = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy(),
}

def create_cnn_model() -> models.Model:
    """Creates an improved CNN model for CIFAR-100."""
    num_classes = 100

    def forward_pass(batch):
        batch = preprocess_batch(batch)  # Apply preprocessing
        network = hk.Sequential([
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='SAME'),  # Conv Layer 1
            jax.nn.relu,                                                             # Activation
            hk.MaxPool(window_shape=2, strides=2, padding='SAME'),                   # Pooling

            hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='SAME'), # Conv Layer 2
            jax.nn.relu,
            hk.MaxPool(window_shape=2, strides=2, padding='SAME'),

            hk.Flatten(),                                                            # Flatten
            hk.Linear(256),                                                          # Fully Connected Layer
            jax.nn.relu,
            hk.Linear(num_classes),                                                  # Output Layer
        ])
        return network(batch['x'])

    transformed_forward_pass = hk.transform(forward_pass)
    return models.create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch=_HAIKU_SAMPLE_BATCH,
        train_loss=_TRAIN_LOSS,
        eval_metrics=_EVAL_METRICS
    )
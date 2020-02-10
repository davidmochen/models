# Lint as: python3
"""Benchmarks for Resnet50 inference."""

import os
import time

# pylint: disable=g-bad-import-order
import numpy as np
import tensorflow as tf

from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark

_RESNET50_INFERENCE_FOLDER = 'gs://tf-perfzero-data/resnet/resnet50/inference'


def _get_input_constant(shape, dtype):
  input_value = np.random.random_sample(shape).astype(dtype)
  input_fp = tf.constant(input_value)
  return input_fp


def _get_saved_model(saved_model_dir):
  with tf.device('/GPU:0'):
    root = tf.saved_model.load(saved_model_dir)
  return root


class Resnet50InferenceBenchmark(PerfZeroBenchmark):
  """Benchmarks for Resnet50 inference."""

  def __init__(self, output_dir=None, root_data_dir=None, tpu=None):
    super(Resnet50InferenceBenchmark, self).__init__(
        output_dir=output_dir, tpu=tpu)

  def run(self, saved_model_dir):
    shape = [1, 224, 224, 3]
    input_tensor = _get_input_constant(shape, np.float32)
    root = _get_saved_model(saved_model_dir)
    concrete_func = root.signatures['serving_default']

    # warmup runs
    with tf.device('/GPU:0'):
      for _ in range(5):
        concrete_func(input_tensor)

    # benchmark runs
    start_time_sec = time.time()
    with tf.device('/GPU:0'):
      for _ in range(100):
        concrete_func(input_tensor)
    wall_time_sec = time.time() - start_time_sec

    self.report_benchmark(iters=100, wall_time=wall_time_sec)

  def benchmark_1_gpu_channels_last_fp32(self):
    self.run(
        saved_model_dir=os.path.join(
            _RESNET50_INFERENCE_FOLDER,
            'keras_resnet50_gpu_1_fp32_channels_last_eager_graph_cfit',
            'saved_model'))

  def benchmark_1_gpu_channels_first_fp32(self):
    self.run(
        saved_model_dir=os.path.join(
            _RESNET50_INFERENCE_FOLDER,
            'keras_resnet50_gpu_1_fp32_channels_first_eager_graph_cfit',
            'saved_model'))

  def benchmark_1_gpu_channels_last_fp16(self):
    self.run(
        saved_model_dir=os.path.join(
            _RESNET50_INFERENCE_FOLDER,
            'keras_resnet50_gpu_1_fp16_channels_last_eager_graph_cfit',
            'saved_model'))

  def benchmark_1_gpu_channels_first_fp16(self):
    self.run(
        saved_model_dir=os.path.join(
            _RESNET50_INFERENCE_FOLDER,
            'keras_resnet50_gpu_1_fp16_channels_first_eager_graph_cfit',
            'saved_model'))


if __name__ == '__main__':
  tf.test.main()

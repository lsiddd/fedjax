"""Tests for fedjax.core.prefetch."""

from concurrent import futures
import time

from absl.testing import absltest
from fedjax.core import prefetch
import tensorflow as tf


class AsyncTFDatasetIteratorTest(absltest.TestCase):

  def test_one_pass(self):
    values = []
    with futures.ThreadPoolExecutor(1) as executor:
      for i in prefetch.AsyncTFDatasetIterator(
          executor, lambda: tf.data.Dataset.range(5)):
        values.append(i)
    self.assertEqual(values, [0, 1, 2, 3, 4])

  def test_dataset_fn_exception(self):

    def dataset_fn():
      raise ValueError

    with futures.ThreadPoolExecutor(1) as executor:
      it = prefetch.AsyncTFDatasetIterator(executor, dataset_fn)
      with self.assertRaises(ValueError):
        next(it)

  def test_elapsed_time(self):
    # This test may fail if we do not build with -c opt.
    create_time = 0.5
    next_time = 0.2

    def work(x):
      time.sleep(next_time)
      return x

    def tf_work(x):
      return tf.py_function(work, [x], tf.int64)

    def dataset_fn():
      time.sleep(create_time)
      return tf.data.Dataset.range(5).map(tf_work)

    with futures.ThreadPoolExecutor(1) as executor:
      start = time.perf_counter()
      it = prefetch.AsyncTFDatasetIterator(executor, dataset_fn)
      time.sleep(create_time + next_time)
      next(it)
      time_to_first_value = time.perf_counter() - start
    # (create_time + next_time) * 2 is how long it takes to execute the code
    # above synchronously.
    self.assertLess(time_to_first_value, (create_time + next_time) * 2)


class PrefetchClientDatasetsIteratorTest(absltest.TestCase):

  def test_one_pass(self):

    def dataset_fn_0(client_id):
      return tf.data.Dataset.range(int(client_id))

    def dataset_fn_1(client_id):
      return tf.data.Dataset.range(int(client_id) + 1)

    client_data = (dataset_fn_0, dataset_fn_1)
    client_ids = ['0', '1', '2']
    values = list(
        prefetch.PrefetchClientDatasetsIterator(client_data, client_ids))
    self.assertLen(values, 3)
    it00, it01 = values[0]
    self.assertEqual(list(it00), [])
    self.assertEqual(list(it01), [0])
    it10, it11 = values[1]
    self.assertEqual(list(it10), [0])
    self.assertEqual(list(it11), [0, 1])
    it20, it21 = values[2]
    self.assertEqual(list(it20), [0, 1])
    self.assertEqual(list(it21), [0, 1, 2])


if __name__ == '__main__':
  absltest.main()

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from model.architectures.transformations import spatial_dropout, compose_transformation, \
    history_crop, add_normal_bias, history_cutout


class TransformationTest(parameterized.TestCase, tf.test.TestCase):

    def assert_not_nan(self, tensor):
        return self.assertTrue(np.all(~tf.math.is_nan(tensor)))

    @parameterized.named_parameters(('spatial_do_10', 0.1),
                                    ('spatial_do_50', 0.5))
    def test_spatial_dropout(self, rate):
        tf_generator = tf.random.Generator.from_seed(1234)
        batch = tf.ones((2, 48, 41))
        dropped_out = spatial_dropout(batch, tf_generator, rate)
        self.assertNotAllEqual(dropped_out[0], dropped_out[1])
        self.assertEqual(len(np.unique(tf.reduce_sum(dropped_out[0], axis=1))), 1)
        count = 0
        for k in range(300):
            dropped_out = spatial_dropout(batch, tf_generator, rate)
            count += tf.reduce_sum(dropped_out) / (2 * 48 * 41)
        count /= 300
        self.assertAlmostEqual(1 - count, rate, delta=1e-2)

    @parameterized.named_parameters(('low_hist_high_prob', 0.1, 1.0),
                                    ('mid_hist_high_prob', 0.5, 1.0),
                                    ('mid_high_prob', 0.5, 0.5))
    def test_history_crop(self, min_history, proba):
        tf_generator = tf.random.Generator.from_seed(1234)
        batch = tf.ones((1, 1000, 40))
        cropped_seq = 0
        for k in range(1000):
            seq = history_crop(batch, tf_generator, p=proba, min_history=min_history)
            if np.any(seq.numpy() == 0):
                cropped_seq += 1

            self.assertTrue(np.all(seq[0, -int(1000 * min_history):] == 1))
        self.assertAlmostEqual(cropped_seq / 1000, proba, delta=1e-1)

    @parameterized.named_parameters(('noise_1', 0.10),
                                    ('noise_5', 0.5))
    def test_add_bias(self, std):
        tf_generator = tf.random.Generator.from_seed(1234)
        batch = tf.ones((1, 48, 41))
        noised_seq = add_normal_bias(batch, tf_generator, std)
        noise = noised_seq - batch
        self.assertAllEqual(noise[0], noise[-1])
        batch_zeros = tf.zeros((50, 48, 41))
        noised_seq = add_normal_bias(batch_zeros, tf_generator, std)
        self.assertAllEqual(noised_seq, batch_zeros)

    @parameterized.named_parameters(('100', 100),
                                    ('200', 200))
    def test_history_cutout(self, size):
        tf_generator = tf.random.Generator.from_seed(1234)
        batch = tf.ones((1, 1000, 10))
        cutout_seq = history_cutout(batch, tf_generator, size)
        self.assertEqual(tf.reduce_sum(cutout_seq), (1000 - size) * 10)

    @parameterized.named_parameters(('compose_spatial_temp_drop', [spatial_dropout, history_cutout]))
    def test_composition(self, transformations):
        batch = tf.ones((2, 48, 41))
        tf_generator = tf.random.Generator.from_seed(1234)

        composed = compose_transformation(batch, tf_generator, transformations)
        self.assertEqual(batch.shape, composed.shape)


if __name__ == '__main__':
    tf.test.main()

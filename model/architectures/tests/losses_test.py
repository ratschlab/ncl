import tensorflow as tf
from absl.testing import parameterized

from model.architectures.losses import Momentum_Neighbors_NT_X, Momentum_NT_X, get_neighbors_mask_temporal, \
    get_neighbors_dt_label_multiclass


class LossTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.named_parameters(("base",))
    def test_NCL_loss(self, ):
        samples = tf.eye(100)  # all samples are orthogonal one another
        neigh_queue = tf.random.uniform((500, 2), 0, 1000.0)
        queue = tf.concat([samples, tf.zeros((400, 100))], axis=0)
        outputs_regular_CL = Momentum_NT_X([samples, queue], 0.05)
        outputs_CL_as_NCL = Momentum_Neighbors_NT_X([samples, queue], neigh_queue, 0.05, 1.0,
                                                    get_neighbors_mask_temporal)
        loss_CL = outputs_regular_CL[0]
        loss_CL_as_NCL = outputs_CL_as_NCL[0]

        self.assertAlmostEqual(loss_CL.numpy(), 0.0)
        self.assertAlmostEqual(loss_CL_as_NCL.numpy(), 0.0)
        self.assertEqual(loss_CL_as_NCL.numpy(), loss_CL.numpy())

    @parameterized.named_parameters(('base', tf.stack([tf.range(100) + 1, -tf.range(100) - 1], axis=1),
                                     tf.stack([-tf.range(400) - 1, tf.range(400) + 1], axis=1), 0))
    def test_n_w(self, samples, queue, threshold):
        neigh_mat_diag = get_neighbors_mask_temporal(samples, tf.concat([samples, queue], axis=0), threshold)
        neigh_mat_double = get_neighbors_mask_temporal(samples, tf.concat([samples, samples], axis=0), threshold)
        self.assertAllEqual(neigh_mat_diag[:100, :100], tf.eye(100))
        self.assertAllEqual(tf.reduce_sum(neigh_mat_double, axis=1), 2 * tf.ones((100,)))

    @parameterized.named_parameters(
        ('base', tf.random.uniform((100, 1), 1, 1000.0), - tf.random.uniform((400, 1), 1, 1000.0)))
    def test_n_Y(self, samples, queue):
        neigh_mat_diag = get_neighbors_dt_label_multiclass(samples, tf.concat([samples, queue], axis=0))
        neigh_mat_double = get_neighbors_dt_label_multiclass(samples, tf.concat([samples, samples], axis=0))
        self.assertAllEqual(neigh_mat_diag[:100, :100], tf.eye(100))
        self.assertAllEqual(tf.reduce_sum(neigh_mat_double, axis=1), 2 * tf.ones((100,)))


if __name__ == '__main__':
    tf.test.main()

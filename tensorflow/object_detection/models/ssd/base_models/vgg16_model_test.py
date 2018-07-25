import unittest
import tensorflow as tf
from vgg16_model import Model


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        inputs = tf.zeros(shape=[1, 300, 300, 3], dtype=tf.float32)
        m = Model(21, 64)
        _ = m(inputs, True)

        conv5_3 = m.layers['conv5_3']
        shape = conv5_3.get_shape().as_list()
        self.assertEqual(shape[1], 19)


if __name__ == '__main__':
    unittest.main()
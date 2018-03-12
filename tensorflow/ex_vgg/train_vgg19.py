# -*- coding: utf-8 -*-
"""
Train and run VGG19.
"""
import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import os
import random

print('List files')
filenames = [n for n in os.listdir('./train_data') if n.endswith('.png')]

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        true_out = tf.placeholder(tf.float32, [None, 2])
        train_mode = tf.placeholder(tf.bool)

        vgg = vgg19.Vgg19()
        vgg.build(images, train_mode)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print(vgg.get_var_count())

        sess.run(tf.global_variables_initializer())

        # simple 1-step training
        print('Train')
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
        for _ in range(1000):
            batch_filenames = random.choices(filenames, k = 10)
            targets = [n.split('_')[3] == '0.png' and [1, 0] or [0, 1] for n in batch_filenames]
            batch = [utils.load_image('./train_data/' + n) for n in batch_filenames]
            sess.run(train, feed_dict={images: batch, true_out: targets, train_mode: True})

            prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
            utils.print_prob(prob, batch_filenames)

        # test classification again, should have a higher probability about tiger
        print('Test')
        test_filenames = random.choices(filenames, k = 10)
        targets = [n.split('_')[3] == '1.png' and [1, 0] or [0, 1] for n in test_filenames]
        batch = [utils.load_image('./train_data/' + n) for n in test_filenames]
        prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
        utils.print_prob(prob, test_filenames)

        # save trained net.
        vgg.save_npy(sess, './stock.npy')


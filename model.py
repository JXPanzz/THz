## Some codes from https://github.com/yenchenlin/pix2pix-tensorflow


from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
from ops import *
from utils import *
from scipy.io import *
class pix2pix(object):
    def __init__(self, sess, image_size_H=256,image_size_W=256,
                 batch_size=1, sample_size=1, output_size_H=256,output_size_W=256,
                 gf_dim=64, df_dim=64, L1_lambda=200,
                 input_c_dim=3, output_c_dim=3, dataset_name='segmentation',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size_H = image_size_H
        self.image_size_W = image_size_W
        self.sample_size = sample_size
        self.output_size_H = output_size_H
        self.output_size_W = output_size_W

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.c_bn1 = batch_norm(name='c_bn1')
        self.c_bn2 = batch_norm(name='c_bn2')
        self.c_bn3 = batch_norm(name='c_bn3')
        self.c_bn4 = batch_norm(name='c_bn4')

        self.g_c_bn_e2 = batch_norm(name='g_c_bn_e2')
        self.g_c_bn_e3 = batch_norm(name='g_c_bn_e3')
        self.g_c_bn_e4 = batch_norm(name='g_c_bn_e4')
        self.g_c_bn_e5 = batch_norm(name='g_c_bn_e5')
        self.g_c_bn_e6 = batch_norm(name='g_c_bn_e6')
        self.g_c_bn_e7 = batch_norm(name='g_c_bn_e7')
        self.g_c_bn_e8 = batch_norm(name='g_c_bn_e8')

        self.g_bn_d9 = batch_norm(name='g_bn_d9')
        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')


        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()


    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size_H, self.image_size_W,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.real_label = tf.placeholder(tf.int32, [self.batch_size, 4], name='real_label')
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B, self.C_logits = self.generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

        self.real_fake_B= tf.concat([self.real_B, self.fake_B], 3)

        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample, self.C_logits_ = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)


        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.c_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.real_label, logits=self.C_logits))


        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                     + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) \
                     + 50.0 * tf.reduce_mean(tf.abs(self.fake_B + 1.0)) #\
                     #+ 15.0 * self.c_loss



        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.c_vars = [var for var in t_vars if 'g_c_' in var.name]
        self.saver = tf.train.Saver()


    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file) for sample_file in data]



        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()

        label = np.zeros([self.batch_size, 4])

        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images, self.real_label: label}
        )

        tmp = np.concatenate((sample_images[:,:,:,:3],samples),axis=2)
        tmp = np.concatenate((tmp,sample_images[:,:,:,3:6]),axis=2)

        save_images(tmp, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr / 1.9, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr * 2.2, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        c_optim = tf.train.AdamOptimizer(args.lr * 50.0, beta1=args.beta1) \
            .minimize(self.c_loss, var_list=self.c_vars)
        # g_optim = tf.train.GradientDescentOptimizer(args.lr * 2.0) \
        #     .minimize(self.g_loss, var_list=self.g_vars)



        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum, self.c_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.c_sum = tf.summary.merge([self.c_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        data_in = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
        #
        batch_idxs = min(len(data_in), args.train_size) // self.batch_size
        labels = loadmat('./datasets/{}/label/train.mat'.format(self.dataset_name))
        labels = labels['name']
        ### load training labels
        for epoch in xrange(args.epoch):
            np.random.shuffle(data_in)
            for idx in xrange(0, batch_idxs):
                batch_files = data_in[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]

                # label_files = labels[idx*self.batch_size:(idx+1)*self.batch_size]
                label = np.zeros([self.batch_size, 4])
                for ii in xrange(0, self.batch_size):
                    tmp = batch_files[ii][-7:]
                    a = np.min(np.where(labels >= tmp))
                    index = int(labels[a][-1])
                    label[ii, index] = 1
                    ### one-hot code as label for each sample
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)
                #Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.real_data: batch_images, self.real_label: label})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images, self.real_label: label})
                self.writer.add_summary(summary_str, counter)

                # Run c_optim
                _, summary_str = self.sess.run([c_optim, self.c_sum],
                                               feed_dict={self.real_data: batch_images, self.real_label: label})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images, self.real_label: label})
                errC = self.c_loss.eval({self.real_data: batch_images, self.real_label: label})
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, real: %.8f, fake: %.8f c_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG, errD_real, errD_fake, errC))

                #gloss.append(errG)
                #dloss.append(errD_real + errD_fake)

                if np.mod(counter, 120) == 0:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 6000) == 0:
                    self.save(args.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4
    def classifier(self, image, y=None, reuse=False):

        with tf.variable_scope("classifier") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='c_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.c_bn1(conv2d(h0, self.df_dim*2, name='c_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.c_bn2(conv2d(h1, self.df_dim*4, name='c_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.c_bn3(conv2d(h2, self.df_dim*8, name='c_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 4, 'd_h4_lin')
            return tf.nn.softmax(logits=h4), h4
    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size_H
            s2, s4, s8, s16, s32, s64, s128, s256 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128), int(s/256)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_c_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_c_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_c_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_c_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_c_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_c_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_c_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_c_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_c_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_c_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_c_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_c_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_c_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_c_bn_e8(conv2d(lrelu(e7), self.gf_dim*16, name='g_c_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)


            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            # d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            # d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            # d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)
            c_out = linear(tf.reshape(e8, [self.batch_size, -1]), 4, 'g_c_lin') ## classification
            return tf.nn.tanh(self.d8), c_out

    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size_H
            s2, s4, s8, s16, s32, s64, s128, s256 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64), int(s / 128), int(s / 256)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_c_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_c_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_c_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_c_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_c_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_c_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_c_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_c_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_c_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_c_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_c_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_c_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_c_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_c_bn_e8(conv2d(lrelu(e7), self.gf_dim*16, name='g_c_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 1.0)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 1.0)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 1.0)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            # d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            # d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            # d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)
            c_out = linear(tf.reshape(e8, [self.batch_size, -1]), 4, 'g_c_lin')
            return tf.nn.tanh(self.d8), c_out

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size_H)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size_H)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/test/*.jpg'.format(self.dataset_name))

        # sort testing input
        #n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        #sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        time_start = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        Weights = self.sess.run(fetches=self.g_vars)
        Weights = np.array(Weights)
        # for i in xrange(60):
        #     txtname = ['W' + str(i) + '.mat']
        #     savemat(txtname[0], {"foo": Weights[i]})

        for i, sample_image in enumerate(sample_images):
            idx = i+1

            # sample = sample_image[:, :, :, :3]
            # sample = tf.convert_to_tensor(sample)
            # y = self.sampler(sample)

            print("sampling image ", idx)

            samples, score = self.sess.run(
                [self.fake_B_sample, self.C_logits_],
                feed_dict={self.real_data: sample_image}
            )
            txtname = ['./score/' + 'S' + str(i) + '.mat']
            savemat(txtname[0], {"foo": score})
            tmp = np.concatenate((sample_image[:, :, :, :3], samples), axis=2)
            tmp = np.concatenate((tmp, sample_image[:, :, :, 3:6]), axis=2)
            # tmp = sample_image[:, :, :, 3:6]
            save_images(tmp, [self.batch_size, 1],
                        './{}/test_{:04d}.png'.format(args.test_dir, idx))
        time_end = time.time()
        print('totally cost', time_end - time_start)


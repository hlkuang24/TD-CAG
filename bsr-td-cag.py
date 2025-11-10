# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.stats as st
import tf_slim

from imageio import imread, imsave
import tensorflow_addons as tfa

import tensorflow._api.v2.compat.v1 as tf
from skimage import img_as_ubyte


tf.disable_v2_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

slim = tf_slim

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_float('portion', 0.2, 'protion for the mixed image')

tf.flags.DEFINE_integer('num_block',2,'number of blocks to shuffle')

tf.flags.DEFINE_integer('num_copies',20,'the number of shuffled copies in each iteration')

tf.flags.DEFINE_float('max_angles',0.2,'the maximum rotating degree (Radians)')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data/val_rs',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './outputs_bsr-td-cag',
                       'Output directory with images.')



FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            # imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')
            imsave(f, img_as_ubyte((images[i, :, :, :] + 1.0) * 0.5), format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def fun_v3(x):
    num_classes = 1001
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v3, end_points_v3

def fun_v4(x):
    num_classes = 1001
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
           x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v4, end_points_v4

def fun_resv2(x):
    num_classes = 1001
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_res_v2, end_points_res_v2

def fun_res101(x):
    num_classes = 1001
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
            x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_resnet, end_points_resnet



def admix(x):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    return tf.concat([(x + FLAGS.portion * tf.gather(x, tf.random.shuffle(indices))) for _ in range(FLAGS.size)], axis=0)

def get_length(length):
    num_block = FLAGS.num_block  #2
    length = int(length) #299
    rand = np.random.uniform(size=num_block)  #生成一个具有均匀分布的随机样本
    rand_norm = np.round(rand*length/rand.sum()).astype(np.int32) #得到0-299间的两个值，比如 [134,250]
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)  #还是得到0-299间的两个值，比如 (134,165)


def shuffle(x):
    num_block = FLAGS.num_block
    _,w,h,_ = x.shape
    width_length, height_length = get_length(w), get_length(h)
    width_perm, height_perm = np.random.permutation(np.arange(num_block)), np.random.permutation(np.arange(num_block))
    x_split_w = tf.split(x,width_length, axis=1)
    x_w_perm = tf.concat([x_split_w[i] for i in width_perm], axis=1)
    x_split_h = tf.split(x_w_perm,height_length, axis=2)
    x_h_perm = tf.concat([x_split_h[i] for i in height_perm], axis=2)
    return x_h_perm


def shuffle_rotate(x):
    num_block = FLAGS.num_block
    _,w,h,_ = x.shape
    width_length, height_length = get_length(w), get_length(h)
    width_perm, height_perm = np.random.permutation(np.arange(num_block)), np.random.permutation(np.arange(num_block))
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    x_split_w = tf.split(x, width_length, axis=1)
    x_w_perm = tf.concat([x_split_w[i] for i in width_perm], axis=1)
    x_spilt_h_l = [tf.split(x_split_w[i], height_length, axis=2) for i in width_perm]
    x_h_perm = tf.concat([tf.concat(
                            [tfa.image.rotate(strip[i], tf.truncated_normal([tf.shape(x)[0]], stddev=FLAGS.max_angles), interpolation='BILINEAR')
                            for i in height_perm], axis=2 )for strip in x_spilt_h_l], axis=1)
    return x_h_perm

def admix(x):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    return tf.concat([(x + FLAGS.portion * tf.gather(x, tf.random.shuffle(indices))) for _ in range(FLAGS.size)], axis=0)

def BSR(x):
    return tf.concat([shuffle_rotate(x) for i in range(FLAGS.num_copies)], axis=0)



def TD(pth, input_tensor):
    input_tensor = tf.ensure_shape(input_tensor, [None, 299, 299, 3])

    padded = tf.pad(
        input_tensor,
        paddings=[[0, 0], [pth, pth], [pth, pth], [0, 0]],
        mode="CONSTANT",
        name = None,
        constant_values = 0
    )

    # 生成随机偏移量（确保不越界）
    max_offset = 2 * pth
    h_start = tf.random.uniform([], 0, max_offset + 1, dtype=tf.int32)
    w_start = tf.random.uniform([], 0, max_offset + 1, dtype=tf.int32)

    cropped = tf.image.crop_to_bounding_box(
        padded,
        offset_height=h_start,
        offset_width=w_start,
        target_height=299,
        target_width=299
    )
    cropped.set_shape([None, 299, 299, 3])
    return cropped



def graph(x, y, i, x_max, x_min, xb, gb, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001

    logits, end_points = fun_v3(x)
    one_hot = tf.one_hot(y, num_classes)
    cross_entropy_x = tf.losses.softmax_cross_entropy(one_hot, logits)
    g_x = tf.gradients(cross_entropy_x, x)[0]

    x_td = TD(10, x)
    x_batch = BSR(x_td)
    logits, end_points = fun_v3(x_batch)
    one_hot_batch = tf.concat([tf.one_hot(y, num_classes)] * FLAGS.num_copies, axis=0)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_batch, logits)
    g_x_bsr = tf.gradients(cross_entropy, x)[0]

    def compute_g_xb():
        x_b = x - alpha * tf.sign(g_x)
        logits_b, end_points_b = fun_v3(x_b)
        cross_entropy_xb = tf.losses.softmax_cross_entropy(one_hot, logits_b)
        g_xb = tf.gradients(cross_entropy_xb, x_b)[0]

        return g_xb, x_b

    def reuse_gb():
        return gb, xb

    g_xb, x_b = tf.cond(tf.equal(i, 0), compute_g_xb, reuse_gb)

    gg = tf.sign((g_x - g_xb) / (alpha * tf.sign(grad) + 1e-8))

    delta = 0.6
    x_ta = x + g_x * delta + 0.5 * gg * delta ** 2

    x_ta_td = TD(10, x_ta)
    x_ta_bsr = BSR(x_ta_td)
    logits, end_points = fun_v3(x_ta_bsr)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_batch, logits)
    g_xta_bsr = tf.gradients(cross_entropy, x_ta)[0]


    noise = g_xta_bsr + g_x_bsr
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)

    noise = momentum * grad + noise

    xb = x
    x = x +  alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, xb, g_x, noise

def stop(x, y, i, x_max, x_min, xb, gb, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


# def image_augmentation(x):
#     # img, noise
#     one = tf.fill([tf.shape(x)[0], 1], 1.)
#     zero = tf.fill([tf.shape(x)[0], 1], 0.)
#     transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
#     rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
#     return images_transform(x, transforms + rands, interpolation='BILINEAR')
#
def image_augmentation(x):
    # img, noise
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)

    affine_transform = tf.concat([transforms, rands], axis=1)
    affine_transform = tf.reshape(affine_transform, [-1, 3, 3])

    return tf.image.transform(x, affine_transform, interpolation='BILINEAR')

# def image_rotation(x):
#     """ imgs, scale, scale is in radians """
#     rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
#     return images_rotate(x, rands, interpolation='BILINEAR')
#
def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)

    def rotation_matrix(angle):
        angle_deg = tf.math.degrees(angle)
        rot_mat = tf.stack([
            [tf.cos(angle), -tf.sin(angle), tf.zeros_like(angle)],
            [tf.sin(angle), tf.cos(angle), tf.zeros_like(angle)],
            [tf.zeros_like(angle), tf.zeros_like(angle), tf.ones_like(angle)]
        ], axis=-1)
        return rot_mat

    rotation_matrices = tf.vectorized_map(rotation_matrix, rands)

    rotation_matrices = tf.reshape(rotation_matrices, [-1, 3, 3])

    return tf.image.transform(x, rotation_matrices, interpolation='BILINEAR')


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
               x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

        pred = tf.argmax(end_points_v3['Predictions'], 1)

        xb = tf.constant(np.zeros(batch_shape), tf.float32)
        gb = tf.constant(np.zeros(batch_shape), tf.float32)
        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)

        x_adv, _, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, pred, i, x_max, x_min, xb, gb, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])

            idx = 0
            l2_diff = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))

                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))

    print(FLAGS.output_dir)


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()

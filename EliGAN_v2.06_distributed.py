import tensorflow as tf
import numpy as np
try:
    from tensorflow.keras.datasets import cifar10
except:
    from keras.datasets import cifar10
import random
import math
import sys
import pickle
from tensorflow import layers
from tensorflow.nn import relu
from PIL import Image
import datetime
from cifar10_inception_weights.inceptionv3 import get_activations
import tensorflow.contrib
import tensorflow.contrib.gan as gan
import time

fid_from_acts = gan.eval.frechet_classifier_distance_from_activations


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train * 1.0/ 128) - 1 #checkthis


is_self_modulating = False
will_load = True
will_train = False
selected_devices = ['/device:GPU:0', '/device:GPU:1']
checkpoint_to_load = "./models/ckpt_WGANGP_256_isNOTselfmodulating_iter49000_datetimeinfo_4_23_3_12_7.ckpt"
starting_iter_val = 25000
n_final_iter = 50000
n_filters = 256
initial_lr = 2e-4     # 2e-4
batch_size = 256
n_dis = 5
compute_IS_at_end = False

netid = str(batch_size)
if(is_self_modulating):
    netid += "_selfmodulating_"
else:
    netid += "_isNOTselfmodulating"



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.amax(x, axis=1),1))
    return e_x / np.expand_dims(e_x.sum(axis=1), 1)


def average_gradients(tower_grads):

    avg_grads = []
    for grad_var_bunch in zip(*tower_grads):
        grads = []
        for g, _ in grad_var_bunch:
            grads.append(tf.expand_dims(g, 0))
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_var_bunch[0][1]
        average_grads.append((grad, v))
    return avg_grads

def upsample_conv2d(x, n_filters=256, kernel_size=(3, 3)):
    _x = tf.concat([x, x, x, x], axis=-1)
    _x = tf.depth_to_space(_x, block_size=2)
    _x = layers.conv2d(_x, n_filters, kernel_size=kernel_size, padding='same')
    return _x

def downsample_conv2d(x, scope, n_filters=256, kernel_size=(3, 3)):
    with tf.variable_scope(scope):
        _x = tf.space_to_depth(x, block_size=2)
        _x = tf.add_n(tf.split(_x, 4, axis=-1))/4
        _x = layers.conv2d(_x, n_filters, kernel_size=kernel_size, padding='same', name='downsample_conv2d')
        return _x

def resBlockUp(x, latent, n_filters=256, kernel_size=(3, 3), scope='default_scope'):
    with tf.variable_scope(scope):
        scale = None
        offset = None

        if(is_self_modulating):
            #latent_reshaped = tf.reshape(latent, [1, -1])
            print("latent shape:", np.shape(latent))
            scale = layers.dense(latent, 64, activation='relu', name='a')
            scale = layers.dense(scale, 1, name='b')[0]
            offset = layers.dense(latent, 64, activation='relu', name='c')
            offset = layers.dense(offset, 1, name='d')[0]
            print("scaleshape", np.shape(scale))
        mean, var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        _x = tf.nn.batch_normalization(x, mean, var, offset=offset, scale=scale, variance_epsilon=1e-3)
        _x = tf.nn.relu(_x)

        _x = layers.conv2d(_x, n_filters, kernel_size = kernel_size, padding='same', name='resBlockUp_conv2d')

        if(is_self_modulating):
            #latent_reshaped = tf.reshape(latent, [1, -1])
            scale = layers.dense(latent, 64, activation='relu', name='e')
            scale = layers.dense(scale, 1, name='f')[0]
            offset = layers.dense(latent, 64, activation='relu', name='g')
            offset = layers.dense(offset, 1, name='h')[0]
        mean, var = tf.nn.moments(_x, [0, 1, 2], keep_dims=True)
        _x = tf.nn.batch_normalization(_x, mean, var, offset=offset, scale=scale, variance_epsilon=1e-3)
        _x = tf.nn.relu(_x)

        residual = upsample_conv2d(_x, kernel_size=kernel_size, n_filters=n_filters)
        shortcut = upsample_conv2d(x, kernel_size=(1, 1), n_filters=n_filters)
        return residual + shortcut

def resBlockDown(x, n_filters=256, kernel_size=(3, 3), scope='default_arg'):
    with tf.variable_scope(scope):
        _x = tf.nn.relu(x)
        _x = layers.conv2d(_x, n_filters, kernel_size = kernel_size, padding='same', activation = 'relu', name='ResBlockDown_conv2d')
        residual = downsample_conv2d(_x, scope+"downsample_conv2d_residual", n_filters=n_filters, kernel_size=kernel_size)
        residual = tf.nn.relu(residual)
        shortcut = downsample_conv2d(x, scope+"downsample_conv2d_shortcut", n_filters=n_filters, kernel_size=(1, 1))
        return residual + shortcut

def resBlock(x, scope, kernel_size=(3, 3), n_filters=128):
    with tf.variable_scope(scope):
        _x = tf.nn.relu(x)
        _x = layers.conv2d(_x, n_filters, kernel_size=kernel_size, padding='same', activation='relu', name='resblock_conv2d_0')
        residual = tf.nn.relu(_x)
        shortcut = x
        return residual + shortcut

def generator(n_samples=1024, scope='default_scope'):
    with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
        z = tf.random_normal([n_samples, 128])
        out = layers.dense(z, 4 * 4 * 256)
        out = tf.reshape(out, [-1, 4, 4, 256])
        for block in range(3):
            out = resBlockUp(out, z, n_filters, kernel_size=(3, 3), scope='res_up_block'+str(block))
        mean, var = tf.nn.moments(out, [0, 1, 2], keep_dims=True)
        out = tf.nn.batch_normalization(out, mean, var, offset=None, scale=None, variance_epsilon=1e-3)
        out = tf.nn.relu(out)
        out = layers.conv2d(out, 3, kernel_size=(3, 3), padding='same')
        out = tf.nn.tanh(out)
        return out

def discriminator(x, scope='default_scope'):
    with tf.variable_scope(scope + '_discriminator', reuse=tf.AUTO_REUSE):
        x = resBlockDown(x, n_filters=128, scope='res_down_block_0')
        x = resBlockDown(x, n_filters=128, scope='res_down_block_1')
        x = resBlock(x, scope='res_block_0')
        x = resBlock(x, scope='res_block_1')
        x = tf.nn.relu(x)
        x = tf.reduce_sum(x, [1, 2])
        x = layers.dense(x, 1, name='discriminator_dense')
        return x[:,0]

iteration = tf.placeholder(tf.float32, [])
reals = [] #List of placeholders

lr = tf.placeholder(tf.float32, [])
opt1 =  tf.train.AdamOptimizer(lr, 0, 0.9)
opt2 =  tf.train.AdamOptimizer(lr, 0, 0.9)

towers_grads_d = []
towers_grads_g = []

for i in range(len(selected_devices)):
    with tf.device(selected_devices[i]):
        scope = "active_device_"+str(i)
        with tf.variable_scope(scope):
            fake = generator(batch_size, scope)
            real = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
            reals.append(real)
            epsilons = tf.random.uniform(shape = [batch_size, 1, 1, 1])
            x_hat = epsilons * real + (1 - epsilons) * fake
            d_real = discriminator(real, scope)
            d_fake = discriminator(fake, scope)
            d_x_hat = discriminator(x_hat, scope)

            gradients = tf.gradients(d_x_hat, x_hat)
            gradient_norms = tf.sqrt(tf.reduce_sum(tf.multiply(gradients, gradients), [1, 2, 3]))
            gradient_penalty = 10 * tf.reduce_mean((gradient_norms - 1)**2)

            loss_d = tf.reduce_mean(d_fake) - 1 * tf.reduce_mean(d_real) + gradient_penalty
            loss_g = -tf.reduce_mean(d_fake)

            vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='active_device_' + str(i) + '/active_device_' + str(i) + '_discriminator')
            vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='active_device_' + str(i) + '/active_device_' + str(i) + '_generator')

            grads_d =  opt1.compute_gradients(loss_d, var_list = vars_d)
            towers_grads_d.append(grads_d)
            grads_g =  opt2.compute_gradients(loss_g, var_list = vars_g)
            towers_grads_g.append(grads_g)

reduced_grads_d = average_gradients(towers_grads_d)
reduced_grads_g = average_gradients(towers_grads_g)

opt_fn1 = opt1.apply_gradients(reduced_grads_d)
opt_fn2 = opt2.apply_gradients(reduced_grads_g)

init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth=True

saver = tf.train.Saver()
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)


with tf.Session(config=config) as sess:

    sess.run(init)

    # function to generate, render, and save a composite image of generator samples.
    def generate_composite(iter_num):
        gen_images = sess.run([fake])[0][0:100,:,:,:]
        gen_images = np.uint8(np.clip((gen_images + 1) * 128, 0, 255))
        image_rows = np.array([np.concatenate(gen_images[10*a:10*a+10,:,:,:], axis=1) for a in range(10)])
        composite =  np.concatenate(image_rows, axis=0)
        img = Image.fromarray(composite, 'RGB')
        #img.show()
        img.save('./images/' + netid + "_GAN_samples_at_iteration" + str(iter_num) + '.png')

    # Code to load model parameters if starting with model parameters saved in a checkpoint file.
    if(will_load):
        saver.restore(sess, checkpoint_to_load)
        print("loaded model")

    if(will_train):
        # This is the main training loop. Prints and saves a composite every 50 steps. Saves network parameters every 1,000 steps.
        last_time = time.time()
        for a in range(starting_iter_val, n_final_iter+1):
            print('a:', a, 'took time:', time.time()-last_time)
            last_time = time.time()

            # Does 5 optimzer steps on the discriminator network, then 1 on the generator network.


            for i in range(n_dis):
                np.random.shuffle(x_train)
                loss_val, real_disc_vals, fake_disc_vals, _, grad_pen = sess.run([loss_d, d_real, d_fake, opt_fn1, gradient_penalty],
                    feed_dict={lr: (initial_lr * (n_final_iter-a)/n_final_iter), iteration: a, reals[0]: x_train[0:batch_size]}, options=run_options)
            sess.run([opt_fn2], feed_dict={lr: (initial_lr * (n_final_iter-a)/n_final_iter), iteration: a}, options=run_options)

            # Every 50 iterations, render and save a composite image of generator samples.
            if(a%10==0):
                generate_composite(a)

            # Every 1000 steps, save model parameters in a checkpoint.
            if(a%1000==0):
                dt = datetime.datetime.now()
                now = str(dt.month) + "_" + str(dt.day) + "_" + str(dt.hour) + "_" + str(dt.minute) + "_" + str(dt.second)
                save_path = saver.save(sess, "./models/ckpt_WGANGP_" + netid + "_iter" + str(a) + "_datetimeinfo_" + now + ".ckpt")
                print("Model saved in path: %s" % save_path)

    my_num = 100
    num_gen_batches = 10

    generator_samples = sess.run([fake])[0]
    last_time = time.time()
    for i in range(1, num_gen_batches):
        generator_samples = np.concatenate([generator_samples, sess.run([fake])[0]], axis=0)
        print("time of gen_batch:", time.time() - last_time)
        last_time = time.time()

    gen_images = np.uint8(np.clip((generator_samples[0:800:8] + 1) * 128, 0, 255))
    image_rows = np.array([np.concatenate(gen_images[10*a:10*a+10,:,:,:], axis=1) for a in range(10)])
    composite =  np.concatenate(image_rows, axis=0)
    img = Image.fromarray(composite, 'RGB')
    img.show()
    img.save('./images/' + netid + "_GAN_samples_composite"+ '.png')

    def get_inception_score(samples, n_splits=10):
        print("print point 1")
        split_n = int(np.shape(samples)[0]/n_splits)
        print("split n", split_n)
        scores = []
        logit_values = get_activations(samples, sess, batch_size=100)
        inception_p_ests = softmax(logit_values)
        print("print point 2")
        for i in range(10):
            p_batch = inception_p_ests[i*split_n:(i+1)*split_n, :]
            print("inception p ests", p_batch[0,:])
            avg_label_vals = np.expand_dims(np.average(p_batch, axis=0), axis=0)
            print("avg_label_vals", avg_label_vals[0])
            inception_score = math.exp(np.average(np.sum(p_batch * (np.log(p_batch) - np.log(avg_label_vals)), axis=1)))
            print("inception score:", inception_score)
            scores.append(inception_score)
        print("print point 3")
        print("scores:", scores)
        return np.average(scores), np.std(scores)
    print(np.shape(generator_samples))
    print("print point 0")
    #generate_composite(0)
    if(compute_IS_at_end):
        print("Inception Score:", get_inception_score(generator_samples))

    # Commented out code from initial attempt to incorporate Frechet Inception Score.
    """
    #pool_logits_r = get_activations(x_test, sess, num_images=my_num, batch_size=100)
    #pool_logits_f = get_activations(fake_images, sess, num_images=my_num, batch_size=100)

    def FID(acts_real, acts_fake, splits=10):

        fid_batch_vals = []
        print("shape of activation batch fed to FID:", np.shape(acts_real))

        for a in range(splits):
            n = acts_fake.shape[0]//splits
            b_acts_real = acts_real[n*a:n*(a+1), :]
            b_acts_fake = acts_fake[n*a:n*(a+1), :]
            val = fid_from_acts(b_acts_real, b_acts_fake)
            fid_batch_vals.append(val)
        return tf.reduce_mean(fid_batch_vals), tf.std(fid_batch_vals)

    #acts_fake = tf.squeeze(get_activations(fake_images, sess, layer="PreLogits"))
    #acts_real = tf.squeeze(get_activations(x_train, sess, layer="PreLogits"))
    #frechet_inception_score = sess.run([FID(acts_real, acts_fake)])
    """
# Compute  and print Inception Score

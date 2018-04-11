import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
from glob import glob

def prepare_data(dataset_name, size, gray_to_RGB=False):
    input_list = sorted(glob('./dataset/{}/*.*'.format(dataset_name + '/trainA')))
    target_list = sorted(glob('./dataset/{}/*.*'.format(dataset_name + '/trainB')))

    trainA = []
    trainB = []

    if gray_to_RGB :
        for image in input_list:
            trainA.append(np.expand_dims(misc.imresize(misc.imread(image, mode='L'), [size, size]), axis=-1))

        for image in input_list:
            trainB.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))

        # trainA = np.repeat(trainA, repeats=3, axis=-1)
        # trainA = np.array(trainA).astype(np.float32)[:, :, :, None]

    else :
        for image in input_list :
            trainA.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))

        for image in target_list :
            trainB.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))


    trainA = preprocessing(np.asarray(trainA))
    trainB = preprocessing(np.asarray(trainB))

    return trainA, trainB

def shuffle(x, y) :
    seed = np.random.random_integers(low=0, high=1000)
    np.random.seed(seed)
    np.random.shuffle(x)

    np.random.seed(seed)
    np.random.shuffle(y)

    return x, y

def load_test_data(image_path, size=256, gray_to_RGB=False):
    if gray_to_RGB :
        img = misc.imread(image_path, mode='L')
        img = misc.imresize(img, [size, size])
        img = np.expand_dims(img, axis=-1)
    else :
        img = misc.imread(image_path, mode='RGB')
        img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img


def preprocessing(x):

    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

import tensorflow as tf
import random

AUTO = tf.data.AUTOTUNE

@tf.function
def gaussian_blur(image, kernel_size=23, padding='SAME'):
	sigma = tf.random.uniform((1,))* 1.9 + 0.1
	radius = tf.cast(kernel_size / 2, tf.int32)
	kernel_size = radius * 2 + 1
	
	x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
	blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
	blur_filter /= tf.reduce_sum(blur_filter)
	blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
	blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
	num_channels = tf.shape(image)[-1]
	blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
	blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
	expand_batch_dim = image.shape.ndims == 3
	if expand_batch_dim:
		image = tf.expand_dims(image, axis=0)
	blurred = tf.nn.depthwise_conv2d(image, blur_h, strides=[1, 1, 1, 1], padding=padding)
	blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
	if expand_batch_dim:
		blurred = tf.squeeze(blurred, axis=0)
	return blurred

@tf.function
def color_jitter(x, s=0.5):
	x = tf.image.random_brightness(x, max_delta=0.8*s)
	x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
	x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
	x = tf.image.random_hue(x, max_delta=0.2*s)
	x = tf.clip_by_value(x, 0, 1)
	return x

@tf.function
def color_drop(x):
	x = tf.image.rgb_to_grayscale(x)
	x = tf.tile(x, [1, 1, 3])
	return x

@tf.function
def random_apply(func, x, p):
	return tf.cond(
		tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
				tf.cast(p, tf.float32)),
		lambda: func(x),
		lambda: x)

@tf.function
def custom_augment(image):
	image = random_apply(tf.image.flip_left_right, image, p=0.5)
	image = random_apply(gaussian_blur, image, p=0.5)
	image = random_apply(color_jitter, image, p=0.8)
	image = random_apply(color_drop, image, p=0.2)
	return image

@tf.function
def random_resize_crop(image, min_scale, max_scale, crop_size):
	h = tf.shape(image)[0]
	w = tf.shape(image)[1]
	base_dim = tf.cast(tf.minimum(h, w), tf.float32)

	rand_size = tf.random.uniform([], min_scale*base_dim, max_scale*base_dim, dtype=tf.float32)
	rand_size = tf.cast(tf.clip_by_value(rand_size, 2.0, base_dim), tf.int32)

	crop = tf.image.random_crop(image, size=[rand_size, rand_size, 3])
	crop_resize = tf.image.resize(crop, (crop_size, crop_size))
	return crop_resize

@tf.function
def tie_together(image, min_scale, max_scale, crop_size):
	image = random_resize_crop(image, min_scale,max_scale, crop_size)
	image = custom_augment(image)
	return image

def get_multires_dataset(dataset,
	size_crops,
	num_crops,
	min_scale,
	max_scale,
	options=None):
	loaders = tuple()
	base = dataset.with_options(options) if options is not None else dataset
		
	for i, n in enumerate(num_crops):
		for _ in range(n):
			loader = base.map(
					lambda x: tie_together(x, min_scale[i], max_scale[i], size_crops[i]),
					num_parallel_calls=AUTO,
					deterministic=False
				)
			loaders += (loader,)
	return loaders

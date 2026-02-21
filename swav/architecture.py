from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda
import tensorflow as tf

def get_resnet_backbone():
	base_model = tf.keras.applications.ResNet50(include_top=False,
		weights=None, input_shape=(None, None, 3))
	base_model.trainable = True
	inputs = layers.Input((None, None, 3))
	h = base_model(inputs, training=True)
	h = layers.GlobalAveragePooling2D()(h)
	backbone = models.Model(inputs, h)
	return backbone

def get_projection_prototype(dense_1=1024, dense_2=256, prototype_dimension=100):
	inputs = layers.Input((2048, ))
	x = layers.Dense(dense_1)(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)
	x = layers.Dense(dense_2)(x)
	proj = Lambda(lambda z: tf.math.l2_normalize(z, axis=1), name='projection')(x)
	prototype = layers.Dense(prototype_dimension, use_bias=False, name='prototype')(proj)
	return models.Model(inputs=inputs, outputs=[proj, prototype])

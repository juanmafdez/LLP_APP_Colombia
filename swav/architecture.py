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
	projection_1 = layers.Dense(dense_1)(inputs)
	projection_1 = layers.BatchNormalization()(projection_1)
	projection_1 = layers.Activation("relu")(projection_1)

	projection_2 = layers.Dense(dense_2)(projection_1)
	projection_2_normalize = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='projection')(projection_2)

	prototype = layers.Dense(prototype_dimension, use_bias=False, name='prototype')(projection_2_normalize)

	return models.Model(inputs=inputs,
		outputs=[projection_2_normalize, prototype])

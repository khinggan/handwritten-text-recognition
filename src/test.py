# import tensorflow as tf
#
# sess = tf.Session()
# a = tf.get_variable("a", [3, 3, 32, 64], initializer=tf.random_normal_initializer())
# gv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#
# print(gv)

from src.network.model import HTRModel

model = HTRModel(architecture="puigcerver_words",
                 input_size=(128, 32, 1),
                 vocab_size=81,
                 greedy=True)

model.compile(learning_rate=0.001)
model.summary()
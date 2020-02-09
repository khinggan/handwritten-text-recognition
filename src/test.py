import tensorflow as tf

sess = tf.Session()
a = tf.get_variable("a", [3, 3, 32, 64], initializer=tf.random_normal_initializer())
gv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

print(gv)
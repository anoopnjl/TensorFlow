#urls
# https://www.quora.com/Where-can-I-start-learning-how-to-use-TensorFlow
# http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
# https://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow-tutorial/

import tensorflow as tf
#without placeholder
const = tf.constant(2.0)
b = tf.Variable(2.0)
c = tf.Variable(1.0)
d = tf.add(b,c)
e = tf.add(c,const) #3
a = tf.multiply(d,e)
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
sess.run(a)

#with placeholder
const = tf.constant(2.0)
b = tf.placeholder(tf.float32)
c = tf.Variable(1.0)
d = tf.add(b,c)
e = tf.add(c,const) #3
a = tf.multiply(d,e)
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})

#some nueralnet example
mnist = tf.keras.datasets.mnist
mnist_data = mnist.load_data()
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W1 = tf.Variable(tf.random_normal([784,300],stddev=0.03))
b1 = tf.Variable(tf.random_normal([300]))
W2 = tf.Variable(tf.random_normal([300,10],stddev=0.03))
b2 = tf.Variable(tf.random_normal([10]))

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)
hidden_out2=tf.add(tf.matmul(hidden_out,W2),b2)
y_ = tf.nn.softmax(hidden_out2)

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)+ (1 - y) * tf.log(1 - y_clipped), axis=1))

learning_rate=0.05
epochs=100
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
init_op = tf.global_variables_initializer()


#tensorflow graphs

graph = tf.get_default_graph()
graph.get_operations()
for op in graph.get_operations(): 
    print(op.name)
	
#create a random normal distribution
w=tf.Variable(tf.random_normal([3, 2], stddev=0.01))
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    a=sess.run(w)
    print(a)

#reduce mean
b = tf.Variable([10,20,30,40,50,60],name='t')
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.reduce_mean(b)))
	
#ArgMax
a=[ [0.1, 0.2,  0.3  ],
    [20,  2,       3   ]
  ]
 b = tf.Variable(a)
 with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.argmax(b,1)))
	
#linear regression

trainX = np.linspace(-1, 1, 101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33
X = tf.placeholder(float)
Y = tf.placeholder(float)
w = tf.Variable(0.1, name = "weights")
y_model = tf.multiply(X, w)
cost = tf.pow(Y-y_model,2)
train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X:x, Y:y})
    print(sess.run(w))







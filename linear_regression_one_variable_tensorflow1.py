import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


sample_no = 30
test_no = 10
learning_rate = 0.01
epochs = 3000

train_x = np.linspace(0,20,sample_no)
train_y = 3*train_x + 4*np.random.randn(sample_no)

test_x = np.random.randint(low = 0, high = 20,size=(test_no))
test_y = 3*test_x + 4*np.random.randn(test_no)

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name = "weight")
B = tf.Variable(np.random.randn(), name = "bias")

linear_model = W*X + B

cost = tf.reduce_sum(tf.square(linear_model - y)) / (2*sample_no)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(optimizer,feed_dict = {X : train_x, y : train_y})

        if epoch%100 == 0 :
            c = sess.run(cost, feed_dict = {X : train_x, y : train_y})
            print("Epoch : ",epoch," Cost : ",c," W : ",sess.run(W)," B : ",sess.run(B))

    weight = sess.run(W)
    bias = sess.run(B)
    c = sess.run(cost,feed_dict = {X : train_x, y : train_y})
    print("Optimisation Finished!")
    print("Training Cost = ",c," W = ",weight," B = ",bias)
    plt.plot(train_x,train_y,'o')
    plt.plot(train_x, weight*train_x + bias)
    plt.show()

    test_cost = sess.run(tf.reduce_sum(tf.square(linear_model - y)) / (2*test_no), feed_dict = {X : test_x, y : test_y})
    print("Test cost = ",test_cost)
    plt.plot(train_x,train_y,'ro')
    plt.plot(train_x,weight*train_x+bias)
    plt.show()

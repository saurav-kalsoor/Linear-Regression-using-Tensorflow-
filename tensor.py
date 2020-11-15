import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

learning_rate = 0.004
epochs = 2000
sample_no = 30

x_garb = np.linspace(-10,10,sample_no).transpose()

x1 = 0.03*x_garb*x_garb
x2 = -0.5*x_garb
x3 = 0*np.random.random([sample_no]).transpose();

train_x = np.zeros((sample_no,2))
train_x[:,0] = x1
train_x[:,1] = x2
train_x = train_x.astype(np.float32)
train_y = x1 + x2 + x3



X = tf.placeholder(tf.float32 , shape = (None,2))
y = tf.placeholder(tf.float32 , shape = (None,))

print(tf.shape(X)," ",tf.shape(y))

W = tf.cast(tf.Variable(np.random.randn(2,1),name = "weight"),tf.float32)
B = tf.Variable(np.random.randn(), name = "bias")

linear_model = tf.add(tf.matmul(train_x,W),B)

cost = tf.reduce_sum(tf.square(linear_model - y)) /(2*sample_no)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(optimizer, feed_dict = {X : train_x, y : train_y})

        if epoch%50 == 0 :
            c = sess.run(cost, feed_dict = {X : train_x, y : train_y})
            print("Epoch : ",epoch," cost : ",c," w : ",sess.run(W))

    weight = sess.run(W)
    bias = sess.run(B)
    train_cost = sess.run(cost,feed_dict = {X : train_x, y : train_y})
print("Optimisation Completed!")
pred_y = np.add(np.matmul(train_x,weight),bias)
plt.plot(x_garb,train_y,'o')
plt.plot(x_garb,pred_y,'r')
plt.show()


print("Final training cost = ",train_cost)

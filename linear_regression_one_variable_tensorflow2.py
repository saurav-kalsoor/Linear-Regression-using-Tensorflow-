import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


sample_no = 30
learning_rate = 0.1
epochs = 400

x = np.linspace(-10,20,sample_no)
y = 3*x + 4*np.random.randn(sample_no)

my_data = np.zeros((sample_no,2))
my_data[:,0] = x.transpose();
my_data[:,1] = y.transpose();
my_column = ['X','y']
df = pd.DataFrame(data = my_data, columns = my_column)


# plt.scatter(df['X'],df['y'])
# plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[1] ))
model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate))

# model.summary()

history = model.fit(df['X'],df['y'],epochs = epochs)
plt.plot(history.history['loss'])
plt.show()
df['prediction'] = model.predict(df['X'])

plt.scatter(df['X'],df['y'])
plt.plot(df['X'],df['prediction'],'r')
plt.show()

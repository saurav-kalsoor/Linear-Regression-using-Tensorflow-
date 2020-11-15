import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

sample_no = 30
learning_rate = 0.04
epochs = 200

values = np.linspace(-10,20,sample_no)
my_data = np.zeros((sample_no,3))
my_data[:,0] = (3*values*values).transpose()
my_data[:,1] = (-5*values).transpose()

my_data[:,2] = my_data[:,0] + my_data[:,1] + 10*np.random.randn(sample_no).transpose()
my_column = ['X1','X2','y']

df = pd.DataFrame(data = my_data, columns = my_column)

# plt.scatter(values,df['y'])
# plt.show()

features = my_data[:,0:2]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape = [2]))

model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate))

# model.summary()

history = model.fit(features,df['y'],epochs = epochs)

plt.plot(history.history['loss'])
plt.show()

df['prediction'] = model.predict(features)
plt.scatter(values,df['y'])
plt.plot(values,df['prediction'],'r')
plt.show()

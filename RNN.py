import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt

time_step = 7
data_dim = 6
hidden_size = 10
output_dim = 1
learning_rate = 0.01
iterations = 7000

X = tf.placeholder(tf.float32, [None, time_step, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

def readCSV():
    xy = np.loadtxt('stock_weather.csv', delimiter=',')
    #xy = np.loadtxt('samsung.txt', unpack=False, dtype='float32')  # TEXT
    xy = xy[::-1]  # reverse order (chronically ordered)
    xy = nomalization(xy)
    x=xy # 전체 학습
    #x = xy[:,4:]  # 날씨만 가지고 학습
    #x = xy[:, :-1]
    y = xy[:, [-1]]  # Close
    #y = nomalization(y)
    print(x)
    print(y)
    dataX = []
    dataY = []
    for i in range(0, len(y) - time_step):
        _x = x[i:i + time_step]
        _y = y[i + time_step]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    return dataX , dataY

def splitDATA(x , y):
    train_size = int(len(y) * 0.7)
    test_size = len(y) - train_size
    trainX, testX = np.array(x[0:train_size]), np.array( x[train_size:len(x)])
    trainY, testY = np.array(y[0:train_size]), np.array(y[train_size:len(y)])
    return  trainX, testX , trainY, testY

def nomalization(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def plot_results_all(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, 'k', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, 'b--', label='predict')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, 'r--', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def plot_results_predic(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    #plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, 'b--', label='predict')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, 'r--', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def trainANDtest():

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)
    #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size,  activation=tf.tanh)

    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

    # cost/loss
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test
        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})

        print("target : ", testY)
        print("prediction : ", test_predict)
        print("RMSE: {}".format(rmse_val))

        plot_results_all(trainY, test_predict, testY, 'total(stock).png')
        plot_results_predic(trainY, test_predict, testY, 'prediction(stock).png')

read_data_x , read_data_y = readCSV()
trainX, testX , trainY, testY = splitDATA(read_data_x , read_data_y)
trainANDtest()





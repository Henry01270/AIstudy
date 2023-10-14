import tensorflow as tf

train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11, 13, 15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)

def 손실함수(): #손실함수(a, b)
    예측_y = train_x * a + b
    return tf.keras.losses.mse(train_y, 예측_y)
    #실제값, 예측값


opt = tf.keras.optimizers.Adam(learning_rate=0.001)

for i in range(900):
    opt.minimize(손실함수, var_list=[a, b])
    #lambda:손실함수(a, b)
    print(a.numpy(), b.numpy())
import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('gpascore.csv') #데이터프레임 불러오기

data = data.dropna() #데이처 전처리(빈칸 삭제)
#data = data.fillna(100) #전처리 #빈칸이 100으로 바뀜
#print(data.isnull().sum()) #빈칸의 개수를 더함

#print(data['gpa']) #gpa열에 있는 값 불러오기
#.min() #.max() #.count()

y데이터 = data['admit'].values
x데이터 = []

for i, rows in data.iterrows():
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x데이터), np.array(y데이터), epochs=1000)

예측값 = model.predict([ [750, 3.70, 3], [400, 2.2, 1] ])
print(예측값)
#
#model.fit(x데이터, y데이터, epochs=100)
#x데이터: [380, 3.21, 3], [660, 3.67, 3], ...
#y데이터: [0, 1, 0, 1, 1, ...] 
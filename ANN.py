import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold, cross_val_score


data = pd.read_csv('unformatted-data', delimiter=',', header = None)

X = data.iloc[:, 1:10].values
y = data.iloc[:, 10].values



from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = Sequential()


classifier.add(Dense(16, kernel_initializer='uniform', activation='relu', input_dim=9))

classifier.add(Dropout(0.1))



classifier.add(Dense(16, kernel_initializer='uniform', activation='relu'))

classifier.add(Dropout(0.1))


classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#result= classifier.fit(X_train, y_train, batch_size=50, epochs=100, validation_split=0.30)


result= classifier.fit(X_train, y_train, batch_size=100, epochs=100, validation_data=(X_test, y_test))

y_pred = classifier.predict(X_test)
print(y_pred)
print(result.history.keys())

#  "Accuracy"
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

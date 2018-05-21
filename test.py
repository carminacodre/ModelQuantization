from keras.layers import Dense
from keras.models import Sequential

def create_model():

    model = Sequential()

    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

    return model
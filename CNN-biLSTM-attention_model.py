
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional,GRU
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers
from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2

class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weight = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


max_features = 26
embedding_size = 128
kernel_size = 8
filters = 120
pool_size = 4
lstm_output_size = 64
hidden_dims = 400
#hidden_dims2=200

model = Sequential()
model.add(Embedding(max_features, embedding_size))
model.add(Dropout(0.5))
#model.add(Conv1D(filters, kernel_size,padding ='valid',activation = 'relu',strides = 1))
#model.add(MaxPooling1D(pool_size = pool_size))
model.add(Conv1D(filters,kernel_size = 12,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Dropout(0.3))
model.add(Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
#model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(hidden_dims))
#model.add(Dense(hidden_dims2))
#model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(lstm_output_size,return_sequences=True)))
#model.add(LSTM(lstm_output_size,return_sequences=True))
#model.add(GRU(32, return_sequences=True))
#model.add(AttLayer(64))
#model.add(GRU(32))
#model.add(Dropout(0.3))
model.add(AttLayer(128))
#model.add(Bidirectional(LSTM(lstm_output_size)))
#model.add(Dense(64, activation = 'relu',kernel_regularizer = l2(1e-5)))
#model.add(Dense(32))

#model.add(Dense(16, activation = 'relu',kernel_regularizer = l2(1e-5)))
#model.add(Dense(8, activation = 'relu',kernel_regularizer = l2(1e-5)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(0.01),metrics = ['acc'])
#model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(lr=0.0005),metrics = ['acc'])

'''
model = Sequential()
model.add(Embedding(max_features, embedding_size))
model.add(Dropout(0.45))
model.add(Conv1D(filters,kernel_size = 12,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Dropout(0.23))
model.add(Conv1D(filters,kernel_size = kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Dropout(0.23))
model.add(Dense(hidden_dims))
model.add(Bidirectional(LSTM(lstm_output_size,return_sequences=True)))
model.add(AttLayer(128))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(0.05),metrics = ['acc'])
'''

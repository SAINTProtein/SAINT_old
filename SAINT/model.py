from attention_module_tools import *
from keras.models import Model
from keras.layers import RepeatVector, Multiply, Flatten, Dot, Softmax, Lambda, Add, BatchNormalization, Dropout, concatenate
from keras.layers import Input,SpatialDropout1D, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, Masking, Bidirectional
from keras.regularizers import l2
import config

def inceptionBlock(x):
  x = BatchNormalization()(x)
  conv1_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv1_1 = Dropout(config.conv_layer_dropout_rate)(conv1_1)  # https://www.quora.com/Can-l-combine-dropout-and-l2-regularization
  conv1_1 = BatchNormalization()(conv1_1)

  conv2_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv2_1 = Dropout(config.conv_layer_dropout_rate)(conv2_1)
  conv2_1 = BatchNormalization()(conv2_1)
  conv2_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2_1)
  conv2_2 = Dropout(config.conv_layer_dropout_rate)(conv2_2)
  conv2_2 = BatchNormalization()(conv2_2)

  conv3_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv3_1 = Dropout(config.conv_layer_dropout_rate)(conv3_1)
  conv3_1 = BatchNormalization()(conv3_1)
  conv3_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_1)
  conv3_2 = Dropout(config.conv_layer_dropout_rate)(conv3_2)
  conv3_2 = BatchNormalization()(conv3_2)
  conv3_3 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_2)
  conv3_3 = Dropout(config.conv_layer_dropout_rate)(conv3_3)
  conv3_3 = BatchNormalization()(conv3_3)
  conv3_4 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_3)
  conv3_4 = Dropout(config.conv_layer_dropout_rate)(conv3_4)
  conv3_4 = BatchNormalization()(conv3_4)

  concat = concatenate([conv1_1, conv2_2, conv3_4])
  concat = BatchNormalization()(concat)

  return concat

def deep3iBLock_with_attention(x, pos_ids=None):
  block1_1 = inceptionBlock(x)
  block1_1 = attention_module(block1_1, pos_ids)

  block2_1 = inceptionBlock(x)
  block2_2 = inceptionBlock(block2_1)
  block2_2 = attention_module(block2_2, pos_ids)

  block3_1 = inceptionBlock(x)
  block3_2 = inceptionBlock(block3_1)
  block3_3 = inceptionBlock(block3_2)
  block3_4 = inceptionBlock(block3_3)
  block3_4 = attention_module(block3_4, pos_ids)

  concat = concatenate([block1_1, block2_2, block3_4])
  concat = BatchNormalization()(concat)

  return concat


def get_model():
  pssm_input = Input(shape=(700, 21,), name='pssm_input')
  seq_input = Input(shape=(700, 22,), name='seq_input')
  pos_ids = Input(batch_shape=(None, 700), name='position_input', dtype='int32')

  # pos_emb = position_embedding(pos_ids, output_dim=50)
  main_input = concatenate([seq_input, pssm_input])

  block1 = deep3iBLock_with_attention(main_input, pos_ids)
  #   att_layer_4 = attention_scaled_dot(block1)
  #   block1 = MyLayer()([att_layer_4 ,block1])
  #   block1 = BatchNormalization()(block1)

  block2 = deep3iBLock_with_attention(block1, pos_ids)
  block2 = attention_module(block2, pos_ids)

  conv11 = Convolution1D(100, 11, activation='relu', padding='same', kernel_regularizer=l2(0.001))(block2)
  conv11 = attention_module(conv11, pos_ids)

  dense1 = TimeDistributed(Dense(units=256, activation='relu'))(conv11)
  dense1 = Dropout(config.dense_layer_dropout_rate)(dense1)
  dense1 = attention_module(dense1, pos_ids)

  main_output = TimeDistributed(Dense(units=8, activation='softmax', name='main_output'))(dense1)

  model = Model([pssm_input, seq_input, pos_ids], main_output)
  return model


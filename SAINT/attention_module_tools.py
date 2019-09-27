from utility import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import add
from keras.layers import RepeatVector, Multiply, Flatten, Dot, Softmax, Lambda, Add, BatchNormalization, Dropout
from keras import backend
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input,SpatialDropout1D, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, Masking, Bidirectional
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import optimizers, callbacks
from keras.layers import BatchNormalization, Dropout


class WeightedSumLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightedSumLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self._x = K.variable(0.2)
        self._x._trainable = True
        self.trainable_weights = [self._x]

        super(WeightedSumLayer, self).build(input_shape)

    def call(self, x):
        A, B = x
        result = add([self._x*A ,(1-self._x)*B])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def attention(activations): # not used
#https://arxiv.org/pdf/1703.03130.pdf
    d_a = 10
    r = 1
    units = activations.shape[2]
    #print(activations.shape)
    attention = TimeDistributed(Dense(d_a, activation='tanh', use_bias=False))(activations)
    attention = Dropout(.5)(attention)
    #print(attention.shape)
    attention = TimeDistributed(Dense(r, activation='softmax', use_bias=False))(activations)
    #print(attention.shape)
    attention = Flatten()(attention)
    #attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    #print(attention.shape)
    attention = Permute([2, 1])(attention)
    #print(attention.shape)

    # apply the attention
    sent_representation = Multiply()([activations, attention])
    #print(sent_representation.shape)
    return sent_representation


def shape_list(x):
    if backend.backend() != 'theano':
        tmp = backend.int_shape(x)
    else:
        tmp = x.shape
    tmp = list(tmp)
    tmp[0] = -1
    return tmp

def attention_scaled_dot(activations):
#https://arxiv.org/pdf/1706.03762.pdf
    units = int(activations.shape[2])
    words = int(activations.shape[1])
    Q = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    Q = Dropout(.2)(Q)
    K = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    K = Dropout(.2)(K)
    V = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    V = Dropout(.2)(V)
    #print(Q.shape)
    QK_T = Dot(axes=-1, normalize=False)([Q,K]) # list of two tensors
    #print(QK_T.shape)
    QK_T = Lambda( lambda inp: inp[0]/ backend.sqrt(backend.cast(shape_list(inp[1])[-1], backend.floatx())))([QK_T, V])
    #print(QK_T.shape)

    QK_T = Softmax(axis=-1)(QK_T)
    QK_T = Dropout(.2)(QK_T)
    #print(V.shape)
    V = Permute([2, 1])(V)
    #print(V.shape)
    V_prime = Dot(axes=-1, normalize=False)([QK_T,V]) # list of two tensors
    #print(V_prime.shape)
    return V_prime


def _get_pos_encoding_matrix(protein_len: int, d_emb: int) -> np.array:
  pos_enc = np.array(
    [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
     range(protein_len)], dtype=np.float32)
  pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
  pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
  return pos_enc


def embeddings(inputs):
  gene_ids, pos_ids = inputs
  gene_vocab_len = 22
  protein_len = 700
  output_dim = 50
  gene_embedding = Embedding(gene_vocab_len, output_dim, input_length=protein_len,
                             name='GeneEmbedding')(gene_ids)

  pos_embedding = Embedding(protein_len, output_dim, trainable=False, input_length=protein_len,
                            name='PositionEmbedding',
                            weights=[_get_pos_encoding_matrix(protein_len, output_dim)])(pos_ids)

  summation = Add(name='AddEmbeddings')([Dropout(.1, name='EmbeddingDropOut1')(gene_embedding),
                                         Dropout(.1, name='EmbeddingDropOut2')(pos_embedding)])

  #     summation = concatenate([Dropout(.1, name='EmbeddingDropOut1')(gene_embedding),
  #                              Dropout(.1, name='EmbeddingDropOut2')(pos_embedding)])

  summation = LayerNormalization(1e-5)(summation)
  return summation


def gene_embeddings(gene_ids, output_dim=50):
  gene_vocab_len = 22
  protein_len = 700

  gene_emb = Dropout(.1)(Embedding(gene_vocab_len, output_dim, input_length=protein_len,
                                   name='GeneEmbedding')(gene_ids))

  gene_emb = LayerNormalization(1e-5)(gene_emb)
  return gene_emb


def position_embedding(pos_ids, output_dim=50):
  # gene_vocab_len = 22
  protein_len = 700
  output_dim = int(output_dim)

  pos_emb = Dropout(.1)(Embedding(protein_len, output_dim, trainable=False, input_length=protein_len,
                                  # name='PositionEmbedding',
                                  weights=[_get_pos_encoding_matrix(protein_len, output_dim)])(pos_ids))

  pos_emb = LayerNormalization(1e-5)(pos_emb)
  return pos_emb

def attention_module(x, pos_ids=None, drop_rate=.1):
  original_dim = int(x.shape[-1])
  if pos_ids is not None:
    pos_embedding = position_embedding(pos_ids=pos_ids, output_dim=original_dim)
    # x = concatenate([x, pos_embedding])
    x = Add()([x, pos_embedding])
  att_layer = attention_scaled_dot(x)
  att_layer = Dropout(drop_rate)(att_layer)
  x = WeightedSumLayer()([att_layer, x])
  x = Dropout(drop_rate)(x)
  x = BatchNormalization()(x)
  #     if False:
  #         # reduce dim
  #         x = Convolution1D(original_dim, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  #         x = Dropout(drop_rate*2)(x)
  #         x = BatchNormalization()(x)
  return x

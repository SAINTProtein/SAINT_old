import config
import numpy as np
import seaborn as sns
sns.set_style("white")
from attention_distribution_visulization_ulits import *

from utility import *
from attention_module_tools import *
import config
from precision_recall_F1score_utils import *
from model import *
from metric import *
import keras
from keras.callbacks import LambdaCallback
from keras import callbacks, backend
from keras.optimizers import Adam
from pprint import pprint
from keras.models import Model
import gc

###................ load the model...........###
model = get_model()
if config.load_saved_model:
  best_model_file = config.best_model_file
  model = keras.models.load_model(best_model_file, custom_objects={'LayerNormalization':LayerNormalization,'shape_list':shape_list,'backend':backend,'MyLayer':WeightedSumLayer, 'truncated_accuracy':truncated_accuracy})

adam = Adam(lr=config.lr)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              sample_weight_mode='temporal',
              metrics=['accuracy', 'mae', truncated_accuracy])


intermediate_layer_models = []
layer_names = ['softmax_3', 'softmax_4', 'softmax_5','softmax_6', 'softmax_7', 'softmax_8','softmax_9', 'softmax_10', 'softmax_11']
for layer_name in layer_names:
  # intermediate_layer_models.append(Model(inputs=[model.get_layer('pssm_input'), model.get_layer('seq_input'), model.get_layer('position_input')], outputs=model.get_layer(layer_name).output))
  intermediate_layer_models.append(Model(inputs=model.input, outputs=model.get_layer(layer_name).output))

if config.visualize_attention_cb513:
  dataindex = list(range(35, 56))
  labelindex = range(22, 30)
  print('Loading Test data [ CB513 ]...')
  cb513 = load_gz(config.cb513_dataset_dir)
  cb513 = np.reshape(cb513, (514, 700, 57))
  # print(cb513.shape)
  x_test = cb513[:, :, dataindex]
  y_test = cb513[:, :, labelindex]

  cb513_protein_one_hot = cb513[:, :, : 21]
  cb513_protein_one_hot_with_noseq = cb513[:, :, : 22]
  lengths_cb = np.sum(np.sum(cb513_protein_one_hot, axis=2), axis=1).astype(int)
  # print(cb513_protein_one_hot_with_noseq.shape)
  del cb513_protein_one_hot
  gc.collect()

  cb513_seq = np.zeros((cb513_protein_one_hot_with_noseq.shape[0], cb513_protein_one_hot_with_noseq.shape[1]))
  for j in range(cb513_protein_one_hot_with_noseq.shape[0]):
    for i in range(cb513_protein_one_hot_with_noseq.shape[1]):
      datum = cb513_protein_one_hot_with_noseq[j][i]
      cb513_seq[j][i] = int(decode(datum))

  cb513_pos = np.array(range(700))
  cb513_pos = np.repeat([cb513_pos], 514, axis=0)


  attention_distribution = intermediate_layer_models[config.attention_module_number].predict([x_test, cb513_protein_one_hot_with_noseq, cb513_pos])
  evaluateAndShowAttention(cb513_seq, attention_distribution, protein_number=config.protein_number, lengths=lengths_cb,
                           number_to_name_map=number_to_aminoAcidName_map, Total_Protein_Map=config.show_total_protein, amino_acid_number=config.amino_acid_number)

elif config.visualize_attention_casp10:
  casp_one_hot, casp_one_hot_token, casp_pssm, casp_label, lengths_casp, casp_protein_one_hot_with_noseq, casp_pos = load_casp(casp='casp10')
  attention_distribution = intermediate_layer_models[config.attention_module_number].predict([casp_pssm, casp_protein_one_hot_with_noseq, casp_pos])
  evaluateAndShowAttention(casp_protein_one_hot_with_noseq, attention_distribution, protein_number=config.protein_number, lengths=lengths_casp,
                           number_to_name_map=number_to_aminoAcidName_map, Total_Protein_Map=config.show_total_protein, amino_acid_number=config.amino_acid_number)

elif config.visualize_attention_casp11:
  casp_one_hot, casp_one_hot_token, casp_pssm, casp_label, lengths_casp, casp_protein_one_hot_with_noseq, casp_pos = load_casp(casp='casp11')
  attention_distribution = intermediate_layer_models[config.attention_module_number].predict([casp_pssm, casp_protein_one_hot_with_noseq, casp_pos])
  evaluateAndShowAttention(casp_protein_one_hot_with_noseq, attention_distribution, protein_number=config.protein_number, lengths=lengths_casp,
                           number_to_name_map=number_to_aminoAcidName_map, Total_Protein_Map=config.show_total_protein, amino_acid_number=config.amino_acid_number)


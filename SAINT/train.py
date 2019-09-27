from utility import *
from attention_module_tools import *
import config
from model import *
from metric import *
import keras
from keras.callbacks import LambdaCallback
from keras import callbacks, backend
from keras.optimizers import Adam
import gc


## dataset link: http://www.princeton.edu/~jzthree/datasets/ICML2014/

###................. load and process CB6133 dataset for training .............###

cb6133 = load_gz(config.train_dataset_dir)

dataindex = list(range(35, 56))
labelindex = range(22, 30)

total_proteins = cb6133.shape[0]
cb6133 = np.reshape(cb6133, (total_proteins, 700, 57))

cb6133_protein_one_hot = cb6133[:, :, : 21]
cb6133_protein_one_hot_with_noseq = cb6133[:, :, : 22]
lengths = np.sum(np.sum(cb6133_protein_one_hot, axis=2), axis=1).astype(int)

traindata = cb6133[:, :, dataindex]
trainlabel = cb6133[:, :, labelindex]
lengths_train = lengths[:]

cb6133_seq = np.zeros((cb6133_protein_one_hot_with_noseq.shape[0],cb6133_protein_one_hot_with_noseq.shape[1]))
for j in range(cb6133_protein_one_hot_with_noseq.shape[0]):
    for i in range(cb6133_protein_one_hot_with_noseq.shape[1]):
        datum = cb6133_protein_one_hot_with_noseq[j][i]
        #print('index: %d' % i)
        #print('encoded datum: %s' % datum)
        cb6133_seq[j][i] = int(decode(datum))
        #print('decoded datum: %s' % cb6133_seq[j][i])
        #print()
    #break

positions_seq = np.array(range(700))
positions_seq = np.repeat([positions_seq],total_proteins, axis=0)

## freeup some memory of RAM
del cb6133
del cb6133_protein_one_hot
del lengths
gc.collect()


###............... load and process CB513 to validate ..............###

cb513 = load_gz("CB513.npy.gz")
cb513 = np.reshape(cb513, (514, 700, 57))
#print(cb513.shape)
x_test = cb513[:,:,dataindex]
y_test = cb513[:,:,labelindex]

cb513_protein_one_hot = cb513[:, :, : 21]
cb513_protein_one_hot_with_noseq = cb513[:, :, : 22]
lengths_cb = np.sum(np.sum(cb513_protein_one_hot, axis=2), axis=1).astype(int)
#print(cb513_protein_one_hot_with_noseq.shape)
del cb513_protein_one_hot
gc.collect()

cb513_seq = np.zeros((cb513_protein_one_hot_with_noseq.shape[0],cb513_protein_one_hot_with_noseq.shape[1]))
for j in range(cb513_protein_one_hot_with_noseq.shape[0]):
    for i in range(cb513_protein_one_hot_with_noseq.shape[1]):
        datum = cb513_protein_one_hot_with_noseq[j][i]
        cb513_seq[j][i] = int(decode(datum))

cb513_pos = np.array(range(700))
cb513_pos = np.repeat([cb513_pos],514, axis=0)


###................ generate the model...........###
model = get_model()

###............. generate necessary callbacks ..........###
best_model_file = config.best_model_file
checkpoint = callbacks.ModelCheckpoint(best_model_file, monitor='val_truncated_accuracy', verbose=1, save_best_only=True, mode='max')
lr_decay = callbacks.LearningRateScheduler(StepDecay(initAlpha=0.0005, factor=0.9, dropEvery=60))

###............. generate weight masks ..........###
weight_mask_cb513 = np.zeros((x_test.shape[0], 700))
for i in range(len(lengths_cb)):
  weight_mask_cb513[i, : lengths_cb[i]] = 1.0

weight_mask_train = np.zeros((traindata.shape[0], 700))
for i in range(len(lengths_train)):
  weight_mask_train[i, : lengths_train[i]] = 1.0

###............... load saved model ............ ###
if config.load_saved_model:
  model = keras.models.load_model(best_model_file, custom_objects={'LayerNormalization':LayerNormalization,'shape_list':shape_list,'backend':backend,'MyLayer':WeightedSumLayer, 'truncated_accuracy':truncated_accuracy})

adam = Adam(lr=config.lr)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              sample_weight_mode='temporal',
              metrics=['accuracy', 'mae', truncated_accuracy])


###........... train ...............###
if config.to_train:
  model.fit(x=[traindata, cb6133_protein_one_hot_with_noseq, positions_seq], y=trainlabel,
            batch_size=12,
            epochs=100000,
            validation_data=([x_test, cb513_protein_one_hot_with_noseq, cb513_pos], y_test, weight_mask_cb513),
            shuffle=True,
            sample_weight=weight_mask_train,
            callbacks=[checkpoint], verbose=1)
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 23:42:30 2022

@author: Maysam
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import TUDataset
from spektral.layers import GlobalAvgPool
from GWCN_Func import computeLoaderCombine

from GWCN_Net import AGWConv
from tensorflow.keras.regularizers import l2
################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
channels = 128  # Hidden units
layers = 2  # GWCN layers 3
epochs = 200  # Number of training epochs
batch_size = 32  # Batch size  32 10
es_patience = 10  # Patience for early stopping
################################################################################
# LOAD DATA
# ENZYMES  PROTEINS MUTAG
################################################################################
dataset = TUDataset("PROTEINS", clean=True)

idxs = np.random.permutation(300)




# Parameters
F = dataset.n_node_features  # Dimension of node features
n_out = dataset.n_labels  # Dimension of the target

idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr = dataset[idx_tr]
dataset_va = dataset[idx_va]
dataset_te = dataset[idx_te]

steps_per_epoch_tr=int(np.ceil(len(dataset_tr) / batch_size))
steps_per_epoch_te=int(np.ceil(len(dataset_te) / batch_size))
steps_per_epoch_va=int(np.ceil(len(dataset_va) / batch_size))
################################################################################
# Compute Phsi 
################################################################################
# AGWC parameters
thr=1e-1    # threshold parameter for check sparsity of wavelet
scales=[0.4]      # range of scales  [0.4,0.9] 
N_scales = len(scales)
m=40                #Order of polynomial approximation
apx_phsi= True     # approximate Phsi


data_tr=dataset[idx_tr]
data_te=dataset[idx_te]
data_va=dataset[idx_va]


loader_tr,psi_tr,psii_tr=computeLoaderCombine(data_tr,apx_phsi,N_scales,scales,m,epochs,thr)

loader_te,psi_te,psii_te=computeLoaderCombine(data_te,apx_phsi,N_scales,scales,m,1,thr)

loader_va,psi_va,psii_va=computeLoaderCombine(data_va,apx_phsi,N_scales,scales,m,1,thr)

################################################################################
# Parmeters
iterations =1  # Number of layers
share_weights = False  # Share weights 
dropout_skip = 0.75  # Dropout rate for the internal skip connection 
l2_reg = 5e-4  # L2 regularization rate


# BUILD MODEL
################################################################################

class GWCN(Model):
    def __init__(self, channels, n_layers):
        super().__init__()
        self.conv1=AGWConv(channels,iterations=iterations,order=N_scales,
                           share_weights=share_weights,dropout_rate=dropout_skip,
                           activation="elu",gcn_activation="elu",kernel_regularizer=l2(l2_reg),)
        self.convs = []
        for _ in range(1, n_layers):
            self.convs.append(
                AGWConv(channels,iterations=iterations,order=N_scales,
                        share_weights=share_weights,dropout_rate=dropout_skip,
                        activation="elu",gcn_activation="elu",kernel_regularizer=l2(l2_reg),)            )
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(channels, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(n_out, activation="softmax")

    def call(self, inputs):
        x, psi, psii, a, i= inputs
        x = self.conv1([x, psi, psii, a])
        for conv in self.convs:
            x = conv([x, psi, psii, a])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


# Build model
model = GWCN(channels, layers)
opt = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()


################################################################################
# FIT MODEL
################################################################################
# @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs,target):
    with tf.GradientTape() as tape:
        predictions = model(inputs,training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    output = []
    for batch in loader:
        inputs, target = batch
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
        )
        output.append(outs)
    return np.mean(output, 0)


b=[]
print("Fitting model")
patience = es_patience
best_val_loss = np.inf
epoch = 0
current_batch = 0
model_lss = model_acc = 0
for batch in loader_tr:
    lss, acc = train_step(*batch)

    model_lss += lss.numpy()
    model_acc += acc.numpy()
    current_batch += 1
   
    if current_batch == steps_per_epoch_tr:
        model_lss /= steps_per_epoch_tr
        model_acc /= steps_per_epoch_tr
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}".format(
                epoch, model_lss, model_acc, val_loss, val_acc
            )
        )

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        model_loss = 0
        model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print("Testing model")
model_lss = model_acc = 0
for batch in loader_te:
    inputs, target = batch
 
    predictions = model(inputs, training=False)
    model_lss += loss_fn(target, predictions)
    model_acc += tf.reduce_mean(categorical_accuracy(target, predictions))

model_lss /= steps_per_epoch_te
model_acc /= steps_per_epoch_te
print("Done. Test loss: {}. Test acc: {}".format(model_lss, model_acc))


# get parameters 
ww=model.get_weights()
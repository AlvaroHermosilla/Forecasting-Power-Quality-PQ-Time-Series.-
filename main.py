import numpy as np 
import pandas as pd
import tensorflow as tf
from utils import to_supervised,Model_wrapped,lr_scheduler
from sklearn.utils import shuffle
from Networks import ModelFactory
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision
import time
import json
from functools import partial
#-----------------------------------------------------------------------------------------------------------------------
# 1 - Import Dataset.
#     This dataset is public and can be dowload https://github.com/joaquinjgz/Dataset_university_data_center
#-----------------------------------------------------------------------------------------------------------------------
ni = 144*0
nf = 144*247
df = pd.read_csv('dataset_full.csv', header=0)
imag3a = np.array(df['imag3a'],dtype=float)[ni:nf]
imag3b = np.array(df['imag3b'],dtype=float)[ni:nf]
imag3c = np.array(df['imag3c'],dtype=float)[ni:nf]
imag3n = np.array(df['imag3n'],dtype=float)[ni:nf]

MTS = np.stack([imag3a,imag3b,imag3c,imag3n], axis=1)

supervised_transform = {
                            'n_in':            144,
                            'n_out':           144,
                            'n_out_features':  4
                        }
global_settings =   {
                        'training_split':       0.4,
                        'validation_split':     0.8
                    }
Hyperparameter = {
                            'min_layer': 1,                                                            
                            'max_layer': 5,                                                            
                            'min_neurons': 32,                                                         
                            'max_neurons': 128,
                            'kernel_min' : 2,
                            'kernel_max' : 7,                                                    
                            'activation_functions':['relu','tanh','sigmoid'],                          
                            'lr_min':0.01,
                            'lr_max':0.1,                                          
                            }
Parameters =    {   
                    'verbose':              1,
                    'Dropout':              0.1,
                    'epochs_max':           300,
                    'warmup_epochs':        30,
                    'lr_max':               0.1,
                    'batch_size':           512,
                    'loss':                 'mse'
                }
Tuner = {
            'objective': 'val_loss',
            'max_trials': 40,
            'directory': "afinacion_red",
            'project_name': 'Tuner',
            'overwrite': True,
            'num_initial': 15,
    
}

#-----------------------------------------------------------------------------------------------------------------------
# 2 - Data preprocessing
#-----------------------------------------------------------------------------------------------------------------------
# Convert the data into a supervised problem
X, Y= to_supervised(dataset = MTS,
                    n_in = supervised_transform['n_in'],
                    n_out = supervised_transform['n_out'],
                    n_out_features = supervised_transform['n_out_features'])
X, Y = shuffle(X, Y,random_state = 1)
X = np.array(X)
Y = np.array(Y)
#Normalize the data.
Xmax = np.max(a=X, axis=0)
Xmin = np.min(a=X, axis=0)
Ymax = np.max(a=Y, axis=0)
Ymin = np.min(a=Y, axis=0)
X_normalized = (X-Xmin)/(Xmax-Xmin)
Y_normalized = (Y-Ymin)/(Ymax-Ymin)
# Split the data into train,validation and test.
X_train = X_normalized[:int(global_settings['training_split']*len(X_normalized))]
X_valid = X_normalized[int(global_settings['training_split']*len(X_normalized)):int(global_settings['validation_split']*len(X_normalized))]
X_test = X_normalized[int(global_settings['validation_split']*len(X_normalized)):]

Y_train = Y_normalized[:int(global_settings['training_split']*len(Y_normalized))]
Y_valid = Y_normalized[int(global_settings['training_split']*len(Y_normalized)):int(global_settings['validation_split']*len(Y_normalized))]
Y_test = Y_normalized[int(global_settings['validation_split']*len(Y_normalized)):]
#-----------------------------------------------------------------------------------------------------------------------
# 3 - Instantiate the models
#-----------------------------------------------------------------------------------------------------------------------
factory = ModelFactory(X_train, Y_train, Hyperparameter, Parameters)
model = factory.get_model('MLP')

#-----------------------------------------------------------------------------------------------------------------------
# 3 - Hyperparameter optimization.
#-----------------------------------------------------------------------------------------------------------------------
tf.config.optimizer.set_jit(False)  # Activa la compilación JIT (XLA)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

model_checkpoint_callback = ModelCheckpoint(
      filepath=f'./Resultados/{model.__name__.replace('model_', '')}.h5',
      save_weights_only=False,
      monitor="denorm_mse",
      mode='max',
      save_best_only=True
      )

with tf.device('/device:GPU:0'):
        best_model,best_hps = Model_wrapped(model=model,train=(X_train,Y_train),valid=(X_valid,Y_valid),Parameters=Parameters,Hyperparameter=Hyperparameter,Tuner=Tuner)


#-----------------------------------------------------------------------------------------------------------------------
# 4 - Train the model.
#-----------------------------------------------------------------------------------------------------------------------
time0 = time.time()
X_set = np.concatenate([X_train, X_valid])
Y_set = np.concatenate([Y_train, Y_valid])
with tf.device('/device:GPU:0'):
        schedule_fn = partial(lr_scheduler, lr_max=Parameters['lr_max'], warmup_epochs=Parameters['warmup_epochs'])
        lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=schedule_fn, verbose=Parameters['verbose'])
        history = best_model.fit(X_set, Y_set,
                                 epochs=Parameters['epochs_max'],
                                 batch_size=Parameters['batch_size'],
                                 callbacks=[lr_callback,model_checkpoint_callback])
        training_time = time.time() - time0
with open(f'./Resultados/history_{model.__name__.replace('model_', '')}.json', 'w') as f:
      json.dump(history.history, f)
hyperparam_dict = best_hps.values
    
df_hyperparameters = pd.DataFrame([hyperparam_dict])
df_hyperparameters.to_csv(f'./Resultados/hyperparameters_{model.__name__.replace('model_', '')}.csv',
                               mode= 'w',
                               index=False)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 5 - Model Evaluation.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

testY_hat = best_model.predict(X_test)
error_metrics = []
RMSE = np.sqrt(np.mean(np.square(Y_test-testY_hat), axis=(1,2)))
RSE = np.sqrt(np.sum(np.square(Y_test-testY_hat), axis=(1,2)))/np.sqrt(np.sum(np.square(Y_test-np.mean(Y_test, axis=(1,2), keepdims=True)), axis=(1,2)))
MAE = np.mean(np.abs(Y_test-testY_hat), axis=(1,2))
epsilon = 0.0001  
MAPE = 100.0 * np.mean(np.abs((Y_test - testY_hat) / np.where(Y_test == 0, epsilon, Y_test)), axis=(1,2))
MBE = np.mean(Y_test-testY_hat, axis=(1,2))
error_metrics.append([np.mean(RMSE), np.std(RMSE),
                        np.mean(RSE), np.std(RSE),
                        np.mean(MAE), np.std(MAE),
                        np.mean(MAPE), np.std(MAPE),
                        np.mean(MBE), np.std(MBE),
                        training_time])
error_metrics
df_error_metrics = pd.DataFrame(error_metrics)
df_error_metrics.to_csv(path_or_buf=f'./Resultados/error_metrics_{model.__name__.replace('model_', '')}.csv',
                            header=['RMSE_mean', 'RMSE_std', 'RSE_mean', 'RSE_std', 'MAE_mean', 'MAE_std','MAPE_mean','MAPE_std', 'MBE_mean', 'MBE_std','Time'],
                            mode='w',
                            index=False)

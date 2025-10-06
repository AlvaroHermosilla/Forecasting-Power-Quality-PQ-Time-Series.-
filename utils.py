import numpy as np
import gc
import tensorflow as tf
import keras_tuner as kt
from functools import partial

def to_supervised(dataset, n_in, n_out, n_out_features):
    """
    Function to convert a time series into a supervised learning problem with X/Y

    ARGUMENTS:
    dataset:            Time series with the folowing format: [[x0, y0, z0...],[x1, y1, z1...]...]
    n_in:               Input time steps (i.e.: 10, 20, 30...)
    n_out:              Output time steps (i.e.: 5, 10, 15...)
    n_out_features:     Just the first n_out_features input features will be available in Y, therefore,
                        it must be less than or equal to the number of input features (i.e.: 1, 2, 3, 4...).
    
    RETURNS:
    X/Y:                Input and output data (supervised learning)
    """
    X = []
    Y = []
    #Build a version of the dataset without the features to be removed at the output
    if((len(np.shape(dataset)))<2):                    
        dataset_without_features = dataset.copy()
    elif(n_out_features>=np.shape(dataset)[1]):         
        dataset_without_features = dataset.copy()
    else:
        dataset_without_features = np.delete(arr=dataset, obj=range(n_out_features,np.shape(dataset)[1]), axis=1)

    if(len(dataset)>n_in+n_out+1):
        for i in range(len(dataset)-n_in-n_out+1):
            X.append(dataset[i:i+n_in])
            Y.append(dataset_without_features[i+n_in:i+n_in+n_out])
        return np.array(X), np.array(Y)
    else:
        return np.array([]), np.array([])
    
def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def lr_scheduler(epoch, lr, lr_max, warmup_epochs):
                    if epoch < warmup_epochs:
                        return (lr_max/warmup_epochs)*(epoch+1)
                    else:
                        return lr_max/np.sqrt(epoch+1-warmup_epochs)
                    
def Model_wrapped(model,train,valid,Parameters,Hyperparameter,Tuner):
    """
    Function to run hyperparameter tuning for the models.

    ARGUMENTS:
        model:              Neural network model such as MLP, LSTM, Conv1D, etc.
        train:              Training dataset split into X_train and Y_train.
        valid:              Validation dataset split into X_valid and Y_valid.
        parameters:         Dictionary of model parameters.
        hyperparameters:    Dictionary defining the hyperparameter search space.

    RETURN:
        best_model, best_hps:   The model with the best performance and its hyperparameters.
    """

    tf.random.set_seed(1) 
    X_train,Y_train = train 
    X_valid,Y_valid = valid
    # Build the pipeline.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(Parameters['batch_size']).prefetch(tf.data.AUTOTUNE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid)).batch(Parameters['batch_size']).prefetch(tf.data.AUTOTUNE)
    schedule_fn = partial(lr_scheduler, lr_max=Parameters['lr_max'], warmup_epochs=Parameters['warmup_epochs'])
    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=schedule_fn, verbose=Parameters['verbose'])
    clear_memory()
    #Define the hyperparameters tuning
    with tf.device('/device:GPU:0'):
        tuner = kt.BayesianOptimization(
            hypermodel=model,     
            objective=kt.Objective("val_mse", direction="min"),       
            max_trials=Tuner['max_trials'], 
            num_initial_points=Tuner['num_initial'],  
            executions_per_trial=1,  
            directory=Tuner['directory'],  
            project_name=Tuner['project_name'],      
            overwrite = Tuner['overwrite'],           
        )
        tuner.search(
            train_dataset,
            epochs=Parameters['epochs_max'],
            batch_size=Parameters['batch_size'],
            validation_data=valid_dataset,
            callbacks=[lr_callback]  
        )
    # Get the best model and hyperparameters.
    best_hps = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hps)
    clear_memory()

    return best_model,best_hps
                        
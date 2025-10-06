import tensorflow as tf
import keras_tuner as kt
from keras.layers import LSTM, GRU, Bidirectional, Dense, Conv1D

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
#---------------------------------------------------------------
# Build the base class.
#---------------------------------------------------------------
class BaseTimeSeriesModel:
    def __init__(self, X_train, Y_train, Hyperparameter: dict, Parameters: dict):
        self.X_train = X_train
        self.Y_train = Y_train
        self.Hyperparameter = Hyperparameter
        self.Parameters = Parameters

    def build_input_layer(self):
        return tf.keras.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]), name='input_layer')

    def build_output_layer(self, x):
        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.Y_train.shape[2]))(x)

    def compile_model(self, inputs, outputs, hp):
        initial_lr = hp.Float('learning_rate',
                            min_value=self.Hyperparameter['lr_min'],
                            max_value=self.Hyperparameter['lr_max']
                            )
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=self.Parameters['loss'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-8),
            metrics=["mse"]
        )
        return model

#---------------------------------------------------------------
# Build the models classes.
#---------------------------------------------------------------
class MultilayerPerceptron(BaseTimeSeriesModel):
    def model_MLP(self, hp):
        with strategy.scope():
            inputs = self.build_input_layer()
            x = inputs
            num_layers = hp.Int('Num Layers', 
                                self.Hyperparameter['min_layer'], 
                                self.Hyperparameter['max_layer']
                                )

            for i in range(num_layers):
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(
                        units=hp.Int(f'Layer Neurons {i+1}', 
                                     self.Hyperparameter['min_neurons'], 
                                     self.Hyperparameter['max_neurons']),
                        activation=hp.Choice(f'Activation Function {i+1}', 
                                             self.Hyperparameter['activation_functions'])
                    )
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)

            outputs = self.build_output_layer(x)
            return self.compile_model(inputs, outputs, hp)
        

class LongShortTermMemory(BaseTimeSeriesModel):
    def model_LSTM(self, hp):
        with strategy.scope():
            inputs = self.build_input_layer()
            x = inputs
            num_layers = hp.Int('Num Layers', 
                                self.Hyperparameter['min_layer'], 
                                self.Hyperparameter['max_layer'])

            for i in range(num_layers):
                x = LSTM(
                    units=hp.Int(f'Layer Neurons {i+1}', 
                                 self.Hyperparameter['min_neurons'], 
                                 self.Hyperparameter['max_neurons']),
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)

            outputs = self.build_output_layer(x)
            return self.compile_model(inputs, outputs, hp)
     

class GatedRecurrentUnit(BaseTimeSeriesModel):
    def model_GRU(self, hp):
        with strategy.scope():

            inputs = self.build_input_layer()
            x = inputs
            num_layers = hp.Int('Num Layers', 
                                self.Hyperparameter['min_layer'], 
                                self.Hyperparameter['max_layer'])
            # Hidden layers
            for i in range(num_layers):
                x = GRU(
                    units=hp.Int(f'Layer Neurons {i+1}', 
                                 self.Hyperparameter['min_neurons'], 
                                 self.Hyperparameter['max_neurons']),
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)

            outputs = self.build_output_layer(x)
            return self.compile_model(inputs, outputs, hp)
        

class BidirectionalLongShortTermMemory(BaseTimeSeriesModel):
    def model_BiLSTM(self, hp):
        with strategy.scope():
            inputs = self.build_input_layer()
            x = inputs
            # First bidirectional layer
            x = tf.keras.layers.Bidirectional(
                LSTM(
                    units=hp.Int('Layer Neurons 1',
                                 self.Hyperparameter['min_neurons'],
                                 self.Hyperparameter['max_neurons']),
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True
                ),
                merge_mode='concat'
            )(x)
            x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)
            # Hidden layers.
            num_layers = hp.Int('Num Layers',
                                self.Hyperparameter['min_layer'],
                                self.Hyperparameter['max_layer'])
            for i in range(num_layers):
                x = tf.keras.layers.Bidirectional(
                    LSTM(
                        units=hp.Int(f'Layer Neurons {i+2}',
                                     self.Hyperparameter['min_neurons'],
                                     self.Hyperparameter['max_neurons']),
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        return_sequences=True
                    ),
                    merge_mode='concat'
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)
            outputs = self.build_output_layer(x)
            return self.compile_model(inputs, outputs, hp)

class BidirectionalGatedRecurrentUnit(BaseTimeSeriesModel):
    def model_BiGRU(self, hp):
        with strategy.scope():
            # Capa de entrada
            inputs = self.build_input_layer()
            x = inputs
            # First biddirectional layer
            x = tf.keras.layers.Bidirectional(
                GRU(
                    units=hp.Int('Layer Neurons 1',
                                 self.Hyperparameter['min_neurons'],
                                 self.Hyperparameter['max_neurons']),
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True
                ),
                merge_mode='concat'
            )(x)
            x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)
            # Hidden layers
            num_layers = hp.Int('Num Layers',
                                self.Hyperparameter['min_layer'],
                                self.Hyperparameter['max_layer'])
            for i in range(num_layers):
                x = tf.keras.layers.Bidirectional(
                    GRU(
                        units=hp.Int(f'Layer Neurons {i+2}',
                                     self.Hyperparameter['min_neurons'],
                                     self.Hyperparameter['max_neurons']),
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        return_sequences=True
                    ),
                    merge_mode='concat'
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)
            # Capa de salida
            outputs = self.build_output_layer(x)
            # Compilación del modelo
            return self.compile_model(inputs, outputs, hp)

class Convolutional1D(BaseTimeSeriesModel):
    def model_Conv1D(self, hp):
        with strategy.scope():
            inputs = self.build_input_layer()
            x = inputs
            # Convolutional layers
            num_layers = hp.Int('Num Layers',
                                self.Hyperparameter['min_layer'],
                                self.Hyperparameter['max_layer'])
            for i in range(num_layers):
                x = tf.keras.layers.Conv1D(
                    filters=hp.Int(f'Layer Neurons {i+1}',
                                   self.Hyperparameter['min_neurons'],
                                   self.Hyperparameter['max_neurons']),
                    kernel_size=hp.Int(f'Kernel Size {i+1}',
                                       self.Hyperparameter['kernel_min'],
                                       self.Hyperparameter['kernel_max']),
                    padding='causal',
                    activation=hp.Choice(f'Activation Function {i+1}',
                                         self.Hyperparameter['activation_functions'])
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)
            outputs = self.build_output_layer(x)
            return self.compile_model(inputs, outputs, hp)

class ConvolutionalLSTM(BaseTimeSeriesModel):
    def model_Conv1D_LSTM(self, hp):
        with strategy.scope():
            inputs = self.build_input_layer()
            x = inputs
            # Convolutional layers
            num_conv_layers = hp.Int('Num Conv Layers',
                                     self.Hyperparameter['min_layer'],
                                     self.Hyperparameter['max_layer'])
            for i in range(num_conv_layers):
                x = tf.keras.layers.Conv1D(
                    filters=hp.Int(f'Conv Filters {i+1}',
                                   self.Hyperparameter['min_neurons'],
                                   self.Hyperparameter['max_neurons']),
                    kernel_size=hp.Int(f'Kernel Size {i+1}',
                                       self.Hyperparameter['kernel_min'],
                                       self.Hyperparameter['kernel_max']),
                    padding='causal',
                    activation=hp.Choice(f'Conv Activation {i+1}',
                                         self.Hyperparameter['activation_functions'])
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)
            # LSTM layers
            num_lstm_layers = hp.Int('Num LSTM Layers',
                                     self.Hyperparameter['min_layer'],
                                     self.Hyperparameter['max_layer'])
            for i in range(num_lstm_layers):
                x = tf.keras.layers.LSTM(
                    units=hp.Int(f'LSTM Units {i+1}',
                                 self.Hyperparameter['min_neurons'],
                                 self.Hyperparameter['max_neurons']),
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True
                )(x)
                x = tf.keras.layers.Dropout(rate=self.Parameters['Dropout'])(x)
            outputs = self.build_output_layer(x)
            return self.compile_model(inputs, outputs, hp)

#---------------------------------------------------------------
# Build the model factory.
#---------------------------------------------------------------
class ModelFactory:
    def __init__(self, X_train, Y_train, Hyperparameter, Parameters):
        self.X_train = X_train
        self.Y_train = Y_train
        self.Hyperparameter = Hyperparameter
        self.Parameters = Parameters

    def get_model(self, model_type):
        model_type = model_type.upper()

        if model_type == "MLP":
            return MultilayerPerceptron(self.X_train, self.Y_train, self.Hyperparameter, self.Parameters).model_MLP

        elif model_type == "LSTM":
            return LongShortTermMemory(self.X_train, self.Y_train, self.Hyperparameter, self.Parameters).model_LSTM

        elif model_type == "GRU":
            return GatedRecurrentUnit(self.X_train, self.Y_train, self.Hyperparameter, self.Parameters).model_GRU
        
        elif model_type == "BILSTM":
            return BidirectionalLongShortTermMemory(self.X_train, self.Y_train, self.Hyperparameter, self.Parameters).model_BiLSTM
        
        elif model_type == "BIGRU":
            return BidirectionalGatedRecurrentUnit(self.X_train, self.Y_train,self.Hyperparameter, self.Parameters).model_BiGRU
        
        elif model_type == "CONV1D":
            return Convolutional1D(self.X_train, self.Y_train,self.Hyperparameter, self.Parameters).model_Conv1D
        
        elif model_type == "CNN_LSTM":
            return ConvolutionalLSTM(self.X_train, self.Y_train,self.Hyperparameter, self.Parameters).model_Conv1D_LSTM

        else:
            raise ValueError(f"Model type '{model_type}' not recognized. Use 'MLP', 'LSTM', or 'GRU'.")
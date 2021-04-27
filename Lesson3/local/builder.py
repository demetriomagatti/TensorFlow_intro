import numpy as np
import tensorflow as tf
import preprocessing as pr


class ModelBuilder:
    def __init__(self,*args,**kwargs):
        # Default parameters
        self.dataset = None
        self.metadata = None
        self.flatten_input = True
        self.normalize = True
        self.cache_dataset  = True
        self.shuffle_train = True
        self.epochs = 5
        self.input_shape = (28, 28, 1)
        self.num_hidden_layers = 1
        self.hidden_layer_neurons = 128
        self.hidden_layer_activation = tf.nn.relu
        self.prediction_layer_neurons = 10
        self.prediction_layer_activation = tf.nn.softmax
        self.batch_size = 32
        self.optimizer = 'adam'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics= ['accuracy']
        # Update parameters
        self.__dict__.update(kwargs)


    def define(self):
        model = keras.Sequential()
        if self.flatten_input:
            model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
        if isinstance(self.hidden_layer_neurons, int):
            for i in range(self.num_hidden_layers):
                model.add(tf.keras.layers.Dense(self.hidden_layer_neurons, activation=self.hidden_layer_activation))
        else:
            for i in range(self.num_hidden_layers):
                model.add(tf.keras.layers.Dense(self.hidden_layer_neurons[i], activation=self.hidden_layer_activation[i]))
        model.add(tf.keras.layers.Dense(self.prediction_layer_neurons, activation=self.prediction_layer_activation))
        return model


    def train(self,model):
        train_dataset, test_dataset = self.dataset['train'], self.dataset['test']
        class_names = self.metadata.features['label'].names
        num_train_examples = self.metadata.splits['train'].num_examples
        num_test_examples = self.metadata.splits['test'].num_examples
        if self.normalize:
            train_dataset.map(pr.normalize)
        if self.cache_dataset:
            train_dataset =  train_dataset.cache()
        model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
        return train_history = model.fit(train_dataset, epochs=self.epochs, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
import os
import datetime
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import L2


class ModelTrainer:
    def __init__(self, train_dir, val_dir, save_dir, hyperparams):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.save_dir = save_dir
        self.hyperparams = hyperparams
        self.model_names = {}

    def get_generators(self, preprocessing_function, target_size):
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocessing_function
        )
        val_datagen = ImageDataGenerator(rescale=1.0/255, preprocessing_function=preprocessing_function)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=target_size,
            batch_size=self.hyperparams['batch_size'],
            class_mode='categorical',
            shuffle=True
        )
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=target_size,
            batch_size=self.hyperparams['batch_size'],
            class_mode='categorical'
        )
        return train_generator, val_generator

    def get_checkpoint_callback(self, model, iteration):
        model_name = self.model_names[model]
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        filepath = os.path.join(save_path, f'iteration-{iteration:02d}-epoch-{{epoch:02d}}-val_loss-{{val_loss:.3f}}-val_acc-{{val_accuracy:.3f}}.h5')
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        return checkpoint
    
    def save_history(self, history, model):
        model_name = self.model_names[model]
        with open(os.path.join(self.save_dir, f'{model_name}_report.json'), 'w') as file:
            json.dump(history, file, indent=4)

    def build_model(self, base_model_func, image_size, num_classes):
        if num_classes < 2:
            raise ValueError("Output size cannot be lower than 2")
        base_model = base_model_func(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(self.hyperparams['dense_units'], activation='relu', kernel_regularizer=L2(self.hyperparams['l2_reg']))(x)
        x = Dropout(self.hyperparams['dropout_rate'])(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        self.model_names[model] = base_model_func.__name__
        return base_model, model

    def get_optimizer(self, optimizer_name, learning_rate):
        optimizers = {
            'sgd': lambda: SGD(
                learning_rate=learning_rate,
                momentum=self.hyperparams.get('momentum', 0.0),
                nesterov=self.hyperparams.get('nesterov', False)
            ),
            'adam': lambda: Adam(
                learning_rate=learning_rate,
                beta_1=self.hyperparams.get('beta_1', 0.9),
                beta_2=self.hyperparams.get('beta_2', 0.999),
                epsilon=self.hyperparams.get('epsilon', 1e-07),
                amsgrad=self.hyperparams.get('amsgrad', False)
            ),
            'rmsprop': lambda: RMSprop(
                learning_rate=learning_rate,
                rho=self.hyperparams.get('rho', 0.9),
                momentum=self.hyperparams.get('momentum', 0.0),
                epsilon=self.hyperparams.get('epsilon', 1e-07),
                centered=self.hyperparams.get('centered', False)
            )
        }

        try:
            return optimizers[optimizer_name.lower()]()
        except KeyError:
            raise ValueError(f"Unsupported optimizer. Supported options are: {', '.join(optimizers.keys())}.")

    def create_log_dir(self, model_name, iteration_number):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", f"{model_name}_iter{iteration_number}", current_time)
        return log_dir

    def train_model(self, model, base_model, train_generator, val_generator, iteration):
        model_name = self.model_names[model]
        print(f"Starting training for {model_name}")

        # Freeze all layers initially
        for layer in base_model.layers:
            layer.trainable = False

        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer_feature_extraction = self.get_optimizer(self.hyperparams['optimizer'], self.hyperparams['learning_rate_feature_extraction'])
        model.compile(optimizer=optimizer_feature_extraction, loss=loss_fn, metrics=['accuracy'])
        
        log_dir = self.create_log_dir(model_name, iteration)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Callbacks for feature extraction (without learning rate scheduling)
        callbacks_feature_extraction = [
            self.get_checkpoint_callback(model, iteration),
            EarlyStopping(monitor='val_loss', mode='min', patience=self.hyperparams['patience_es_feature_extraction'], verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=self.hyperparams['factor_lr'], patience=self.hyperparams['patience_lr'], min_lr=self.hyperparams['min_lr'], verbose=1),
            tensorboard_callback
        ]

        feature_extraction_epochs = int(self.hyperparams['epochs'] * 0.4)
        history_feature_extraction = model.fit(
            train_generator,
            epochs=feature_extraction_epochs,
            validation_data=val_generator,
            callbacks=callbacks_feature_extraction
        )

        print(f"Starting fine tuning for {model_name}")

        # Unfreeze a percentage of layers for fine tuning (hyperparameter controlled)
        unfreeze_percentage = self.hyperparams['unfreeze_percentage']
        num_layers_to_unfreeze = int(len(base_model.layers) * unfreeze_percentage)
        
        for layer in base_model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True

        # Fine-tuning with learning rate scheduler (ReduceLROnPlateau)
        callbacks_fine_tuning = [
            self.get_checkpoint_callback(model, iteration),
            EarlyStopping(monitor='val_loss', mode='min', patience=self.hyperparams['patience_es_fine_tuning'], verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=self.hyperparams['factor_lr'], patience=self.hyperparams['patience_lr'], min_lr=self.hyperparams['min_lr'], verbose=1),
            tensorboard_callback
        ]

        optimizer_fine_tuning = self.get_optimizer(self.hyperparams['optimizer'], self.hyperparams['learning_rate_fine_tuning'])
        model.compile(optimizer=optimizer_fine_tuning, loss=loss_fn, metrics=['accuracy'])
        history_fine_tuning = model.fit(
            train_generator,
            epochs=self.hyperparams['epochs'],
            initial_epoch=len(history_feature_extraction.epoch),
            validation_data=val_generator,
            callbacks=callbacks_fine_tuning
        )

        # Combined history for both phases
        full_history = {
            'accuracy': history_feature_extraction.history['accuracy'] + history_fine_tuning.history['accuracy'],
            'val_accuracy': history_feature_extraction.history['val_accuracy'] + history_fine_tuning.history['val_accuracy'],
            'loss': history_feature_extraction.history['loss'] + history_fine_tuning.history['loss'],
            'val_loss': history_feature_extraction.history['val_loss'] + history_fine_tuning.history['val_loss']
        }

        return full_history





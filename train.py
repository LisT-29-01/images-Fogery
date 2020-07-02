from model import *

import os
import data_processor
import argparse
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', tpye=int, default=512)    #input size for training
parser.add_argument('--batth_size', type=int, default=32)   #batch size for training
parser.add_argument('--init_learning_rate',type=float, default=0.0001)    #initial learning rate
parser.add_argument('--lr_decay_rate',type=float, default=0.94)     #decay rate for the learning rate
parser.add_argument('--lr_decay_steps',type=int,default=130)    #nums of the steps after which the learning rate is decayed by decay rate
parser.add_argument('--max_epochs', type=int, default=800) # maximum number of epochs
parser.add_argument('--real_training_data_path', type=str, default='')   # path to real training data
parser.add_argument('--fake_training_data_path', type=str, default='')   # path to fake training data
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--checkpoint_path', type=str, default='tmp/image_classification') # path to a directory to save model checkpoints during training
parser.add_argument('--last_epoch_train',type=int,default=0)    #last epoch has trained
FLAGS = parser.parse_args()

lastEpoch = FLAGS.last_epoch_train

class ValidationEvaluator(Callback):
    def __init__(self, validation_data, validation_log_dir, period=5):
        super(Callback, self).__init__()

        self.period = period
        self.validation_data = validation_data
        self.validation_log_dir = validation_log_dir
        self.val_writer = tf.summary.create_file_writer(self.validation_log_dir)

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.period == 0:
            val_loss = self.model.evaluate([self.validation_data[0]],[self.validation_data[1]],batch_size=FLAGS.batch_size)
            print('\nEpoch %d: val_loss: %.4f' % (epoch + lastEpoch+1, val_loss))
            with self.val_writer.as_default():
                tf.summary.scalar('loss',val_loss,step = epoch+lastEpoch+1)
            self.val_writer.flush()

class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, period, save_weights_only):
        super(CustomModelCheckpoint, self).__init__()
        self.period = period
        self.path = path
        # We set the model (non multi gpu) under an other name
        self.model_for_saving = model
        self.epochs_since_last_save = 0
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_weights_only:
                self.model_for_saving.save_weights(self.path.format(epoch=epoch + lastEpoch + 1, **logs), overwrite=True)
            else:
                self.model_for_saving.save(self.path.format(epoch=epoch + lastEpoch + 1, **logs), overwrite=True)

def lr_decay(epoch):
    return FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, epoch + lastEpoch // FLAGS.lr_decay_steps)

def main(argv = None):
    os.mkdir(FLAGS.checkpoint_path)

    train_data_generator = data_processor.generator(FLAGS)
    val_data = data_processor.load_val_data(FLAGS)
    train_sample_count = data_processor.count_sample(FLAGS)

    classifier = model_classification()
    model = classifier.model

    lr_scheduler = LearningRateScheduler(lr_decay)
    validation_evaluator = ValidationEvaluator(val_data, validation_log_dir=FLAGS.checkpoint_path + '/val')
    ckpt = CustomModelCheckpoint(model=classifier.model, path=FLAGS.checkpoint_path + '/model-{epoch:02d}.h5', period=FLAGS.save_checkpoint_epochs, save_weights_only=True)
    callbacks = [lr_scheduler,ckpt,validation_evaluator]

    opt = Adam(FLAGS.init_learning_rate)

    model.compile(loss='binary-crossentropy',optimizer=opt,metrics=['accuracy'])
    model.summary()

    model_json = model.model_to_json()
    with open(FLAGS.checkpoint_path + '/model.json', 'w') as json_file:
        json_file.write(model_json)

    history = model.fit_generator(train_data_generator,epochs=FLAGS.max_epochs,steps_per_epoch=train_sample_count/FLAGS.batch_size,callbacks=callbacks)
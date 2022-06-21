import tensorflow as tf
from tensorflow.core.protobuf.config_pb2 import ConfigProto, GPUOptions
from tensorflow.python.client.session import InteractiveSession
from keras import backend as K
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpu_options = GPUOptions(allow_growth=True)
session = InteractiveSession(config=ConfigProto(gpu_options=gpu_options))

img_height, img_width = 197, 197

num_classes         = 7
epochs_top_layers   = 5
epochs_all_layers   = 30
batch_size          = 32

train_datagen = ImageDataGenerator(
    rotation_range  = 10,
    shear_range     = 10,
    zoom_range      = 0.2,
    fill_mode       = 'reflect',
    horizontal_flip = True
)
validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    'dataset/splited/train',
    target_size=(img_height, img_width),
    # subset='training',
    batch_size  = batch_size,
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/splited/validation',
    target_size=(img_height, img_width),
    # subset='training',
    batch_size=batch_size,
)

test_set = test_datagen.flow_from_directory('dataset/splited/test',
                                            target_size=(img_height, img_width),
                                            # subset='validation',
                                            shuffle=False
                                            )

base_model = VGGFace(
    model       = 'vgg16',
    include_top = False,
    weights     = 'vggface',
    input_shape = (img_height, img_width, 3))

for layer in base_model.layers:
    layer.trainable = True

x = base_model.output

x = Dense(1024)(x)
x = Dense(1024)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Flatten()(x)
x = Dense(2048)(x)
x = Dense(2048)(x)
x = Dense(2048)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(4096)(x)
x = Dense(4096)(x)
x = Dense(4096)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(8192)(x)
x = Dense(8192)(x)
x = Dense(8192)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

predictions = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(
    optimizer   = SGD(learning_rate = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True),
    loss        = 'categorical_crossentropy',
    metrics     = ['accuracy'])

def scheduler(epoch):
    updated_lr = K.get_value(model.optimizer.lr) * 0.25
    if (epoch % 3 == 0) and (epoch != 0):
        K.set_value(model.optimizer.lr, updated_lr)
        print(K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

reduce_lr_plateau = ReduceLROnPlateau(
	monitor 	= 'val_loss',
	factor		= 0.5,
	patience	= 3,
	mode 		= 'auto',
	min_lr		= 1e-8)

early_stop = EarlyStopping(
	monitor 	= 'val_loss',
	patience 	= 10,
	mode 		= 'auto')


model.summary()

r = model.fit_generator(
    generator           = train_generator,
    validation_data     = validation_generator,
    validation_steps    =  validation_generator.samples // batch_size,
    steps_per_epoch     = train_generator.samples // batch_size,
    epochs              = epochs_all_layers,
    callbacks           = [reduce_lr, reduce_lr_plateau, early_stop])

model.save('resnet_trained.h5')

print("Evaluate on test data")
results = model.evaluate_generator(
    test_set
)
print("test loss, test acc:", results)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
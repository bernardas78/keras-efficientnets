from keras_efficientnets import EfficientNetB0
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from sys import platform

gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)


def get_iterator(data_folder, batch_size=32, target_size=224, shuffle=True):
    dataGen = ImageDataGenerator(
        rescale=None,
        preprocessing_function=None)

    real_target_size = target_size

    data_iterator = dataGen.flow_from_directory(
        directory=data_folder,
        target_size=(real_target_size, real_target_size),
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode='categorical')
    return data_iterator


def train_model(epochs=1):
    input_size = (224,224,3)

    if platform == 'linux':
        data_dir = r"/home/bernardas/IsKnown_Images_ForEfficientNet/Bal_v14/Ind-0"
    else:
        data_dir=r"c:/IsKnown_Images_IsVisible/Bal_v14/Ind-0"
    data_dir_train = os.path.join(data_dir, "Train")
    data_dir_val = os.path.join(data_dir, "Val")
    data_dir_test = os.path.join(data_dir, "Test")

    train_iterator = get_iterator(data_dir_train)
    val_iterator = get_iterator(data_dir_val)
    test_iterator = get_iterator(data_dir_test, shuffle=False) # dont shuffle in order to get proper actual/prediction pairs

    model_eff = EfficientNetB0(input_size, classes=1000, include_top=True, weights='imagenet')
    #remove last dense and last activation
    model_eff.layers.pop()
    model_eff.layers.pop()

    # add dense+activation
    x = model_eff.layers[-1].output
    output = Dense(194, activation="softmax")(x)
    model = Model(model_eff.inputs, [output])

    callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='max',restore_best_weights=True)
    lc_filename = "lc_clsf_EfficientNet.csv"
    callback_csv_logger = CSVLogger(lc_filename, separator=",", append=False)

    model_filename = "model_clsf_EfficientNet.h5"
    mcp_save = ModelCheckpoint(model_filename, save_best_only=True, monitor='val_accuracy', mode='max')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
              validation_data=val_iterator, validation_steps=len(val_iterator),
              callbacks=[callback_csv_logger, callback_earlystop, mcp_save])

    model = load_model(model_filename)
    print ("Eval test:")
    model.evaluate(test_iterator)
    print ("Eval val:")
    model.evaluate(val_iterator)

train_model(epochs=100)
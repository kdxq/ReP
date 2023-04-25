# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, schedules
from model.ConvNeXt import ConvNeXtAE_model
from tensorflow.python.keras import callbacks
import time
import os

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == '__main__':

    # Hyperparameter
    n_epoch = 500
    input_size = (1000, 40, 1)
    embed_dim = 128
    batch_size = 32
    filenm = 'ConvNeXtAE'
    initial_learning_rate = 0.0013
    loss_function = 'mae'
    pat = 15
    # ConvNeXt
    ConvNeXt_filters_array = (64, 32, 16, 8)
    block_layer = (1, 1, 2, 2)
    core_size = 7

    # load_data
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    print(train_data.shape, test_data.shape)

    # build model
    encoder, decoder, ae_model = ConvNeXtAE_model(size=input_size, filters_array=ConvNeXt_filters_array,block_layer=block_layer, core_size=core_size, embed_dim=embed_dim)
    encoder.summary()
    decoder.summary()
    ae_model.summary()
    ae_model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=loss_function)

    # Train the model

    log_dir = os.path.join('logs/' + str(time.time())[5:10])
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = callbacks.TensorBoard(log_dir=log_dir)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=pat,
                                             baseline=None, min_delta=0.00005, restore_best_weights=True)
    lr_decry = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1,
                                           mode='min', cooldown=0, min_lr=0.0001)


    weights_dir = os.path.join('weights/{}'.format(filenm) + '/')  # win10下的bug，
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    check_point = callbacks.ModelCheckpoint('weights//{}/'.format(filenm), monitor='val_loss', verbose=0,
                                                  save_best_only=True, save_weights_only=True, mode='min', period=1)

    history = ae_model.fit(train_data, train_data, epochs=n_epoch, batch_size=batch_size, verbose=1, shuffle=True,
                           callbacks=[tensorboard,check_point, early_stopping, lr_decry], validation_data=(test_data, test_data))
    print('train completed')
    # draw loss curve

    loss_dir = os.path.join('loss/{}'.format(filenm) + '/')  # win10下的bug，
    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)

    x = np.arange(len(history.history.get('loss')))
    best_result = []
    for i in range(1, len(x) + 1):
        best_result.append(min(history.history.get('val_loss')[:i]))
    plt.title('loss best result:{}'.format(min(history.history.get('val_loss'))))
    plt.plot(x, history.history.get('loss'), label='loss')
    plt.plot(x, history.history.get('val_loss'), label='val_loss')
    plt.plot(x, best_result, label='best_result')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.grid()
    plt.savefig(loss_dir + 'loss.png')
    plt.clf()
    np.savez(loss_dir + 'loss_data', loss=history.history.get('loss'), val_loss=history.history.get('val_loss'))

    # save weights

    ae_model.save_weights(weights_dir + 'ae_model.h5')
    encoder.save_weights(weights_dir + 'encoder.h5')
    decoder.save_weights(weights_dir + 'decoder.h5')

    img_coding_train = encoder.predict(train_data)
    img_coding_test = encoder.predict(test_data)
    np.save('img_coding_train', img_coding_train)
    np.save('img_coding_test', img_coding_test)


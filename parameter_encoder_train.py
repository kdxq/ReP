# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from model.MLP_Mixer import MLP_Mixer_encoder
import os

if __name__ == '__main__':
    filename = 'parameter_encoder'
    s = 128
    c = 13
    ds = 512
    dc = 52
    layer = 5
    epochs = 1500
    batch_size = 256
    loss_function = 'mae'

    img_coding_train = np.load('img_coding_train.npy')
    img_coding_test = np.load('img_coding_test.npy')
    img_coding_train = img_coding_train.reshape(-1,c,s)
    img_coding_test = img_coding_test.reshape(-1,c,s)
    img_coding_train = np.transpose(img_coding_train,(0,2,1))
    img_coding_test = np.transpose(img_coding_test,(0,2,1))

    parameter_train = np.load('parameter_train.npy')
    parameter_test = np.load('parameter_test.npy')

    model = MLP_Mixer_encoder(s=s,c=c,ds=ds,dc=dc,layer=layer)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.005), loss=loss_function)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=300,
                                             baseline=None, min_delta=0.0001,restore_best_weights=True)
    lr_decry = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, verbose=1,
                                           mode='min', epsilon=0.0001, cooldown=0, min_lr=0.0001)

    history = model.fit(parameter_train, img_coding_train, batch_size=batch_size, epochs=epochs,
                    callbacks=[early_stopping,lr_decry],shuffle=True,
                    validation_data=(parameter_test, img_coding_test))

    loss_dir = os.path.join('weights/{}/'.format(filename))
    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)

    model.save('{}//parameter_encoder.h5'.format(loss_dir))
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

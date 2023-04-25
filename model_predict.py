from model_train import ResNet_model,ConvNetAE_model,SwinAE_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from label_encoder import label_encoder_model
from mpl_toolkits import mplot3d

if  __name__ == '__main__':

    ConvNeXt_filters_array = (64, 32, 16, 8)
    embed_dim = 128
    block_layer = (1, 1, 2, 2)
    core_size = 7
    encoder, decoder, ae_model = ConvNetAE_model(size=(1000,40,1), filters_array=ConvNeXt_filters_array,
                                                 block_layer=block_layer, core_size=core_size, embed_dim=embed_dim)
    decoder.load_weights('weights\\{}\\decoder.h5'.format('ConvNeXtAE-2'))
    ae_model.load_weights('weights\\{}\\ae_model.h5'.format('ConvNeXtAE-2'))

    label_encoder = label_encoder_model()
    label_encoder.load_weights('label_encoder.h5')

    parameter_train = np.load('parameter_train.npy')
    parameter_test = np.load('parameter_test.npy')
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    test_data = np.transpose(test_data,(0,2,1))
    test_data = test_data[:,:39,:]
    #test_image = decoder.predict(label_encoder.predict(parameter_test))
    z = []
    for i in range(20):
        z.append([0, 0.75+i*0.05, 155, 84, 54, 60, 52, 86, 86, 59])
    divide = np.array([1,1.65,220,180,160,160,160,160,160,160])
    z = np.array(z)
    z = z/divide
    z = label_encoder.predict(z)
    z = tf.reshape(z,(20,128,1))
    z = tf.image.resize(z,(128,128))
    z = tf.reshape(z, (128, 128))
    fig = plt.Figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('increase')
    x = y = np.arange(128)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x,y,z,cmap='rainbow')
    ax.contourf(x, y, z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))

    plt.show()

def show(n):
    test_image = decoder.predict(label_encoder.predict(parameter_test[n].reshape(1,10))).reshape(1000,40)
    test_image = np.transpose(test_image,(1,0))
    test_image = test_image[:39,:]
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(test_data[n],cmap='jet')
    ax[1].imshow(test_image, cmap='jet')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('real',c='b')
    ax[1].set_title('predict',c='r')
    fig.show()


from tensorflow.python.keras.layers import Dense, LayerNormalization, Reshape, Conv2D, Conv2DTranspose, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
import tensorflow as tf
from tensorflow.python.keras.activations import gelu


class Mixer_layer(layers.Layer):
    def __init__(self, s, c, ds, dc):
        super(Mixer_layer, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6, name="norm1")
        self.MLP1_1 = Dense(ds, name="MLP1_1")
        self.MLP1_2 = Dense(s, name="MLP1_2")
        self.norm2 = LayerNormalization(epsilon=1e-6, name="norm2")
        self.MLP2_1 = Dense(dc, name="MLP2_1")
        self.MLP2_2 = Dense(c, name="MLP2_2")

    def call(self, x, *args, **kwargs):
        shortcut = x
        x = tf.transpose(self.norm1(x), perm=(0, 2, 1))
        x = self.MLP1_1(x)
        x = gelu(x)
        x = self.MLP1_2(x)
        x = self.norm2(tf.transpose(x, perm=(0, 2, 1)))
        x = tf.add(x, shortcut)
        shortcut = x
        x = self.MLP2_1(x)
        x = gelu(x)
        x = self.MLP2_2(x)
        x = tf.add(x, shortcut)
        return x


class MLP_Mixer(Model):

    def __init__(self ,s = 128,c = 13 ,DS=512, DC=52, layer=5):
        super(MLP_Mixer, self).__init__()
        self.mixer_layers = []
        for i in range(layer):
            self.mixer_layers.append(Mixer_layer(s=s,c=c,ds=DS,dc=DC))

    def call(self, x, training=None):
        for layer in self.mixer_layers:
            x = layer(x,training=training)
        return x

def MLP_Mixer_encoder(embed_dim=128,s=128,c=13,ds=128,dc=52,layer=5):

    Mixer = MLP_Mixer(s=s,c=c,DS=ds,DC=dc,layer=layer)

    input = Input(shape=(9, 1), name='3_d_label')
    x = Dense(c)(input)
    x = tf.transpose(x, (0, 2, 1))
    x = Dense(embed_dim)(x)
    x = tf.transpose(x, (0, 2, 1))
    output = Mixer(x,training=True)

    model = Model(input, output, name='label_encoder_model')

    return model

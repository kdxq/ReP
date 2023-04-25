from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Reshape, Flatten, LayerNormalization, Conv2DTranspose
from tensorflow.keras.activations import gelu

class Block(layers.Layer):
    def __init__(self, filters=32, core_size=7, name=None):
        super().__init__(name=name)
        self.block = [
            Conv2D(filters, (core_size, core_size), padding='same'),
            LayerNormalization(),
            Conv2D(filters * 4, (1, 1), activation='gelu', padding='same'),
            Conv2D(filters, (1, 1), padding='same'),
        ]

    def call(self, x, training=None, *args, **kwargs):
        shortcut = x
        for layers in self.block:
            x = layers(x, training=training)
        x += shortcut
        return x


class ConvNeXt(Model):

    def __init__(self, core_size=7, filters_array=(32, 16, 8, 4), block_layer=(1, 1, 1, 1), up_down='down', name=None,
                 **kwargs):
        super().__init__(name=name)
        self.conv_block = []
        for i in range(len(filters_array)):
            for j in range(block_layer[i]):
                self.conv_block.append(Block(filters=filters_array[i], core_size=core_size))
            if i != len(filters_array) - 1:
                if up_down == 'down':
                    self.conv_block.append(
                        Conv2D(filters_array[i + 1], (2,2), padding='same', strides=2, activation='gelu'))
                else:
                    self.conv_block.append(
                        Conv2DTranspose(filters_array[i + 1], (2,2), padding='same', strides=2, activation='gelu'))

    def call(self, x, training=None, mask=None):
        for block in self.conv_block:
            x = block(x, training=training)
        return x

def ConvNeXtAE_model(size=(200, 200, 1), filters_array=(3, 6, 12, 24), block_layer=(1, 1, 1, 1), core_size=3,
                    embed_dim=64):
    activation = 'gelu'
    reduce_size = (int(size[0] / 2 ** (len(filters_array) - 1)), int(size[1] / 2 ** (len(filters_array) - 1)))
    down_model = ConvNeXt(filters_array=filters_array, block_layer=block_layer, core_size=core_size, up_down='down')
    down_model.build((1, size[0], size[1], filters_array[0]))
    f2 = filters_array[::-1]
    up_model = ConvNeXt(filters_array=f2, block_layer=block_layer[::-1], core_size=core_size,
                        up_down='up')
    up_model.build((1, reduce_size[0], reduce_size[1], f2[0]))

    # encoder
    img_input = Input(shape=(size[0], size[1], size[2]), name='input_img')

    x = Conv2D(filters_array[0], (1, 1), activation=activation, padding='same')(img_input)
    x = Conv2D(filters_array[0], (3, 3), activation=activation, padding='same')(x)
    x = down_model(x, training=True)
    x = Flatten()(x)
    x = Dense(embed_dim*2)(x)
    x = LayerNormalization()(x)
    x = gelu(x)
    x = Dense(embed_dim)(x)
    x = LayerNormalization()(x)
    encoder_output = gelu(x)

    encoder = Model(img_input, encoder_output, name='encoder')

    # decoder
    decoder_input = Input(shape=(embed_dim), name='encoded_img')
    x = Dense(embed_dim*2)(decoder_input)
    x = LayerNormalization()(x)
    x = gelu(x)
    x = Dense(reduce_size[0] * reduce_size[1] * f2[0], activation=activation)(x)
    x = Reshape((reduce_size[0], reduce_size[1], -1))(x)
    x = up_model(x, training=True)
    x = Conv2D(f2[-1], (3, 3), padding='same', activation=activation)(x)
    decoder_output = Conv2D(size[2], (1, 1), activation='linear', padding='same')(x)

    decoder = Model(decoder_input, decoder_output, name='decoder')

    # ae_model
    ae_model_input = Input(shape=(size[0], size[1], size[2]))
    encoder_img = encoder(ae_model_input)
    ae_model_output = decoder(encoder_img)

    ae_model = Model(ae_model_input, ae_model_output, name='ConvNeXtAE')

    return encoder, decoder, ae_model
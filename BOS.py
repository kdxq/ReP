from bayes_opt import BayesianOptimization
import numpy as np
from model.ConvNeXt import ConvNeXtAE_model
from model.MLP_Mixer import MLP_Mixer_encoder
import time

ConvNeXt_filters_array = (64, 32, 16, 8)
embed_dim = 128
block_layer = (1, 1, 2, 2)
core_size = 7

s = 128
c = 13
ds = 512
dc = 52
layer = 5

encoder, decoder, ae_model = ConvNeXtAE_model(size=(1000,40,1), filters_array=ConvNeXt_filters_array,block_layer=block_layer, core_size=core_size, embed_dim=embed_dim)
decoder.load_weights('decoder.h5')
ae_model.load_weights('ae_model.h5')

parameter_encoder_model = MLP_Mixer_encoder(s=s,c=c,ds=ds,dc=dc,layer=layer)
parameter_encoder_model.load_weights('parameter_encoder.h5')


input_divide_part = np.array([1.65,220,180,160,160,160,160,160,160])

def turn_lq(img):
    data = img
    data = data*(1811-604) + 604
    data = np.where(data <= 1715, 0, data)
    data = np.where(data>=1786,1,data)
    data = np.where((data>1715)&(data<1786) , (data-1715)/(1786-1715), data)
    data = np.where(data>0.95,1,0)
    return data

def get_point(tuple):
    origin = np.array(tuple)
    origin = origin / input_divide_part
    origin = origin.reshape(1, 9, 1)
    original_img = parameter_encoder_model.predict(origin)
    original_img = np.transpose(original_img, (2, 1, 0))
    original_img = original_img.reshape(13, 128)
    original_img = decoder.predict(original_img)
    original_img = np.transpose(original_img, (1, 2, 0, 3))
    original_img = turn_lq(original_img)
    point = list(original_img[:, 0, 0])[::-1].index(1)

    return point

tuple1 = (1.3, 102, 87, 89, 85, 70, 30, 77, 49)
original_point = get_point(tuple1)

tuple2 = (1.37, 102, 87, 89, 85, 70, 30, 77, 49)
mutation_point = get_point(tuple2)

time1 = time.time()
def target(a,b):

    new = np.array([1.37, int(a * 220), int(b * 180), 89, 85, 70, 30, 77, 49])
    new_point = get_point(new)
    val = -abs(new_point-original_point)
    print(-val,time.time()-time1)

    return val


rf_bo = BayesianOptimization(
        target,
        {'a': (0, 1),
        'b': (0, 1),
         })

rf_bo.maximize(n_iter=50)
x = rf_bo.max
print(x)
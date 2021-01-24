import keras
import numpy as np
from os.path import join
import matplotlib.pyplot as plt


json_filename = 'model.json'
h5_filename   = 'weights.h5'



def load_tfl_data(data_id, crop_shape=(81,81)):
    images = np.memmap(join('bins/','data{}.bin'.format(data_id)),mode='r',dtype=np.uint8).reshape([-1]+list(crop_shape) +[3])
    #labels = np.memmap(join(data_dir,'labels.bin'),mode='r',dtype=np.uint8)
    return {'images':images}


#root = '/content/drive/My Drive/mobile_proj/part02'  #this is the root for your val and train datasets
#datasets = {
 #   'val':load_tfl_data(join(root,'val')),
  #  'train': load_tfl_data(join(root,'train')),
#}



def load():
    with open(json_filename, 'r') as j:
        loaded_json = j.read()
        # load the model architecture: 
        loaded_model = keras.models.model_from_json(loaded_json)
        #load the weights:
        loaded_model.load_weights(h5_filename)
        print(" ".join(["Model loaded from", json_filename, h5_filename]))
    return loaded_model


loaded_model = load()


def get_prediction( val ):
    #  val['images']
    val = load_tfl_data(val)
    l_predictions = loaded_model.predict( val['images'] ) 
    # sbn.distplot(l_predictions[:,0])
    l_predicted_label = np.argmax(l_predictions, axis=-1)
    # print ('accuracy:', np.mean(l_predicted_label==val['labels']))
    # for i, img in enumerate( val["images"] ):
    #     plt.figure()
    #     plt.imshow( img ) 
    #     plt.title( str( l_predicted_label[i] ) )
    #     plt.show()
    return l_predicted_label

if __name__ == "__main__":
    # get_prediction( )
    pass
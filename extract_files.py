from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input

from glob import glob
from itertools import cycle,zip_longest
import PIL
from PIL import Image
import matplotlib.pyplot as plt

path = "./"
image_size=(224,224)
img_height = 224
img_width = 224
total_image_num = 206949
#total_image_num = 100000
batch_size = 100

#define model

input_tensor = Input(shape=(224, 224, 3))
model = VGG19(input_tensor=input_tensor,weights='imagenet',include_top=False)
vgg_output = model(input_tensor)
shape = (7,7,512)
pooling = GlobalAveragePooling2D(input_shape=shape)(vgg_output)
comb_model = Model(inputs=input_tensor, outputs=pooling)

def grouper(n, iterable, fillvalue=None):
  args = [iter(iterable)]*n
  return zip_longest(*args, fillvalue=fillvalue)

def get_images(batch_size=10):
  height = img_height
  width = img_width
  #data_dir = "/Users/sashatn/Desktop/Cambridge/Michaelmas/Probabilistic Machine Learning (LE48)/Project/test/"
  data_dir = "/home/ubuntu/project/le49/photos/"
  input_files = glob(data_dir + "*.jpg")
  #print(input_files)
  input_files_infinite = cycle(input_files)
  input_files_grouped = grouper(batch_size,input_files_infinite)
  while 1:
    
    image_names = next(input_files_grouped)
    img = [image.load_img(fname, target_size=(224, 224)) for fname in image_names]
    img_array = [image.img_to_array(image_val) for image_val in img]
    img_array_expand = [np.expand_dims(image, axis=0) for image in img_array]
    image_files = [preprocess_input(image) for image in img_array_expand]
    
    yield zip(np.array(image_files),list(image_names))
    
generator = get_images(batch_size)
all_names = [] 
all_predictions = []
num_batches = int(total_image_num/batch_size)
for i in range(num_batches):
    imgs,names = map(list, zip(*next(generator))) 
    imgs = np.squeeze(imgs,axis=1)
    print("Processed batch: %s" % (i))
    all_predictions.append(comb_model.predict(imgs))
    print("Predicted batch: %s" % (i))
    all_names.append(names)
print("Started Conversion To Array")
all_predictions = np.asarray(all_predictions)
all_names = np.asarray(all_names)
print("Started Reshape To Correct Size")
all_predictions = np.reshape(all_predictions,(num_batches*batch_size,512))
all_names = np.reshape(all_names,(num_batches*batch_size))

print("Started Save")
np.save('predictions.npy', all_predictions) 
np.save('names.npy', all_names)
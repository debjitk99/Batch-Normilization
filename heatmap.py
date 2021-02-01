
import PIL
print('Pillow Version:', PIL.__version__)
# load and show an image with Pillow
from PIL import Image
# Open the image form working directory
image = Image.open('0b36c5a2-6c9f-40d9-af4b-d4b0e66997da___Matt.S_CG 6653.JPG')
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)

from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
image = image.imread('0b36c5a2-6c9f-40d9-af4b-d4b0e66997da___Matt.S_CG 6653.JPG')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()
# show the image


from PIL import Image
from numpy import asarray
import numpy
# load the image
image = Image.open('0b36c5a2-6c9f-40d9-af4b-d4b0e66997da___Matt.S_CG 6653.JPG')
# convert image to numpy array
data = asarray(image)
print(type(data))
# summarize shape
print(data.shape)
print(data[0].shape)
data_got=numpy.squeeze(data[0])
print(data_got.shape)
# create Pillow image
#image2 = Image.fromarray(data)
#print(type(image2))

# summarize image details
#print(image.mode)
#print(image.size)
print(data)
import seaborn as sns
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(12,10))
heat_map=sns.heatmap(data_got)
plt.show()    

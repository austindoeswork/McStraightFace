from PIL import Image
import numpy as np

def showface(f):
    img1 = Image.fromarray(f)
    img1.show()

def facearrtomatrix(np_arr):
    f = np_arr.reshape(64,64)
    f = f.T

    return f

faceclass = np.genfromtxt('../resources/faceclass.csv', delimiter=',')
faceclass = faceclass[1:]

facedata = np.genfromtxt('../resources/facedata.csv', delimiter=',')
faceindex = facedata[:,0]
facedata = facedata[1:]
facedata = facedata[:,1:]

for i in range(0,20):
    f = facearrtomatrix(facedata[i])
    showface(f)

#  w, h = 512, 512
#  data = np.zeros((h, w, 3), dtype=np.uint8)
#  data[256, 256] = [255, 0, 0]
#  img = Image.fromarray(data, 'RGB')
#  img.show()

#  img.save('my.png')

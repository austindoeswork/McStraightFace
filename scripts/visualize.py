from PIL import Image
import numpy as np

def showface(f, features):
    fscaled = f*255

    w, h = 64, 64
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            data[i, j] = [0, fscaled[i,j], 0]

    featscaled = features * 255
    for i in range(64):
        for j in range(64):
            if featscaled[i,j] > 0.0:
                data[i, j] = [featscaled[i,j], 0, featscaled[i,j]]

    img = Image.fromarray(data, 'RGB')
    img.show()

def facearrtomatrix(np_arr):
    f = np_arr.reshape(64,64)
    f = f.T

    return f

def normalize(np_arr):
    maxval = 0.0
    for i in range(0, np.shape(np_arr)[0]):
        if np_arr[i] > maxval:
            maxval = np_arr[i]
    for i in range(0, np.shape(np_arr)[0]):
        np_arr[i] = np_arr[i] / maxval

def getaverage(f, row, col, radius):
    total = 0.0
    count = 0.0
    for i in range(row-radius, row+radius + 1):
        for j in range(col-radius, col+radius + 1):
            total += f[i,j]
            count += 1.0
    return total / count

def main():
    faceclass = np.genfromtxt('../resources/faceclass.csv', delimiter=',')
    faceclass = faceclass[1:]

    facedata = np.genfromtxt('../resources/facedata.csv', delimiter=',')
    facedata = facedata[1:]
    faceindex = facedata[:,0]
    facedata = facedata[:,1:]

    print "FACEDATA :", np.shape(facedata)
    print "FACEINDEX:", np.shape(faceindex)
    print "FACECLASS:", np.shape(faceclass)

    # NORMALIZE THE DATA
    for i in range(0,np.shape(facedata)[0]):
        normalize(facedata[i])

    f = facearrtomatrix(facedata[0])
    showface(f)


if __name__ == "__main__":
    main()



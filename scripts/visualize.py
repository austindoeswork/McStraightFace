from PIL import Image
import numpy as np

def showface(f, features = np.zeros((64, 64), dtype=np.uint8)):
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

def convolute(f, method="boxblur"):
    if method == "boxblur":
        cmatrix = np.ones((3,3), np.float64)/9
        buf = 1
        print cmatrix 
    if method == "edge":
        cmatrix = np.matrix([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]])
        #  cmatrix = np.matrix([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
        buf = 1
        print cmatrix 
    fcopy = f.copy()
    for i in range(buf, 64-buf):
        for j in range(buf, 64-buf):
            fcopy[i,j] = np.dot(cmatrix, f[i-buf:i+buf+1,j-buf:j+buf+1])[1,1]
    return fcopy

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
    fconv = convolute(f, "edge")
    showface(fconv)
    


if __name__ == "__main__":
    main()

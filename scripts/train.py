from PIL import Image
import numpy as np
import math

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
    img = img.resize((1024,1024))
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

def main():
    faceclass = np.genfromtxt('../resources/faceclass.csv', delimiter=',')
    faceclass = faceclass[1:]

    facedata = np.genfromtxt('../resources/facedata.csv', delimiter=',')
    facedata = facedata[1:]

    faceindex = facedata[:,0] # get first col
    facedata = facedata[:,1:] # remove first col

    print "FACEDATA :", np.shape(facedata)
    print "FACEINDEX:", np.shape(faceindex)
    print "FACECLASS:", np.shape(faceclass)

    # NORMALIZE THE DATA
    for i in range(0,np.shape(facedata)[0]):
        normalize(facedata[i])

    #### BEGIN THE REGRESSOR TRAINING
    I = np.zeros((80, 64, 64))
    S = np.zeros((80, 21, 2))
    SFeat = np.zeros((80, 64, 64))
    T_Pairs = []

    # Init Truth
    index = 0
    for i in range(400):
        if i % 10 == 0 or i % 10 == 1:
            marks = np.genfromtxt("./marks/" + str(i) + ".csv", delimiter=',')
            featmarks = np.zeros((64,64), np.uint8)
            for j in range(21):
                featmarks[int(marks[j,1]),int(marks[j,0])] = 1
            f = facearrtomatrix(facedata[i])
            #  showface(f, featmarks)
            I[index] = f
            S[index] = marks
            SFeat[index] = featmarks
            index += 1
    print "T_PAIRS:", index

    # Begin training
    n = index   # # of training examples
    R = 1       # inits per example 
    N = n * R   # total number of triplets
    T = 10      # # of regressors
    v = 0.9     # shrinkage factor

    Triplets = []
    for i in range(2):
        potentials = range(i) + range(i+1, n)
        choices = np.random.choice(potentials, R, replace=False)
        pi_i  = i
        for choice in choices:    
            Ipi_i = I[pi_i]
            Spi_i = S[pi_i]
            Shat  = S[choice]
            DeltaS = Spi_i - Shat
            Triplets.append((Ipi_i, Shat, DeltaS))


if __name__ == "__main__":
    main()

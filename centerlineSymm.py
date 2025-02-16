#Wilson Gregory
#Final project for Intro to Data Math

import csv
import numpy as np
import scipy
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import random
from functools import reduce

#const that specifies row (and col) size
SIZE = 64

#function that converts a row face to a 64x64 matrix, with the right orientation
def facearrtomatrix(np_arr):
    f = np_arr.reshape(64,64)
    f = f.T #transpose the face
    return f

#converts the 64x64 matrix back into the face row
def matrixtofacearr(f):
    f = f.T
    face = f.reshape(1,4096)
    return face

def showface(f, features = np.zeros((64, 64), dtype=np.uint8)):
    fscaled = f*255

    w, h = 64, 64
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0,64):
        for j in range(0,64):
            data[i, j] = [0, fscaled[i,j], 0]

    featscaled = features * 255
    for i in range(0,64):
        for j in range(0,64):
            if featscaled[i,j] > 0.0:
                data[i, j] = [featscaled[i,j], 0, featscaled[i,j]]

    img = Image.fromarray(data, 'RGB')
    img.show()
#end func

def convolute(f, method="boxblur"):
    if method == "boxblur":
        cmatrix = np.ones((3,3), np.float64)/9
        buf = 1
        #  print cmatrix 
    if method == "edge":
        cmatrix = np.matrix([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]])
        #  cmatrix = np.matrix([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
        buf = 1
        #print(cmatrix) 
    #  fcopy = f.copy()
    fcopy = np.zeros_like(f)
    size = np.shape(f)[0]
    for i in range(buf, size-buf):
        for j in range(buf, size-buf):
            #  fcopy[i,j] = np.dot(cmatrix, f[i-buf:i+buf+1,j-buf:j+buf+1])[1,1]
            val = np.dot(cmatrix, f[i-buf:i+buf+1,j-buf:j+buf+1])[1,1]
            #  print val
            if val > -0.8:
            #  if val < -1.5:
                fcopy[i,j] = val
    return fcopy

def writeCSV(data,labelsRow,filename):
    length = len(data)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(labelsRow)
        for i in range(0,length):
            writer.writerow(data[i])
#end func

def transposeFace(face):
    for i in range(0,SIZE):
        for j in range(0,i):
            temp = face[i*SIZE + j]
            face[i*SIZE + j] = face[j*SIZE + i]
            face[j*SIZE + i] = temp


#takes in one row and the centerline to try, returns number of pixels that are similar
def calcSymm(faceRow,centerline,epsilon):
    stop = min(centerline, SIZE-centerline)
    count = 0
    #print(faceRow)
    #e.g., 32 will compare the entire row for similarity
    for i in range(0,stop):
        diff = abs(faceRow[centerline-i-1] - faceRow[centerline+i])
        if((diff - epsilon) <= 0):
            count += 1
        #else:
            #faceRow[centerline-i-1] = faceRow[centerline+i] = 0

    return count,faceRow
#end func

#normalize function that normalizes the whole face
def normalize(np_arr):
    maxval = 0.0
    for i in range(0, np.shape(np_arr)[0]):
        if np_arr[i] > maxval:
            maxval = np_arr[i]
    for i in range(0, np.shape(np_arr)[0]):
        np_arr[i] = np_arr[i] / maxval

#takes in the face, and the centerline that we are normalizing each half
def normalizeFace(face,centerline,fconv,mask):
    
    colsums = np.sum(fconv,axis=0) #sum the columns into a row array
    leftsum = np.sum(colsums[0:centerline])
    rightsum = np.sum(colsums[centerline:SIZE])

    leftsum = float(leftsum) / (centerline*SIZE)
    rightsum = float(rightsum) / ((SIZE-centerline)*SIZE)

    normFace = np.zeros((SIZE,SIZE), dtype=np.int)

    for i in range(0,SIZE):
        for j in range(0,centerline):
            if(mask[i,j] != -1): #do i want to be doing this?
                normFace[i,j] = fconv[i,j] - leftsum

    for i in range(0,SIZE):
        for j in range(centerline,SIZE):
            if(mask[i,j] != -1):
                normFace[i,j] = fconv[i,j] - rightsum

    return normFace


#func. don't need to be calling this every normalize, only need once per face
def edgeMask(face):
    fconv = convolute(face,"edge")
    newf = face.copy()
    mask = np.zeros((SIZE,SIZE),dtype=np.int)
    for i in range(0,SIZE):
        for j in range(0,SIZE):
            if(fconv[i,j] > 0.0):
                newf[i,j] = 0
                mask[i,j] = -1
    return newf, mask
#end func

#updated func
def findCenterline(face,epsilon,start):

    maxSymm = 0
    maxSymmCenter = 32
    #bestFace
    fconv,edgemask = edgeMask(face)
    
    for i in range(start,SIZE-start): #will iterate over all possible centerlines
    #for i in range(32,33):
        totSymm = 0
        normFace = normalizeFace(face,i,fconv,edgemask)
        for j in range(0,SIZE): #j is the row
            count, faceRow = calcSymm(normFace[j],i,epsilon)
            totSymm += count     
        #end inner for loop
        
        #symmPercent = float(totSymm) / (min(i,SIZE-i)*2*SIZE)
        symmPercent = float(totSymm) / (64*64) #divides by total number of pixels
        if(symmPercent > maxSymm):
            maxSymm = symmPercent
            maxSymmCenter = i
            bestFace = normFace

    #update the graph with the best centerline
    #for j in range(0,SIZE):
    #    count, faceRow = calcSymm(face[j],maxSymmCenter,epsilon)
    #    face[j] = faceRow
    
    #for i in range(0,SIZE):
    #    face[i] = bestFace[i] #should correctly update
    
    return (abs((SIZE/2)-maxSymmCenter)), maxSymm
#end func


#function that runs the SVM on the data, with the classes. Returns the score
def runSVM(traindata,trainclasses,testdata,testclasses):
    SVM = LinearSVC(dual=False)
    SVM.fit(traindata,trainclasses)

    return SVM.score(traindata,trainclasses), SVM.score(testdata,testclasses)


#function that gets a random sample from the face data and puts it in a test set
def sample(facedata, testSize, randList):
    testSize = int(testSize)
    facetrain = []
    facetest = []
    randListIndex = 0
    for i in range(0,len(facedata)):
        if(randListIndex < testSize and i == randList[randListIndex]):
            facetest.append(facedata[i])
            randListIndex += 1
        else:
            facetrain.append(facedata[i])
    return facetrain, facetest
#end func


#Runs PCA on facedata and returns the data projected on the first numvecs eigenvectors
def runPCA(facedata,testdata,numvecs):

    data = []
    #for i in range(0,len(facedata)):
    #print(np.shape(facedata[i]))
        
    data = list(map((lambda x: x.reshape(1,4096)),facedata)) #what
    data_arr = reduce((lambda x, y: np.concatenate((x,y),axis=0)), data)
    
    #print(data_arr)
    
    pca = PCA(n_components=numvecs)
    pca.fit(data_arr)
    results = pca.transform(data_arr)

    data = list(map((lambda x: x.reshape(1,4096)),testdata)) #what
    data_arr = reduce((lambda x, y: np.concatenate((x,y),axis=0)), data)
    testresults = pca.transform(data_arr)
    #print(np.shape(results),flush=True)
    return results,testresults
#end func


#MAIN
print("Entering Main:")


#mustards code. Imports data
faceclass = np.genfromtxt('faceclass.csv', delimiter=',')
faceclass = faceclass[1:] #trim the first row of faceclass

facedata = np.genfromtxt('facedata.csv', delimiter=',')
labelsRow = facedata[1]
facedata = facedata[1:] #trim the first row
faceindex = facedata[:,0]
facedata = facedata[:,1:] #trim the first element from each row of allFaces

print("FACEDATA :")
print(np.shape(facedata))

epsilon = 13 #tunable, 13 was best on trainError, should recheck on test error (ECV)
start = 20 #tunable 21 is best, might use 20 to be safe
numpca = 10
runs = 1

allFaces = []
for i in range(0,len(facedata)):
    #transposeFace(allFaces[i])
    #fconv = convolute(f, "edge")
    #allFaces.append(convolute(facearrtomatrix(facedata[i]),"edge")) #convolute each face
    allFaces.append(facearrtomatrix(facedata[i])) #convolute each face

minError = len(allFaces)
minIndex = 0
testSize = len(allFaces) * 0.1
totTestError = 0

for q in range(0,runs):
    trainfeatures = [] #reset features
    testfeatures = []
    #epsilon = 5+q
    #start = 10+q
    randList = random.sample(range(len(facedata)),int(testSize))
    randList.sort()

    facetrain, facetest = sample(allFaces, testSize,randList)
    classtrain, classtest = sample(faceclass,testSize,randList)

    trainVecs, testVecs = runPCA(facetrain,facetest,numpca)
    #print(np.shape(trainVecs[0]))
    for i in range(0,len(facetrain)): #loop on each face
        sim, dist = findCenterline(facetrain[i],epsilon,start )
        featlist = [sim,dist]
        for j in range(0,numpca):
            featlist.append(trainVecs[i,j])
        trainfeatures.append(featlist)
    #end inner for loop

    for i in range(0,len(facetest)): #loop on each face
        sim, dist = findCenterline(facetest[i],epsilon,start )
        featlist = [sim,dist]
        for j in range(0,numpca):
            featlist.append(testVecs[i,j])
        testfeatures.append(featlist)
    #end inner for loop

    trainscore,testscore = runSVM(trainfeatures,classtrain,testfeatures,classtest)
    testerror = 1-testscore
    trainerror = 1-trainscore

    print("Train Error: " + str(trainerror))
    print("Test Error: " + str(testerror), flush=True)
    if(testerror < minError):
        minError = testerror
        minIndex = epsilon
    totTestError += testerror
    
#end for loop

print("Average test error: " + str(float(totTestError) / runs))
for i in range(10,20):
    normalize(facedata[i]) #do i need to normalize first? yes
    fconv = convolute(facearrtomatrix(facedata[i]),"edge")
    #print(fconv)
    showface(fconv)

#convolution "edge" has a lot of 0s



    
#used for visualizing in R
#for i in range(0,len(allFaces)):
#    facedata[i] = matrixtofacearr(allFaces[i])

#writeCSV(allFaces,labelsRow,'faces.csv')
#writeCSV(features,['similarity','distance'],'features.csv')

#might be better if we don't take bad pixels into the account when normalizing


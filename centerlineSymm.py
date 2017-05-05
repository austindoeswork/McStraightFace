#Wilson Gregory
#Final project for Intro to Data Math

import csv
import numpy as np
import scipy
from PIL import Image
from sklearn.svm import LinearSVC

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
#end func

#funtion that parses input file into a list of lists. Returns the list of the labels row
def parseInput(allFaces,filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        labelsRow = next(reader) #first row, the column labels
        for row in reader:
                row = list(map(float,row)) #convert strings to floats in the list
                row = list(map(int, row)) #convert floats to ints, they should be
                allFaces.append(row)
        return labelsRow
#end func

#separate function to parse the faceclass.csv file
def parseClassInput(allFaces,filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        labelsRow = next(reader) #first row, the column labels
        for row in reader:
                row = list(map(float,row)) #convert strings to floats in the list
                row = list(map(int, row)) #convert floats to ints, they should be
                allFaces.append(row)
        return labelsRow
#end func

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



#takes in the face, and the centerline that we are normalizing each half
def normalizeFace(face,centerline):
    colsums = np.sum(face,axis=0) #sum the columns into a row array
    leftsum = np.sum(colsums[0:centerline])
    rightsum = np.sum(colsums[centerline:SIZE])

    leftsum = float(leftsum) / (centerline*SIZE)
    rightsum = float(rightsum) / ((SIZE-centerline)*SIZE)

    for i in range(0,SIZE):
        for j in range(0,centerline):
            face[i,j] = face[i,j] - leftsum

    for i in range(0,SIZE):
        for j in range(centerline,SIZE):
            face[i,j] = face[i,j] - rightsum
    

#updated func
def findCenterline(face,epsilon,start):

    maxSymm = 0
    maxSymmCenter = 32

    
    for i in range(start,SIZE-start): #will iterate over all possible centerlines
    #for i in range(32,33):
        totSymm = 0
        normalizeFace(face,i)
        for j in range(0,SIZE): #j is the row
            count, faceRow = calcSymm(face[j],i,epsilon)
            totSymm += count     
        #end inner for loop
        
        #symmPercent = float(totSymm) / (min(i,SIZE-i)*2*SIZE)
        symmPercent = float(totSymm) / (64*64) #divides by total number of pixels
        if(symmPercent > maxSymm):
            maxSymm = symmPercent
            maxSymmCenter = i

    #update the graph with the best centerline
    #for j in range(0,SIZE):
    #    count, faceRow = calcSymm(face[j],maxSymmCenter,epsilon)
    #    face[j] = faceRow
        
    return (abs((SIZE/2)-maxSymmCenter)), maxSymm
#end func


#function that runs the SVM on the data, with the classes. Returns the score
def runSVM(data,classes):
    SVM = LinearSVC(dual=False)
    SVM.fit(features,classes)
    predictions = SVM.predict(features)

    return SVM.score(features,classes)


#MAIN
print("Entering Main:")


#mustards code
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

allFaces = []
for i in range(0,len(facedata)):
    #transposeFace(allFaces[i])
    allFaces.append(facearrtomatrix(facedata[i]))

minError = len(allFaces)
minIndex = 0

#print(len(allFaces))
#print(np.shape(allFaces[0]))

for q in range(0,1):
    features = [] #reset features
    #epsilon = 5+q
    #start = 10+q
    for i in range(0,len(allFaces)): #loop on each face
        sim, dist = findCenterline(allFaces[i],epsilon,start )
        features.append([sim,dist])
    #end inner for loop

    score = runSVM(features,faceclass)
    error = 1-score

    print("Train Error: " + str(error) +" at epsilon "+str(epsilon),
          flush=True)
    if(error < minError):
        minError = error
        minIndex = epsilon
#end for loop

print("Best train error: " + str(minError) + " at epsilon " + str(minIndex))
for i in range(0,10):
    showface(allFaces[i])

#used for visualizing in R
#for i in range(0,len(allFaces)):
#    facedata[i] = matrixtofacearr(allFaces[i])

#writeCSV(allFaces,labelsRow,'faces.csv')
#writeCSV(features,['similarity','distance'],'features.csv')




#Wilson Gregory
#Final project for Intro to Data Math

import csv
import numpy
import scipy
from sklearn.svm import LinearSVC

#const that specifies row (and col) size
SIZE = 64 

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
    #print("Row length %d" % len(faceRow))
    stop = min(centerline, SIZE-centerline)
    count = 0

    #e.g., 32 will compare the entire row for similarity
    for i in range(0,stop):
        diff = abs(faceRow[centerline-i-1] - faceRow[centerline+i])
        if((diff - epsilon) <= 0):
            count += 1
        else:
            faceRow[centerline-i-1] = faceRow[centerline+i] = 0

    return count,faceRow
#end func

#takes in a list of integers that represents a face. finds the minimizing centerline
def findCenterline(face,epsilon,start):

    maxSymm = 0
    maxSymmCenter = 32
    
    for i in range(start,SIZE-start): #will iterate over all possible centerlines
        totSymm = 0
        for j in range(0,SIZE):
            count, faceRow = calcSymm(face[(SIZE*j):(SIZE*(j+1))],i,epsilon)
            totSymm += count                
        #end inner for loop
        
        #symmPercent = float(totSymm) / (min(i,SIZE-i)*2*SIZE)
        symmPercent = float(totSymm) / (64*64) #divides by total number of pixels
        if(symmPercent > maxSymm):
            maxSymm = symmPercent
            maxSymmCenter = i
    #print("Best centerline at " + str(maxSymmCenter) + " with " + str(maxSymm))

    #update the graph with the best centerline
    for j in range(0,SIZE):
        count, faceRow = calcSymm(face[(SIZE*j):(SIZE*(j+1))],maxSymmCenter,epsilon)
        face[(SIZE*j):(SIZE*(j+1))] = faceRow
        
    return (abs((SIZE/2)-maxSymmCenter)), maxSymm
#end func




#MAIN
print("Entering Main:")

allFaces = []
labelsRow = parseInput(allFaces,"facedata.csv")
for i in range(0,len(allFaces)):
    allFaces[i].pop() #pop the facelabel


classes = []
parseInput(classes,"faceclass.csv")
classes.pop(len(classes)-1)
for i in range(0,len(classes)):
    classes[i] = classes[i][0]


epsilon = 10 #tunable, range 10-15 (?)
start = 20 #tunable, range 10-20 (?) 546 total options

for i in range(0,len(allFaces)):
    transposeFace(allFaces[i])


minCount = len(allFaces)
minEpsilon = 0
for q in range(0,10):
    features = [] #reset features
    for i in range(0,len(allFaces)):
        epsilon = 5+q
        sim, dist = findCenterline(allFaces[i],epsilon,start )
        features.append([sim,dist])
    #end inner for loop

    SVM = LinearSVC(dual=False)
    SVM.fit(features,classes)
    predictions = SVM.predict(features)
    
    count = 0
    for i in range(0, len(allFaces)):
        if(predictions[i] != classes[i]):
            count += 1

    print("Train Error: " + str(float(count)/len(allFaces)))
    if(count < minCount):
        minCount = count
        minEpsilon = epsilon
#end for loop

print("Best train error: " + str(float(minCount)/len(allFaces)) + " at epsilon "
      "" + str(minEpsilon))

#for i in range(0,len(allFaces)):
#    transposeFace(allFaces[i])

#writeCSV(allFaces,labelsRow,'faces.csv')
#writeCSV(features,['similarity','distance'],'features.csv')




#end loop



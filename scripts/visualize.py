from PIL import Image , ImageTk, ImageDraw
import Tkinter
import numpy as np
import random
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

# left brow, left
# left brow, top
# left brow, right
# right brow, left
# right brow, top
# right brow, right
# left eye, left
# left eye, center
# left eye, right
# right eye, left
# right eye, center
# right eye, right
# nose, left
# nose, left nostril
# nose, tip
# nose, right nostril
# nose, right
# mouth, left
# mouth, top
# mouth, bottom
# mouth, right

def markface(f, index):
    # GET IMAGE
    fscaled = f*255
    w, h = 64, 64
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            data[i, j] = [0, fscaled[i,j], 0]
    img = Image.fromarray(data, 'RGB')
    img = img.resize((512,512))
    imgname = "pictures/" + str(index) + ".png"
    img.save(imgname)

    #STORE THE MARKS
    marks = np.zeros((22, 2), dtype=np.uint8)

    def button_click_exit_mainloop (event):
        rawx= event.x
        rawy = event.y
        x = int(math.floor(float(rawx)/8.0))
        y = int(math.floor(float(rawy)/8.0))
        print x, y
        rowindex = marks[21,0]
        marks[rowindex, 0] = x
        marks[rowindex, 1] = y
        #  print marks
        marks[21, 0] = rowindex + 1
        if rowindex + 1 == 21:
            rowindex = 0
            event.widget.quit() # this will cause mainloop to unblock.
    root = Tkinter.Tk()
    root.bind("<Button>", button_click_exit_mainloop)
    root.geometry('512x512')
    #load the image file we created before

    tkpi = ImageTk.PhotoImage(file=imgname)
    label_image = Tkinter.Label(root, image=tkpi)
    label_image.pack()

    #label_image.place(x=0,y=0,width=150,height=150)
    #input()
    root.mainloop()
    root.destroy()

    return marks[:21]

def showfacemini(f, features = np.zeros((16, 16), dtype=np.uint8)):
    fscaled = f*255

    w, h = 16, 16
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            data[i, j] = [0, fscaled[i,j], 0]

    featscaled = features * 255
    for i in range(16):
        for j in range(16):
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

def convolute(f, method="boxblur"):
    if method == "boxblur":
        cmatrix = np.ones((3,3), np.float64)/9
        buf = 1
        #  print cmatrix 
    if method == "edge":
        cmatrix = np.matrix([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]])
        #  cmatrix = np.matrix([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
        buf = 1
        print cmatrix 
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

def downsample(f):
    fdown = np.zeros((16,16), np.float64)
    for i in range(0,16):
        for j in range(0,16):
            r = i * 4
            c = j * 4
            fdown[i,j] = f[r:r+4,c:c+4].mean()
    return fdown

def main():
    faceclass = np.genfromtxt('../resources/faceclass.csv', delimiter=',')
    faceclass = faceclass[1:]

    facedata = np.genfromtxt('../resources/facedata.csv', delimiter=',')
    facedata = facedata[1:]

    # RANDOMIZE THE DATA
    #  perm = np.random.permutation(facedata.shape[0])
    #  facedata = facedata[perm, :]
    #  faceclass = faceclass[perm]

    faceindex = facedata[:,0] # get first col
    facedata = facedata[:,1:] # remove first col

    print "FACEDATA :", np.shape(facedata)
    print "FACEINDEX:", np.shape(faceindex)
    print "FACECLASS:", np.shape(faceclass)

    # NORMALIZE THE DATA
    for i in range(0,np.shape(facedata)[0]):
        normalize(facedata[i])

    ptrain = 0.9
    rows = np.shape(facedata)[0]
    tindex = int(ptrain * float(rows))
    DTrain = facedata[:tindex]
    DTest = facedata[tindex:]
    print "TRAIN :", np.shape(DTrain)
    print "TEST:", np.shape(DTest)

    
    # TRAIN THE MARKS
    for i in range(20):
        if i % 10 == 0 or i % 10 == 1:
            print "TRAINING INDEX", i
            f = facearrtomatrix(facedata[i])
            marks = markface(f, i)
            print marks
            np.savetxt("marks/" +str(i)+".csv", marks, delimiter=",", fmt="%d")

    # RUN TESTS
    #  featpos = np.zeros((16,16), np.uint8)
    #  featpos = np.zeros((64,64), np.uint8)
    #  for i in range(0,64):
        #  featpos[i,0] = 1
    #  for i in range(0,100):
        #  f = facearrtomatrix(facedata[i])
        #  #  fdown = downsample(f)
        #  fconv = convolute(f, "edge")
        #  if faceclass[i] == 1:
            #  showface(fconv, featpos)
        #  else:
            #  showface(fconv)
    


if __name__ == "__main__":
    main()

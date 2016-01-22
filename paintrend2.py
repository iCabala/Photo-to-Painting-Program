import os

###########################################################################
## Handout painting code.
###########################################################################
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import scipy as sci

np.set_printoptions(threshold = np.nan)  

def colorImSave(filename, array):
    imArray = sci.misc.imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        sci.misc.imsave(filename, cm.jet(imArray))
    else:
        sci.misc.imsave(filename, imArray)

def markStroke(mrkd, p0, p1, rad, val):
    # Mark the pixels that will be painted by
    # a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1).
    # These pixels are set to val in the ny x nx double array mrkd.
    # The paintbrush is circular with radius rad>0
    
    sizeIm = mrkd.shape
    sizeIm = sizeIm[0:2];
    nx = sizeIm[1]
    ny = sizeIm[0]
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    rad = max(rad,1)
    # Bounding box
    concat = np.vstack([p0,p1])
    bb0 = np.floor(np.amin(concat, axis=0))-rad
    bb1 = np.ceil(np.amax(concat, axis=0))+rad
    # Check for intersection of bounding box with image.
    intersect = 1
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        intersect = 0
    if intersect:
        # Crop bounding box.
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1],1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1],ny])]), axis=1)
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1],1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1],ny])]), axis=1)
        # Compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1
        szBB = [tmp[1], tmp[0]]
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        t = q1 - q0
        nrmt = np.linalg.norm(t)
        [x,y] = np.meshgrid(np.array([i+1 for i in range(int(szBB[1]))]), np.array([i+1 for i in range(int(szBB[0]))]))
        d = np.zeros(szBB)
        d.fill(float("inf"))
        
        if nrmt == 0:
            # Use distance to point q0
            d = np.sqrt( (x - q0[0])**2 +(y - q0[1])**2)
            idx = (d <= rad)
        else:
            # Use distance to segment q0, q1
            t = t/nrmt
            n = [t[1], -t[0]]
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            idx = (tmp >= 0) & (tmp <= nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q0[0])**2 +(y[np.where(idx)] - q0[1])**2)
            idx = (tmp > nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q1[0])**2 +(y[np.where(idx)] - q1[1])**2)

            #Pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        #Mark the pixels
        if np.any(idx.flatten('F')):
            xy = (bb0[1]-1+y[np.where(idx)] + sizeIm[0] * (bb0[0]+x[np.where(idx)]-2)).astype(int)
            sz = mrkd.shape
            m = mrkd.flatten('F')
            m[xy-1] = val
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order = 'F')

            '''
            row = 0
            col = 0
            for i in range(len(m)):
                col = i//sz[0]
                mrkd[row][col] = m[i]
                row += 1
                if row >= sz[0]:
                    row = 0
            '''
            
            
            
    return mrkd

def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    sizeIm = canvas.shape
    sizeIm = sizeIm[0:2]
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
    # Paint
    if np.any(idx.flatten('F')):
        canvas = np.reshape(canvas, (np.prod(sizeIm),3), "F")
        xy = y[idx] + sizeIm[0] * (x[idx]-1)
        canvas[xy-1,:] = np.tile(np.transpose(colour[:]), (len(xy), 1))
        canvas = np.reshape(canvas, sizeIm + (3,), "F")
    return canvas


if __name__ == "__main__":
    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open('tiger.jpg'))
    imRGB = double(imRGB) / 255.0
    plt.clf()
    plt.axis('off')
    
    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]
    # Set radius of paint brush and half length of drawn lines
    rad = 1
    halfLen = 10
    
    # Set up x, y coordinate images, and canvas.
    [x, y] = np.meshgrid(np.array([i+1 for i in range(int(sizeIm[1]))]), np.array([i+1 for i in range(int(sizeIm[0]))]))
    canvas = np.zeros((sizeIm[0],sizeIm[1], 3))
    canvas.fill(-1) ## Initially mark the canvas with a value out of range.
    # Negative values will be used to denote pixels which are unpainted.
    
    # Random number seed
    np.random.seed(29645)
    
    # Orientation of paint brush strokes
    theta = 2 * pi * np.random.rand(1,1)[0][0]
    # Set vector from center to one end of the stroke.
    delta = np.array([cos(theta), sin(theta)])
       
    time.time()
    time.clock()
    i = 0
    while (len(numpy.where(canvas==-1)[0])>0):
        
        leftHalfLen = 10
        rightHalfLen = 10
        
        # finding a negative pixel
        # Randomly select stroke center
        emptyspots = numpy.where(canvas==-1)
        randpix = np.random.randint(0, len(emptyspots[0]))
        
        emptyspot = array([float(emptyspots[1][randpix]), 
                           float(emptyspots[0][randpix])])
        # Grab colour from image at center position of the stroke.
        if part3[emptyspot[1], emptyspot[0]] == 1:
            leftHalfLen, rightHalfLen = 0, 0
        else:
            i = 0
            for i in range(0, rightHalfLen):
                rightPixel = emptyspot + i * delta 
                if part3[rightPixel[1], rightPixel[0]] == 1 or rightPixel[0] <= 0 or rightPixel[1] >= part3.shape[0] - 1:
                    rightHalfLen = i
                    break
                

            for j in range(1, leftHalfLen):
                leftPixel = emptyspot - j * delta
                if part3[leftPixel[1], leftPixel[0]] == 1 or leftPixel[1] <= 0 or leftPixel[0] >= part3.shape[1] - 1:
                    leftHalfLen = j
                    break
        print(leftHalfLen, rightHalfLen)
        colour = np.reshape(imRGB[emptyspot[1]-1, emptyspot[0]-1, :],(3,1))
        # Add the stroke to the canvas
        nx, ny = (sizeIm[1], sizeIm[0])
        length1, length2 = (rightHalfLen, leftHalfLen)
                 
            
        if abs(delta[0]) > abs(delta[1]):
            print 'a'
            canvas = paintStroke(canvas, x, y, (emptyspot + 1) - np.round(length2 * (delta / abs(delta[0]))), (emptyspot + 1) + np.round(length2 * (delta / abs(delta[0]))), colour, rad)
        else:
            print 'b'
            canvas = paintStroke(canvas, x, y, (emptyspot + 1) - np.round(length2 * (delta / abs(delta[1]))), (emptyspot + 1) + np.round(length2 * (delta / abs(delta[1]))), colour, rad)
        #print imRGB[cntr[1]-1, cntr[0]-1, :], canvas[cntr[1]-1, cntr[0]-1, :]
        numOfStrokes += 1
        print 'stroke', numOfStrokes
        
    print "done!"
    time.time()
    
    canvas[canvas < 0] = 0.0
    plt.clf()
    plt.axis('off')
    plt.imshow(canvas)
    plt.pause(3)
    colorImSave('output.png', canvas)

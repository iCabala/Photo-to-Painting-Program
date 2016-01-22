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
    i = 0
    while (len(numpy.where(canvas==-1)[0])>0):
        # finding a negative pixel
        # Randomly select stroke center
        cntr = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1], sizeIm[0]])) + 1
        cntr = np.amin(np.vstack((cntr, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        
        emptyspots = numpy.where(canvas==-1)
        k = np.random.randint(0, len(emptyspots[0]))
        emptyspot = array([float(emptyspots[1][k]), float(emptyspots[0][k])])
        # Grab colour from image at center position of the stroke.
        colour = np.reshape(imRGB[emptyspot[1]-1, emptyspot[0]-1, :],(3,1))
        # Add the stroke to the canvas
        nx, ny = (sizeIm[1], sizeIm[0])
        length1, length2 = (halfLen, halfLen)
        
        canvas = paintStroke(canvas, x, y, emptyspot - delta * length2, emptyspot + delta * length1, colour, rad)
        #print imRGB[cntr[1]-1, cntr[0]-1, :], canvas[cntr[1]-1, cntr[0]-1, :]
        i += 1
        print 'stroke', i
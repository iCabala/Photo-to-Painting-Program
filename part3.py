
    red, green, blue = imRGB[:,:,0], imRGB[:,:,1], imRGB[:,:,2]
    imGray = 0.30 * red + 0.59 * green + 0.11 * blue
    
    part3 = canny.canny(imGray, 4.0, 25, 5)
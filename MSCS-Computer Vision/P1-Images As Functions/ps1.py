import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.
        
    Returns:
        numpy.array: Output 2D array containing the red channel.
    """

    imgRed = image.copy()
    imgRed = imgRed[:,:,2]

    return np.float64(imgRed)
        
def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    
    imgGreen = image.copy()
    imgGreen = imgGreen[:,:,1]

    return np.float64(imgGreen)


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    imgBlue = image.copy()
    imgBlue = imgBlue[:,:,0]

    return np.float64(imgBlue)


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """

    imgSwapped = image.copy()
    imgSwapped[:,:,0] = extract_green(image).copy()
    imgSwapped[:,:,1] = extract_blue(image).copy()
    
    return np.float64(imgSwapped)


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    
    srcCopy = src.copy()
    dstCopy = dst.copy()

    srcHeight = np.size(srcCopy, 0)
    srcWidth = np.size(srcCopy, 1)
    dstHeight = np.size(dst, 0)
    dstWidth = np.size(dst, 1)
    shapeHeight = shape[0]
    shapeWidth = shape[1]

    if dstHeight < shapeHeight:
        shapeHeight = dstHeight
    if dstWidth < shapeWidth:
        shapeWidth = dstWidth
    '''
    print srcHeight, "srcHeight"
    print srcWidth, "srcWidth"
    print src_yCenter, "src_yCenter"
    print src_xCenter, "src_xCenter"
    print shapeHeight, "shapeHeight"
    print shapeWidth, "shapeWidth"
    '''

    src_yStart = math.trunc(srcHeight/2) - math.trunc(shapeHeight/2)
    src_xStart = math.trunc(srcWidth/2) - math.trunc(shapeWidth/2)
    dst_yStart = math.trunc(dstHeight/2) - math.trunc(shapeHeight/2)
    dst_xStart = math.trunc(dstWidth/2) - math.trunc(shapeWidth/2)
    
    cropped_img = srcCopy[src_yStart:src_yStart+shapeHeight,src_xStart:src_xStart+shapeWidth]
    '''cv2.imwrite('output/cropped.png', cropped_img)'''
    dstCopy[dst_yStart:dst_yStart + shapeHeight,dst_xStart:dst_xStart + shapeWidth] = cropped_img
    
    return np.float64(dstCopy)


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """

    

    imgCopy = image.copy()
    imgMin = np.float64(np.min(imgCopy))
    imgMax = np.float64(np.max(imgCopy))
    imgMean = np.float64(np.mean(imgCopy))
    imgStdDev = np.float64(np.std(imgCopy))
    statTuple = (imgMin, imgMax, imgMean, imgStdDev)

    return statTuple


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    
    imgCopy = np.float64(image.copy())
    
    '''
    imgScaled = np.float64(imgCopy * scale)

    factorA = imgScaled - np.min(imgScaled)
    factorB = np.abs(np.max(imgScaled)-np.min(imgScaled))
    imgNormalized = np.float64((255*(factorA/factorB)))
    '''

    minImg, maxImg, meanImg, stddevImg = image_stats(image)

    imgNormalized = np.float64((((imgCopy-meanImg)/stddevImg)*scale) + meanImg )   

    return imgNormalized

def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """

    imgHeight = np.size(image, 0)
    imgWidth = np.size(image, 1)

    shifted_img = image.copy()
    lastColumn = shifted_img[:,imgWidth-1]
    shifted_img[0:imgHeight,0:imgWidth-shift] = image[0:imgHeight, shift:imgWidth]
    for x in range(imgWidth-shift-1,imgWidth-1):
        shifted_img[:,x] = lastColumn

    return np.float64(shifted_img)
    


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """

    img1Copy = np.float64(img1.copy())
    img2Copy = np.float64(img2.copy())

    imgDiff = img1Copy-img2Copy

    factorA = (imgDiff - np.min(imgDiff))
    factorB = (np.max(imgDiff)-np.min(imgDiff))
    if factorB==0:
        imgNormalized=factorA*0
    else:
        imgNormalized = 255*(factorA/factorB)
    
    print imgNormalized    
    return imgNormalized


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    imgCopy = np.float64(image.copy())

    X = np.size(imgCopy, 0)
    Y = np.size(imgCopy, 1)
    
    noise = np.random.randn(X,Y)
    
    imgCopy[:,:,channel] += (noise*sigma)
    
    return np.float64(imgCopy)

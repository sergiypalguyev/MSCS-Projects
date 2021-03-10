"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os

#import matplotlib.pyplot as plt


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """        
    img = np.copy(image)
    sobel_x = cv2.Sobel( img, cv2.CV_64F, 1, 0, dst = None, ksize=3, scale=0.125, delta=0, borderType=cv2.BORDER_DEFAULT)
    
    #cv2.imshow("sobel_x", sobel_x)
    #cv2.waitKey(0) 
    
    return sobel_x


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """    
    img = np.copy(image)
    sobel_y = cv2.Sobel( img, cv2.CV_64F, 0, 1, dst = None, ksize=3, scale=0.125, delta=0, borderType=cv2.BORDER_DEFAULT)

    return sobel_y


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    import timeit
    start_time = timeit.default_timer()  
    
    A = np.copy(img_a)
    B = np.copy(img_b)
    
    Ix = gradient_x(A).astype(np.float32)
    Iy = gradient_y(A).astype(np.float32)
    It = cv2.subtract(A, B).astype(np.float32)    
        
    rows, cols = A.shape
    flow = np.zeros((rows, cols, 2), dtype=np.float32)    
    
    #print timeit.default_timer() - start_time, "gradients done"
    
    if k_type == 'uniform':
        Fxx = cv2.boxFilter(Ix**2, -1, ksize=(5,5), normalize=True)
        Fxy = cv2.boxFilter(Ix*Iy, -1, ksize=(5,5), normalize=True)
        Fyy = cv2.boxFilter(Iy**2, -1, ksize=(5,5), normalize=True)
        Fxt = cv2.boxFilter(Ix*It, -1, ksize=(5,5), normalize=True)
        Fyt = cv2.boxFilter(Iy*It, -1, ksize=(5,5), normalize=True)
        
        #print timeit.default_timer() - start_time, "filters done" 
    
        A = np.dstack((Fxx, Fxy, Fxy, Fyy))
        Fxt = -Fxt
        Fyt = -Fyt
        b = np.dstack((Fxt, Fyt))
        for r in range(rows):
            for c in range(cols):
                flow[r,c,:] = np.linalg.lstsq(A[r,c].reshape((2,2)), b[r,c])[0]      
                
    elif k_type == 'gaussian':
        kernel = cv2.getGaussianKernel( k_size, sigma)
        Fxx = cv2.sepFilter2D(Ix**2, -1, kernel, kernel)
        Fxy = cv2.sepFilter2D(Ix*Iy, -1, kernel, kernel)
        Fyy = cv2.sepFilter2D(Iy**2, -1, kernel, kernel)
        Fxt = cv2.sepFilter2D(Ix*It, -1, kernel, kernel)
        Fyt = cv2.sepFilter2D(Iy*It, -1, kernel, kernel)     
    
        A = np.dstack((Fxx, Fxy, Fxy, Fyy))
        Fxt = -Fxt
        Fyt = -Fyt
        b = np.dstack((Fxt, Fyt))
        for r in range(rows):
            for c in range(cols):
                flow[r,c,:] = np.linalg.lstsq(A[r,c].reshape((2,2)), b[r,c])[0]

    flow = -flow   
    return (flow[:, :, 0], flow[:, :, 1])

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
       
    imgIn = np.copy(image)
    
    imgIn = imgIn.astype(np.float32)
    
    gBlurImg = cv2.GaussianBlur(imgIn, (5,5), 1.2  )
    reduced1 = gBlurImg[::2,::2]
    
    return reduced1

def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    
    gauss = [image]
    for l in range(levels-1):
        gauss.append(reduce_image(gauss[-1]))
    return gauss    

    #raise NotImplementedError


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    
    
    combined = img_list[0]
    rows, cols = combined.shape
    for i in img_list[1:]:
        newRows, newCols = i.shape
        i = np.concatenate((i, np.zeros((rows-newRows, newCols))), axis=0)
        combined = np.concatenate((combined, i), axis=1)    
    return combined*255




    #raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    h, w = image.shape

    expanded_image = np.zeros((h*2, w*2))
    expanded_image[::2, ::2] = image
    
    kernel = cv2.getGaussianKernel(5, 1.1 )*2
    kernel = np.dot(kernel, kernel.T)
    expanded_image = cv2.filter2D(expanded_image, -1, kernel) 
    
    #cv2.imshow("expanded",expanded_image)
    #cv2.waitKey(0)

    return expanded_image    

    #raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    
    g_pyr = g_pyr[::-1]
    l_pyr = [g_pyr[0]]
    
    for i, img in enumerate(g_pyr[:-1]):        
        temp = expand_image(img)
        tRows, tCols = temp.shape
        gRows, gCols = g_pyr[i+1].shape
        if not tRows == gRows:
            temp = temp[:gRows-tRows, :]
        if not tCols == gCols:
            temp = temp[:, :gCols-tCols]            
        l_pyr.append(np.subtract(g_pyr[i+1].astype(np.float32),temp.astype(np.float32))*255)
   
    return l_pyr[::-1]

    #raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    rows, cols= image.shape
    
    print image.shape

    X, Y = np.meshgrid(range(cols), range(rows))
    
    mapX, mapY = cv2.convertMaps(X.astype(np.float32), Y.astype(np.float32), cv2.CV_16SC2)
    mapU, mapV = cv2.convertMaps(U.astype(np.float32), V.astype(np.float32), cv2.CV_16SC2)
   
    X = np.ndarray.flatten(mapU + mapX)
    Y = np.ndarray.flatten(mapV + mapX)
    
    warped = cv2.remap(image, X, Y, interpolation=interpolation, borderMode=border_mode)

    return warped

    #raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError

"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    #raise NotImplementedError
    
    return np.float(np.sqrt(np.square(p1[1]-p0[1])+np.square(p1[0]-p0[0])))


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    
    #raise NotImplementedError
    tupleList = []
    tupleList.append((0,0))
    tupleList.append((0,np.size(image, 1)))
    tupleList.append((np.size(image, 0),0))
    tupleList.append((np.size(image, 0),np.size(image, 1))) 
    
    return tupleList

def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (imageY, imageX) = img.shape[:2]
    q1 = img[0:np.size(img, 0)/2, 0:np.size(img, 1)/2]
    q2 = img[np.size(img, 0)/2:np.size(img, 0), 0:np.size(img, 1)/2]
    q3 = img[0:np.size(img, 0)/2, np.size(img, 1)/2:np.size(img, 1)]
    q4 = img[np.size(img, 0)/2:np.size(img, 0), np.size(img, 1)/2:np.size(img, 1)]            
    
    tmp = np.copy(template)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    #tmp = cv2.Canny(tmp, 150, 200)
    (templateY, templateX) = tmp.shape[:2]

    #result = cv2.matchTemplate(img,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    result1 = cv2.matchTemplate(q1,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    result2 = cv2.matchTemplate(q2,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    result3 = cv2.matchTemplate(q3,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    result4 = cv2.matchTemplate(q4,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    
    #the get the best match fast use this:
    #(min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result)
    #(x,y) = minloc

    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result1)
    (x1,y1) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result2)
    (x2,y2) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result3)
    (x3,y3) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result4)
    (x4,y4) = minloc    
    
    #get all the matches:
    #result2 = np.reshape(result, result.shape[0]*result.shape[1])
    #sort = np.argsort(result2)
    
    #(y1, x1) = np.unravel_index(sort[0], result.shape) #first best match
    #(y2, x2) = np.unravel_index(sort[1], result.shape) #second best match  
    #(y3, x3) = np.unravel_index(sort[2], result.shape) #third best match  
    #(y4, x4) = np.unravel_index(sort[3], result.shape) #fourth best match  
    
    cv2.circle(image, (           x1+(templateX/2),            y1+(templateY/2)), 1, (0,255,0), 2)  
    cv2.circle(image, (           x2+(templateX/2), (imageY/2)+y2+(templateY/2)), 1, (0,255,0), 2) 
    cv2.circle(image, ((imageX/2)+x3+(templateX/2),            y3+(templateY/2)), 1, (0,255,0), 2)   
    cv2.circle(image, ((imageX/2)+x4+(templateX/2), (imageY/2)+y4+(templateY/2)), 1, (0,255,0), 2)          
     
    return ((           x1+(templateX/2),            y1+(templateY/2)),
            (           x2+(templateX/2), (imageY/2)+y2+(templateY/2)),
            ((imageX/2)+x3+(templateX/2),            y3+(templateY/2)),
            ((imageX/2)+x4+(templateX/2), (imageY/2)+y4+(templateY/2)))


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img = np.copy(image)
    
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    i = 0
    print markers
    for (x,y) in markers:
        print i
        if x1 == 0 and y1 == 0:
            (x1,y1) = markers[i]
        else:
            (x0,y0) = (x1,y1)
            (x1,y1) = markers[i]
            cv2.line(img, (x0,y0), (x1,y1), (255,0,0), thickness)
        i += 1
    
    cv2.imshow('image', img)
    cv2.waitKey(0)        

    return img


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = None

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    raise NotImplementedError

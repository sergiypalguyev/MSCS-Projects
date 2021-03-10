"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""

import os
import cv2
import numpy as np
IMG_DIR = "input_images"


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
    #(0,0), (0, height-1), (width-1, 0), (width-1, height-1)

    (y,x) = image.shape[:2]    
    tupleList = []
    tupleList.append((0,0))
    tupleList.append((0,y-1))
    tupleList.append((x-1,0))
    tupleList.append((x-1,y-1))
    
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

    (templateY, templateX) = template.shape[:2]
    tmp = np.copy(template[int(templateX*0.3):int(templateX*0.7),int(templateY*0.3):int(templateY*0.7)])
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    (tmpY, tmpX) = tmp.shape[:2]
     
    result1 = cv2.matchTemplate(q1,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    result2 = cv2.matchTemplate(q2,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    result3 = cv2.matchTemplate(q3,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    result4 = cv2.matchTemplate(q4,tmp,cv2.cv.CV_TM_SQDIFF_NORMED)
    
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result1)
    (x1,y1) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result2)
    (x2,y2) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result3)
    (x3,y3) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result4)
    (x4,y4) = minloc  
    
    boolArr=[]    
    res = img[y1:y1+tmpY,x1:x1+tmpX][1:5,1:5]
    zer = img[y1:y1+tmpY,x1:x1+tmpX][9:13,1:5]
    if np.all(np.greater(res,zer)):
        boolArr.append(True)
    res = img[(imageY/2)+y2:(imageY/2)+y2+tmpY,x2:x2+tmpX][1:5,1:5]
    zer = img[(imageY/2)+y2:(imageY/2)+y2+tmpY,x2:x2+tmpX][9:13,1:5]
    if np.all(np.greater(res,zer)):
        boolArr.append(True)
    res = img[y3:y3+tmpY,(imageX/2)+x3:(imageX/2)+x3+tmpX][1:5,1:5]
    zer = img[y3:y3+tmpY,(imageX/2)+x3:(imageX/2)+x3+tmpX][9:13,1:5]
    if np.all(np.greater(res,zer)):
        boolArr.append(True)
    res = img[(imageX/2)+y4:(imageX/2)+y4+tmpY,(imageX/2)+x4:(imageX/2)+x4+tmpX][1:5,1:5]
    zer = img[(imageX/2)+y4:(imageX/2)+y4+tmpY,(imageX/2)+x4:(imageX/2)+x4+tmpX][9:13,1:5]
    if np.all(np.greater(res,zer)):
        boolArr.append(True)
        
    if len(boolArr)>2:
        #cv2.circle(image, (           x1+(tmpX/2),            y1+(tmpY/2)), 1, (0,255,0), 2)  
        #cv2.circle(image, (           x2+(tmpX/2), (imageY/2)+y2+(tmpY/2)), 1, (0,255,0), 2) 
        #cv2.circle(image, ((imageX/2)+x3+(tmpX/2),            y3+(tmpY/2)), 1, (0,255,0), 2)   
        #cv2.circle(image, ((imageX/2)+x4+(tmpX/2), (imageY/2)+y4+(tmpY/2)), 1, (0,255,0), 2)   
    
        #cv2.imshow('image', image)
        #cv2.waitKey(0)          
        
        return ((           x1+(tmpX/2),            y1+(tmpY/2)),
                (           x2+(tmpX/2), (imageY/2)+y2+(tmpY/2)),
                ((imageX/2)+x3+(tmpX/2),            y3+(tmpY/2)),
                ((imageX/2)+x4+(tmpX/2), (imageY/2)+y4+(tmpY/2)))      

    tmpFlip = tmp.swapaxes(-2,-1)[...,::-1]
    (tmpFY, tmpFX) = tmpFlip.shape[:2]
    
    result11 = cv2.matchTemplate(q1,tmpFlip,cv2.cv.CV_TM_SQDIFF_NORMED)
    result22 = cv2.matchTemplate(q2,tmpFlip,cv2.cv.CV_TM_SQDIFF_NORMED)
    result33 = cv2.matchTemplate(q3,tmpFlip,cv2.cv.CV_TM_SQDIFF_NORMED)
    result44 = cv2.matchTemplate(q4,tmpFlip,cv2.cv.CV_TM_SQDIFF_NORMED) 

    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result11)
    (xF1,yF1) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result22)
    (xF2,yF2) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result33)
    (xF3,yF3) = minloc
    (min_x,max_y,minloc,maxloc) = cv2.minMaxLoc(result44)
    (xF4,yF4) = minloc  
    
    #cv2.circle(image, (           xF1+(tmpFX/2),            yF1+(tmpFY/2)), 1, (255,0,0), 2)  
    #cv2.circle(image, (           xF2+(tmpFX/2), (imageY/2)+yF2+(tmpFY/2)), 1, (255,0,0), 2) 
    #cv2.circle(image, ((imageX/2)+xF3+(tmpFX/2),            yF3+(tmpFY/2)), 1, (255,0,0), 2)   
    #cv2.circle(image, ((imageX/2)+xF4+(tmpFX/2), (imageY/2)+yF4+(tmpFY/2)), 1, (255,0,0), 2)     
    
    #cv2.imshow('image', image)
    #cv2.waitKey(0)   
    
    return ((           xF1+(tmpX/2),            yF1+(tmpY/2)),
            (           xF2+(tmpX/2), (imageY/2)+yF2+(tmpY/2)),
            ((imageX/2)+xF3+(tmpX/2),            yF3+(tmpY/2)),
            ((imageX/2)+xF4+(tmpX/2), (imageY/2)+yF4+(tmpY/2)))      
             


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
    (x0,y0) = markers[0]
    (x1,y1) = markers[1]
    (x2,y2) = markers[2]
    (x3,y3) = markers[3]

    cv2.line(img, (x0,y0), (x1,y1), (0,255,0), thickness) 
    cv2.line(img, (x1,y1), (x3,y3), (0,255,0), thickness) 
    cv2.line(img, (x3,y3), (x2,y2), (0,255,0), thickness) 
    cv2.line(img, (x2,y2), (x0,y0), (0,255,0), thickness)    
    
    #cv2.imshow('image', img)
    #cv2.waitKey(0)        

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
    
    imA = np.copy(imageA)
    imB = np.copy(imageB)
        
    minX = 0; minY = 0; maxX = 0; maxY = 0;
    tmp = np.dot(homography,np.array([[1],[1],[1]]))
    minX = tmp[0]/tmp[2]
    maxX = tmp[0]/tmp[2]
    minY = tmp[1]/tmp[2]
    maxY = tmp[1]/tmp[2]
    
    tmp = np.dot(homography,np.array([[imA.shape[1]],[1],[1]]))
    if minX > tmp[0]/tmp[2]:
        minX = tmp[0]/tmp[2]
    if maxX < tmp[0]/tmp[2]:
        maxX = tmp[0]/tmp[2]
    if minY > tmp[1]/tmp[2]:
        minY = tmp[1]/tmp[2]
    if maxY < tmp[1]/tmp[2]:
        maxY = tmp[1]/tmp[2]
    
    tmp = np.dot(homography,np.array([[1],[imA.shape[0]],[1]]))
    if minX > tmp[0]/tmp[2]:
        minX = tmp[0]/tmp[2]
    if maxX < tmp[0]/tmp[2]:
        maxX = tmp[0]/tmp[2]
    if minY > tmp[1]/tmp[2]:
        minY = tmp[1]/tmp[2]
    if maxY < tmp[1]/tmp[2]:
        maxY = tmp[1]/tmp[2]
    
    tmp = np.dot(homography,np.array([[imA.shape[1]],[imA.shape[0]],[1]]))
    if minX > tmp[0]/tmp[2]:
        minX = tmp[0]/tmp[2]
    if maxX < tmp[0]/tmp[2]:
        maxX = tmp[0]/tmp[2]
    if minY > tmp[1]/tmp[2]:
        minY = tmp[1]/tmp[2]
    if maxY < tmp[1]/tmp[2]:
        maxY = tmp[1]/tmp[2]   
       
    for i in range(int(maxY-minY)):
        for j in range(int(maxX-minX)):
            tt = np.array([[j+minX],[i+minY],[1]])
            tmp = np.dot(np.linalg.inv(homography),tt)
            x1=int(tmp[0]/tmp[2])
            y1=int(tmp[1]/tmp[2])

            if x1>0 and y1>0 and x1<imA.shape[1] and y1<imA.shape[0]:
                imB[int(i+minY),int(j+minX),:] = imA[y1,x1,:]
    
    #cv2.imshow('image', imB)
    #cv2.waitKey(0)   

    return imB


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

    p1=np.copy(src_points)
    p2=np.copy(dst_points)
    
    M = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        M.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        M.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        
    U, S, V = np.linalg.svd(np.asarray(M))
    L = V[-1,:] / V[-1,-1]
    H = L.reshape(3, 3)
    
    return H   


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

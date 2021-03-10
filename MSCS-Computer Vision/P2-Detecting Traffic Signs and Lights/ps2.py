"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

#Helper function for creating a black image of the same size as input image
def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)

    color = tuple(reversed(rgb_color))
    
    image[:] = color

    return image

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    image = np.copy(img_in);
    
    #Find the green light, unique to traffic-lights
    low = np.array([0,100,0])
    upp = np.array([50,255,50])
    mask = cv2.inRange(img_in, low, upp)
    

    circleX = 0;
    circleY = 0;
    circleR = 0;
    imgYellowX = 0
    imgYellowY = 0
    imgMaxVal = 0
    imgMaxX = 0
    imgMaxY = 0
    imgMaxR = 0
    for i in radii_range:
        hits = cv2.HoughCircles(mask, cv2.cv.CV_HOUGH_GRADIENT,1,20, param1=50,param2=10,minRadius=i, maxRadius=i);
        if hits is not None:
            circles = np.round(hits[0, :]).astype("int")
            for (x,y,r) in circles:
                circleX = x
                circleY = y
                circleR = r

            #create a black background, and paste the traffic light into the black image - size of original image
            onBlack = create_blank(np.size(img_in,1), np.size(img_in,0), (0,0,0))
            onBlack[:,circleX-circleR:circleX+circleR] = img_in[0:np.size(img_in,0),circleX-circleR:circleX+circleR]

            gray = cv2.cvtColor(onBlack,cv2.COLOR_BGR2GRAY);

            #Find the center of the traffic-light (yellow light)
            for i in radii_range:
                lights = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT,1,20, param1=50,param2=10,minRadius=i, maxRadius=i);
                if lights is not None:
                    circles = np.round(lights[0, :]).astype("int")
                    for (x, y, r) in circles:
                        lower = [0, 100, 100]
                        upper = [30, 255, 255]
                        
                        tempImg = img_in[y-r:y+r, x-r:x+r]
                        
                        mask = cv2.inRange(tempImg, np.array(lower, dtype = "uint8"), np.array(upper, dtype = "uint8"))
                                                
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                        if maxVal > 0:
                            imgYellowX = x
                            imgYellowY = y
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray[y-r:y+r, x-r:x+r])
                        if maxVal > imgMaxVal:
                            imgMaxVal = maxVal
                            imgMaxX = x
                            imgMaxY = y
                            imgMaxR = r
   
    center = (imgYellowX, imgYellowY)
    print center

    if imgYellowX == 0 and imgYellowY == 0:
        center = (0,0)
        return (center,'none')
    else:                    
        #Masks for Red, Yellow, Green colors
        ColorMasks = [([0, 0, 200], [0, 0, 255]),
                      ([0, 200, 200], [0, 255, 255]),
                      ([0, 200, 0], [0, 255, 0])]

        #Evaluate each light to see which one is brightest
        brightImg = image[imgMaxY-imgMaxR:imgMaxY+imgMaxR, imgMaxX-imgMaxR:imgMaxX+imgMaxR]

        iterator = 0
        colorInt = 0
        for (lower, upper) in ColorMasks:
            mask = cv2.inRange(brightImg, np.array(lower, dtype = "uint8"), np.array(upper, dtype = "uint8"))
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mask)
            if maxVal > 0:
                colorInt = iterator
            iterator += 1
       
        if colorInt == 0:
            print (center,'red')
            return (center,'red')
        elif colorInt == 1:
            print (center,'yellow')
            return (center,'yellow')
        elif colorInt == 2:
            print (center,'green')
            return (center,'green')

def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinat
    """
    
    image = np.copy(img_in)

    low = np.array([0,0,100])
    upp = np.array([15,15,255])
    mask = cv2.inRange(img_in, low, upp)
    can = cv2.Canny(mask,150,200)   
    
    rho = 1
    theta = np.pi/180
    threshold = 25
    #min_line_length = 93
    max_line_gap = 10

    lineIx = 0
    lineX = 0
    lineY = 0
        
    lengths = range (100,90,-1)
    for length in lengths:
        lines = cv2.HoughLinesP(can, rho, theta, threshold, np.array([]), length, max_line_gap)
        if lines is not None and lineX==0 and lineY==0 and lineIx==0:
            if 11<lines.size:     
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
                        cv2.circle(image,(abs(x1),abs(y1)), 5, (0,255,0), 2)
                        cv2.circle(image,(abs(x2),abs(y2)), 5, (0,255,0), 2)
                        lineX += x1 + x2
                        lineY += y1 + y2
                        lineIx += 1

    if lineIx ==0 and lineX==0 and lineY==0:
        return (0,0)
    return (abs(int(lineX/(lineIx*2))),abs(int(lineY/(lineIx*2))))

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    image = np.copy(img_in)

    low = np.array([0,0,100])
    upp = np.array([20,20,255])
    mask = cv2.inRange(image, low, upp)
    can = cv2.Canny(mask,150,200)
    
    rho = 1
    theta = np.pi/180
    threshold = 23
    min_line_length = 27
    max_line_gap = 30
##    threshold = 10
##    min_line_length = 25
##    max_line_gap = 18
    lines = cv2.HoughLinesP(can, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    lineIx = 0
    lineX = 0
    lineY = 0
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if -2<=abs(x1-x2) - abs(y1-y2)<=2:
                    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.circle(image,(abs(x1),abs(y1)), 5, (0,255,0), 2)
                    cv2.circle(image,(abs(x2),abs(y2)), 5, (0,255,0), 2)
                    lineX += x1 + x2
                    lineY += y1 + y2
                    lineIx += 1
                    
    if lineIx ==0 and lineX==0 and lineY==0:
        return (0,0) 
    return (abs(int(lineX/(lineIx*2)-5)),abs(int(lineY/(lineIx*2))))

def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    image = np.copy(img_in)
    
##    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##    edges = cv2.Canny(gray, 150,200)
    
    low = np.array([0,200,200])
    upp = np.array([20,255,255])
    mask = cv2.inRange(image, low, upp)
    edges = cv2.Canny(mask, 150,200)

    rho = 1
    theta = np.pi/180
    threshold = 40
    min_line_length = 20
    max_line_gap = 10

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    lineIx = 0
    lineX = 0
    lineY = 0
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.circle(image,(abs(x1),abs(y1)), 5, (0,255,0), 2)
                cv2.circle(image,(abs(x2),abs(y2)), 5, (0,255,0), 2)
                if (x2-x1)>100 or y2-y1>100:
                    continue
                lineX += x1 + x2
                lineY += y1 + y2
                lineIx += 1

    if lineIx ==0 and lineX==0 and lineY==0:
        return (0,0) 
    return (abs(int(lineX/(lineIx*2))),abs(int(lineY/(lineIx*2)))) 

def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    image = np.copy(img_in)
    
##    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##    edges = cv2.Canny(gray, 150,200)

    low = np.array([0,100,200])
    upp = np.array([20,200,255])
    mask = cv2.inRange(image, low, upp)
    edges = cv2.Canny(mask, 150,200)
    
    rho = 1
    theta = np.pi/180
    threshold = 40
    min_line_length = 20
    max_line_gap = 10
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    lineIx = 0
    lineX = 0
    lineY = 0
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.circle(image,(abs(x1),abs(y1)), 5, (0,255,0), 2)
                cv2.circle(image,(abs(x2),abs(y2)), 5, (0,255,0), 2)
                if (x2-x1)>100 or y2-y1>100:
                    continue
                lineX += x1 + x2
                lineY += y1 + y2
                lineIx += 1

    if lineIx ==0 and lineX==0 and lineY==0:
        return (0,0) 
    return (abs(int(lineX/(lineIx*2))),abs(int(lineY/(lineIx*2)))) 

def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    
    image = np.copy(img_in)
    
##    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##    edges = cv2.Canny(gray, 150,200)

    low = np.array([0,0,200])
    upp = np.array([20,20,255])
    mask = cv2.inRange(image, low, upp)
    edges = cv2.Canny(mask, 150,200)

    radii_range = range(20,40,1)
    circleX = 0
    circleY = 0
    circleR = 0
    for i in radii_range:
        hits = cv2.HoughCircles(edges, cv2.cv.CV_HOUGH_GRADIENT,1,100, param1=50, param2=15, minRadius=25, maxRadius=38);
        if hits is not None:
            circles = np.round(hits[0, :]).astype("int")
            for (x,y,r) in circles:
                circleX = x
                circleY = y
                circleR = r
    if circleX==0 and circleY==0 and circleR==0:
        return (0,0)
    return (circleX, circleY)

def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    image = np.copy(img_in)    
    
    newDictionary = {}
    entryX, entryY              = do_not_enter_sign_detection(image)
    if entryX>0 and entryY>0:
        newDictionary['no_entry'] = (entryX,entryY)
        
    constructionX,constructionY = construction_sign_detection(image)
    if constructionX>0 and constructionY>0:
        newDictionary['construction'] = (constructionX,constructionY)
        
    warningX, warningY          = warning_sign_detection(image)
    if warningX>0 and warningY>0:
        newDictionary['warning'] = (warningX,warningY)
        
    stopX, stopY                = stop_sign_detection(image)
    if stopX>0 and entryY>0:
        newDictionary['stop'] = (stopX,stopY)
        
    yieldX, yieldY              = yield_sign_detection(image)
    if yieldX>0 and yieldY>0:
        newDictionary['yield'] = (yieldX, yieldY)
    
    radii_range = range(5, 40, 1)
    result = traffic_light_detection(image, radii_range)
    if result is not None:
        coords = result[0]
        state = result[1]
        x = coords[0]
        y = coords[1]
        if x>0 and y>0:
            newDictionary['traffic_light'] = coords

    return newDictionary


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    
    image = np.copy(img_in)

    denoised = cv2.fastNlMeansDenoisingColored(image, None, 25,25, 3,31)

    dictionary = traffic_sign_detection(denoised)

    return dictionary


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    image = np.copy(img_in)
    
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 25,25, 3,31)

    dictionary = traffic_sign_detection(denoised)

    return dictionary

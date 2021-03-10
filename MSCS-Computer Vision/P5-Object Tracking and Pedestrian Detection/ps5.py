"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        #raise NotImplementedError
        pass

    def predict(self):
        #raise NotImplementedError
        pass

    def correct(self, meas_x, meas_y):
        #raise NotImplementedError
        pass

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        
        self.template = template
        self.frame = frame
        
        (r, c) = template.shape 
        (h, w) = frame.shape
    
        rTemp = np.random.randint(r, h/2, self.num_particles)
        cTemp = np.random.randint(c, w/2,self.num_particles)
    
        self.particles = np.column_stack((rTemp, cTemp))
        self.weights = np.ones((self.num_particles, 1))/ self.num_particles    
                       
        raise NotImplementedError

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        
        a, b, c = self.template.shape
    
        image2 = self.frame[int(template - int(a / 2)):int(template + int(a / 2)),int(frame_cutout - int(b / 2)):int(frame_cutout + int(b / 2)+1)]        
        image1 = self.template
    
        gray_image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    
        (r,c) = gray_image1.shape

        temp = np.sum((gray_image1.astype("float") - gray_image2.astype("float")) ** 2)
        temp = temp/float(r*c)
        simVal = np.exp((temp/float(-2 * ((self.sigma_mse ** 2)))))

        return simVal

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """  
        n = len(self.particles)
        particles0 = np.random.choice(a=self.particles[:,0],size=n,p=self.weights[:,0])
        particles1 = np.random.choice(a=self.particles[:,1],size=n,p=self.weights[:,0])
                
        return np.column_stack((particles0,particles1))  

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        
        # update the weights based on similarity
        self.template = self.template
        self.frame = frame

        row = self.template.shape[0]
        col = self.template.shape[1]

        row_frame = frame.shape[0]
        col_frame = frame.shape[1]
        i = 0

        for p in self.particles :
            self.weights[i] = self.get_error_metric(p[0], p[1])
            # print self.weights[i]
            i = i + 1

        (r,c) = self.template.shape
        (h,w) = self.frame.shape
        i = 0
        particles = np.copy(self.particles)
        for p in particles:
            if p[0] <= r : self.weights[i] = 0
            elif  p[0] >= (h - r) :
                self.weights[i] = 0
            elif p[1] <= c :
                self.weights[i] = 0
            elif p[1] >=(w - c) :
                self.weights[i] = 0
            else :
                self.weights [i] = self.weights[i]
            i = i + 1

        total_weight = np.sum(self.weights[:,0])

        if(total_weight == 0):
            total_weight = self.num_particles

        self.weights = self.weights/float(total_weight)

        self.particles = resample_particles()

        if n > self.num_particles:
            self.particles = np.resize(self.particles, (self.num_particles, 2))
            self.weights = np.resize(self.weights, (self.num_particles, 1))

        n = len(self.particles)
        for i in range (n) :
            delta_row = random.gauss(0, self.sigma_dyn)
            delta_col = random.gauss(0, self.sigma_dyn)
            delta = np.array([np.round(delta_row), np.round(delta_col)])
            self.particles[i] += delta.astype("int")

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        wSum = 0
        for i in range(len(self.particles)):
            p = self.particles[i].astype('int')
            wSum = wSum + math.hypot((x_weighted_mean - p[0]),(y_weighted_mean - p[1])) * self.weights[i]

        cv2.circle(frame_in,(y_weighted_mean,x_weighted_mean),wSum,(255,0,0))
        
        a,b,c = self.template.shape
        V0 = y_weighted_mean + b/2
        U0 = x_weighted_mean + a/2
        V1 = y_weighted_mean - b/2
        U1 = x_weighted_mean - b/2

        cv2.rectangle(frame_in,(V0,U0),(V1,U1),(0,255,0))
        
        raise NotImplementedError


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        
        
        self.shrink_window = kwargs.get('shrink_window', False)
        self.frame_num = 0
        if self.shrink_window:
            self.hevery = 1
            self.wevery = 3      
                
        (r, c) = template.shape 
        (h, w) = frame.shape
    
        rTemp = np.random.randint(r, h/2, self.num_particles)
        cTemp = np.random.randint(c, w/2,self.num_particles)
    
        self.particles = np.column_stack((rTemp, cTemp))
        self.weights = np.ones((self.num_particles, 1))/ self.num_particles          
            
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """   
    
        index = np.unravel_index(np.argmax(self.weights),(self.weights.shape))[0]
        u, v = self.particles[index]
        self.template = cv2.addWeighted(self.get_error_metric(u,v),self.alpha,self.template,(1 - self.alpha),0)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError
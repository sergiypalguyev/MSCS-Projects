import cv2
import os
import numpy as np
import math


# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "output"
CLASSIFIER = "classifier"
VIDEO = "video.mp4"

POS_DIR = os.path.join(INPUT_DIR, "pos")
NEG_DIR = os.path.join(INPUT_DIR, "neg")

class ViolaJones:
    def __init__(self, pos, neg, integral_images):
	self.haarFeatures = []
	self.integralImages = integral_images
	self.classifiers = []
	self.alphas = []
	self.posImages = pos
	self.negImages = neg
	self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
	FeatureTypes = {"two_horizontal": (2, 1),
	                "two_vertical": (1, 2),
	                "three_horizontal": (3, 1),
	                "three_vertical": (1, 3),
	                "four_square": (2, 2)}

	haarFeatures = []
	for _, feat_type in FeatureTypes.iteritems():
	    for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
		for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
		    for posi in range(0, 24 - sizei + 1, 4):
			for posj in range(0, 24 - sizej + 1, 4):
			    haarFeatures.append(
			        HaarFeature(feat_type, [posi, posj],
			                    [sizei-1, sizej-1]))
	self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

	scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
	for i, im in enumerate(self.integralImages):
	    scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

	weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
	    2*len(self.posImages))
	weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
	    2*len(self.negImages))
	weights = np.hstack((weights_pos, weights_neg))

	for i in range(num_classifiers):
	    #TODO
	    raise NotImplementedError

    def predict(self, images):
	"""Return predictions for a given list of images.

	Args:
	    images (list of element of type numpy.array): list of images (observations).

	Returns:
	    list: Predictions, one for each element in images.
	"""

	ii = convert_images_to_integral_images(images)

	scores = np.zeros((len(ii), len(self.haarFeatures)))

	result = []

	for x in scores:
	    # TODO
	    raise NotImplementedError

	return result

    def signDetection(self, image, filename):
	TODO
	raise NotImplementedError



def load_images_from_dir(data_dir, size=(24, 24), ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imagesFiles = sorted(imagesFiles)
    
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs


def convert_images_to_integral_images(images):
    int_imgs = []
    for img in images:
	int_imgs.append(np.cumsum(np.cumsum(img, axis=0), axis=1))

    return int_imgs

def run():
	
	pos = load_images_from_dir(POS_DIR)[:20]
	neg = load_images_from_dir(NEG_DIR)
	images = pos + neg

	integral_images = convert_images_to_integral_images(images)
	VJ = ViolaJones(pos, neg, integral_images)
	VJ.createHaarFeatures()
	VJ.train(5)
	
	#TODO
	cascade = cv2.CascadeClassifier(CLASSIFIER)
	cap = cv2.VideoCapture(VIDEO)
	
	while True:
	    ret, frame = cap.read()
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    signs = cascade.detectMultiScale(gray, 30, 30)
	    #draw a box
	    for (x,y,w,h) in signs:
		cv2.rectangle(gray, (x,y), (x+w, y+h), (255,0,0), 2)
	
	    cv2.imshow('gray',gray)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    run()

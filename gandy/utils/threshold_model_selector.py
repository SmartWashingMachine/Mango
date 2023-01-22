from math import sqrt
import numpy as np

class ThresholdModelSelector():
	def __init__(self, models, thresholds):
		"""
		Some image models only work on certain image sizes, but we want
		to minimize just how much we have to resize our input images to get
		an output. This class will find the "optimal" model for the input images.

		models is a list of models.
		thresholds is a list of tuples, where each tuple has two elements:
			The first element is the expected width for the input.
			The second element is the expected height for the input.

		Each models index corresponds to the threshold index.
		"""

		self.models = models
		self.thresholds = thresholds

	def distance(self, p1, p2):
		# d = sqrt ( (x2 - x1)^2 + (y2 - y1)^2 )

		x1, y1 = p1
		x2, y2 = p2
		return sqrt(pow(x2 - x1) + (y2 - y1))

	def get_optimal_model(self, image_size):
		"""
		Gets the best model by minimizing Euclidean distance.

		image_size is a tuple of (width, height).
		"""
		distances = np.array([self.distance(image_size, thresh) for thresh in self.thresholds])
		closest_idx = np.argmin(distances)

		return self.models[closest_idx]

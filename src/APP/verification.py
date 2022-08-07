import os

import numpy as np

from preprocess import preprocess


class verify:

    def __init__(self, model, detection_threshold, verification_threshold):
        self.model = model
        self.detection_threshold = detection_threshold
        self.verification_threshold = verification_threshold
    

    def verify(self):
        results = []
        for image in os.listdir(os.path.join('data/application_data', 'verification_img')):
            input_img = preprocess(os.path.join('data/application_data', 'input_img', 'input_img.jpg')).preprocess()
            verification_img = preprocess(os.path.join('data/application_data', 'verification_img', image)).preprocess()

            result = self.model.predict(list(np.expand_dims([input_img, verification_img])))
            results.append(result)


        detection = np.sum(np.array(results) > self.detection_threshold)
        verification = detection / len(os.listdir(os.path.join('data/application_data', 'verification_img')))
        verified = verification > self.verification_threshold

        return results, verified

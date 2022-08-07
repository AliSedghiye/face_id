
from preprocess import preprocess


class preprocess_twin:
  
  def __init__(self, input_img, validation_img, label) :

    self.input_img = input_img
    self.validation_img = validation_img
    self.label = label


  def preprocess_twin(self):
    return (preprocess(self.input_img).preprocess(), preprocess(self.validation_img).preprocess(), self.label)
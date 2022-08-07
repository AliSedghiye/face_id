import os

import cv2
import numpy as np
from keras.models import load_model
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from capture import camera
from L1Dist import L1Dist
from verification import verify


class camapp(App):

    def __init__(self, model, detection_threshold, verification_threshold):
        self.model = model
        self.detection_threshold = detection_threshold
        self.verification_threshold = verification_threshold
        
        ret, frame = self.cam_open()
        self.save_input_image(frame)
        self.verify = verify(self.model, self.detection_threshold, self.verification_threshold).verify()
        self.results, self.verified = self.verify


    def build(self):

        # save verification images
        self.first_verification()

        # main layout component
        self.img1 = Image(size_hint = (1, 0.8))
        self.button = Button(text='verify', on_press=self.verify, size_hint = (1, 0.1))
        self.verification_Label = Label(text='verification uninitiateiated', size_hint = (1, 0.1))

        # add items to layout
        layout = BoxLayout(orientaion='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_Label)

        # set verification text
        self.verification_Label.text = 'verified' if self.verified == True else 'unverified'


        # log out details
        Logger.info(self.results)
        Logger.info(np.sum(np.array(self.results) > 0.5))

        return layout

    def update(self, *args):

        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = img_texture


    def cam_open(self):
        # setup videocapture device
        self.capture = camera().open_cam()
        Clock.schedule_interval(self.update, 1.0/33.0)
        self.ret, self.frame = self.capture
        return self.ret, self.frame

    def save_input_image(self, frame):
        cv2.imwrite(os.path.join('data/application_data/input_img' , 'input_img.jpg'), frame)

    def first_verification(self):
        # call this function to save verification image
        camera().verification_cam()


model = load_model('data/siamesemodel.h5', custom_objects={'L1Dist' : L1Dist()})


if __name__ == '__main__':
    face_id = camapp(model, 0.5, 0.5)
    face_id.build()

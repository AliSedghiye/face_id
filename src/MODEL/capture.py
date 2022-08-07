import os
import uuid

import cv2


class camera:

    def verification_cam(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            # frame must be 250*250 px
            frame = frame[120:120+250, 200:200+250, :] 
            cv2.imshow('verification', frame)

            veryfication_img = os.path('data/application_data/verification_img')
            if cv2.waitKey(1) & 0xff == ord('v'):
                image_name = os.path.join(veryfication_img, f'/{uuid.uuid1()}.jpg')
                cv2.imwrite(image_name, frame)

            elif cv2.waitKey(1) & 0xff == ord('q'):
                break
            
    def collect(anch_data, pos_data):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            frame = frame[120:120+250, 200:200+250, :] 
            cv2.imshow('collection', frame)

            # collect anchor
            if cv2.waitKey(1) & 0xff == ord('a'):
                imagename = os.path.join(anch_data, f'{uuid.uuid1()}.jpg')
                cv2.imwrite(imagename, frame)

            # collect posetive
            if cv2.waitKey(1) & 0xff == ord('p'):
                imagename = os.path.join(pos_data, f'{uuid.uuid1()}.jpg')
                cv2.imwrite(imagename, frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break


    def open_cam(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            # frame must be 250*250 px
            frame = frame[120:120+250, 200:200+250, :] 

            cv2.imshow('image collection', frame)
            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

        return ret, frame 




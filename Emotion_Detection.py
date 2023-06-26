import time

from kivymd.app import MDApp
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRaisedButton
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from keras.utils import img_to_array
from keras.models import model_from_json
from kivymd.uix.label import MDLabel

model = model_from_json(open("Resources/facial_expression_model_structure.json", 'r').read())
model.load_weights("Resources/facial_expression_model_weights.h5")

Window.size = (360, 600)


class CameraApp(MDApp):

    def build(self):
        self.count = 0
        screen = Screen()
        layout = MDBoxLayout(orientation="vertical")
        self.image = Image()
        self.label = MDLabel(halign="center",
                             theme_text_color="Custom",
                             text_color=(0, 1, 0, 1),
                             font_style="H3")
        self.face_Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        button = MDRaisedButton(text="Click here",
                                pos_hint={"center_x": 0.5, "center_y": 0.5},
                                size_hint=(None, None),
                                on_press=self.take_picture,
                                on_release=self.take_picture_helper)

        layout.add_widget(self.image)
        layout.add_widget(self.label)
        layout.add_widget(button)
        screen.add_widget(layout)

        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0 / 60.0)

        return screen

    def load_video(self, *args):
        emotions = ("angry", "disgust", "fear", "happy", "sad", "suprise", "neutral")
        _, img = self.cap.read()
        self.frame = img
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = self.face_Cascade.detectMultiScale(imgGray, 1.5, 4)
        for (x, y, w, h) in face:
            roi_gray = imgGray[y: y + h, x: x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            img_pixels /= 255
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotion = emotions[max_index]
            print(emotion)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(img, emotion, (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        buffer = cv2.flip(img, 0).tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buffer, colorfmt="bgr", bufferfmt="ubyte")
        self.image.texture = texture

    def take_picture(self, *args):
        image_id = "picture " + str(self.count) + ".jpg"
        cv2.imwrite("Camera_Pictures/" + image_id, self.frame)
        self.label.text = "Picture Saved"
        self.count += 1

    def take_picture_helper(self, *args):
        time.sleep(0.5)
        self.label.text = ""


if __name__ == '__main__':
    CameraApp().run()

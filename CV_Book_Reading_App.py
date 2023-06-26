import pytesseract
from kivymd.app import MDApp
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2


Window.size = (360, 600)
screen_helper = """
Screen:
    MDBoxLayout:
        orientation: "vertical"

        Image:
            source: "Dracaufeu.png"
        MDRaisedButton:
            text: "Click here"
            pos_hint: {"center_x":0.5, "center_y":0.5}
            size_hint: (None, None)


"""


class CameraApp(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.theme_style = "Dark"
        self.count = 0
        screen = Screen()
        layout = MDBoxLayout(orientation="vertical")
        self.image = Image()
        self.label = MDLabel()
        button = MDRaisedButton(text="Click here",
                                pos_hint={"center_x": 0.5, "center_y": 0.5},
                                size_hint=(None, None),
                                on_release=self.take_picture)

        layout.add_widget(self.image)
        layout.add_widget(self.label)
        layout.add_widget(button)
        screen.add_widget(layout)

        self.cap = cv2.VideoCapture("http://192.168.0.23:8080/video")
        Clock.schedule_interval(self.load_video, 1.0 / 60.0)

        return screen

    def load_video(self, *args):
        _, img = self.cap.read()
        self.frame = img
        buffer = cv2.flip(img, 0).tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buffer, colorfmt="bgr", bufferfmt="ubyte")
        self.image.texture = texture

    def take_picture(self, *args):
        image_id = "picture " + str(self.count) + ".jpg"
        imgGray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
        imgThreshold = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        text_data = pytesseract.image_to_string(imgThreshold, lang="eng", config="--oem 3 --psm 6")
        self.label.text = text_data
        cv2.imwrite("Camera_Pictures/" + image_id, imgThreshold)
        self.count += 1


if __name__ == '__main__':
    CameraApp().run()

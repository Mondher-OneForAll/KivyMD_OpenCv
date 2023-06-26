from kivymd.app import MDApp
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRaisedButton
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
        self.count = 0
        screen = Screen()
        layout = MDBoxLayout(orientation="vertical")
        self.image = Image()
        button = MDRaisedButton(text="Click here",
                                pos_hint={"center_x": 0.5, "center_y": 0.5},
                                size_hint=(None, None),
                                on_release=self.take_picture)

        layout.add_widget(self.image)
        layout.add_widget(button)
        screen.add_widget(layout)

        self.cap = cv2.VideoCapture(0)
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
        cv2.imwrite("Camera_Pictures/" + image_id, self.frame)
        self.count += 1


if __name__ == '__main__':
    CameraApp().run()

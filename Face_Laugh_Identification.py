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
        self.smile_Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.face_Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = self.face_Cascade.detectMultiScale(imgGray, 1.5, 4)
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

            roi_Gray = imgGray[y: y + h, x: x + w]
            roi_Color = img[y: y + h, x: x + w]
            smile = self.smile_Cascade.detectMultiScale(roi_Gray, 5, 6)
            for (x, y, w, h) in smile:
                cv2.rectangle(roi_Color, (x, y), (x + w, y + h), (0, 255, 0), 3)

        buffer = cv2.flip(img, 0).tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buffer, colorfmt="bgr", bufferfmt="ubyte")
        self.image.texture = texture

    def take_picture(self, *args):
        image_id = "picture " + str(self.count) + ".jpg"
        cv2.imwrite("Camera_Pictures/" + image_id, self.frame)
        cv2.rectangle(self.frame, (0, 50), (300, 100), (0, 255, 0), cv2.FILLED)
        cv2.putText(self.frame, "Scan Saved", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.waitKey(500)
        self.count += 1


if __name__ == '__main__':
    CameraApp().run()

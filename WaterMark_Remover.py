from kivymd.app import MDApp
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRaisedButton
from kivy.graphics.texture import Texture
import cv2

Window.size = (360, 600)


class WaterMarkRemoverApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        screen = Screen()
        layout = MDBoxLayout(orientation="vertical")
        self.image = Image(source="Resources/BG.png")
        button = MDRaisedButton(text="Click here",
                                pos_hint={"center_x": 0.5, "center_y": 0.5},
                                size_hint=(None, None),
                                on_release=self.remove_WaterMark)

        layout.add_widget(self.image)
        layout.add_widget(button)
        screen.add_widget(layout)
        return screen

    def remove_WaterMark(self, *args):
        src = cv2.imread(self.image.source)
        mask = cv2.imread("Resources/WM.png", cv2.IMREAD_GRAYSCALE)  # put here the WaterMark to remove
        (h, w, _) = src.shape
        mask = cv2.resize(mask, (w, h))
        dest = cv2.inpaint(src, mask, 3, cv2.INPAINT_NS)
        buffer = cv2.flip(dest, 0).tobytes()
        texture = Texture.create(size=(dest.shape[1], dest.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buffer, colorfmt="bgr", bufferfmt="ubyte")
        self.image.texture = texture


if __name__ == '__main__':
    WaterMarkRemoverApp().run()

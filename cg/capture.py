import cv2

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import NumericProperty, StringProperty, BoundedNumericProperty
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle

from dataset_capture import DatasetCapture

class CaptureApp(App):

    def build(self):
        device = 0
        self.capture = DatasetCapture(device)
        objpath = "obj1 v1.obj"
        self.capture.add_object(objpath)
        self.layout = BoxLayout(orientation="horizontal")

        capture = self.capture

        class ImageWindow(Widget):
            before_pos = (0.0, 0.0)

            def on_touch_down(self, prop):
                self.before_pos = prop.pos
            
            def on_touch_move(self, prop):
                delta_pos = [(p - bp)/100 for p, bp in zip(prop.pos, self.before_pos)]
                capture.change_campos(delta_pos)
                self.before_pos = prop.pos

        self.w = ImageWindow()
        self.start_button = Button(text="start", font_size=24, on_press=self.start_update_image, size_hint=[1, 0.2])
        self.layout.add_widget(self.w)
        self.layout.add_widget(self.start_button)
        return self.layout
    
    def start_update_image(self, prop):
        self.evt = Clock.schedule_interval(self.update_image, 0.1)
        stop_button = Button(text="stop", font_size=24, on_press=lambda x: self.evt.cancel())
        self.layout.add_widget(stop_button)
    
    def update_image(self, dt):
        image = self.capture.get_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 0)
        texture = Texture.create(size=(image.shape[1], image.shape[0]))
        texture.blit_buffer(image.tostring())
        with self.w.canvas:
            Rectangle(texture=texture, pos=(0, 0), size=(image.shape[1], image.shape[0]))
    



def main():
    CaptureApp().run()

if __name__ == "__main__":
    main()
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import NumericProperty, StringProperty, BoundedNumericProperty
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.lang import Builder

class MyLayout(BoxLayout):
    pass

class ReduceButton(Button):
    def on_press(self):
        if self.parent.orientation == "horizontal":
            new_orientation = "vertical"
        else:
            new_orientation = "horizontal"
        layout = BoxLayout(orientation=new_orientation)
        layout.add_widget(ReduceButton())
        layout.add_widget(ReduceButton())
        lbl = Label(text="hollow world.", font_size=24)
        layout.add_widget(lbl)
        self.parent.add_widget(layout)
        self.parent.remove_widget(self)

class MuApp(App):

    def build(self):
        a = Builder.load_file("kv.kv")
        layout = BoxLayout(orientation="horizontal")
        button1 = Button(text="hollow", font_size=30)
        # button1.width = 400
        button1.pos_hint = {"x": 0.5, "top": 0.8}
        button2 = Button(text="hollow", font_size=30)
        # button2.width = 100
        button2.size_hint_x = 0.3
        button2.size_hint_y = 0.3
        layout.add_widget(button1)
        layout.add_widget(button2)
        layout.add_widget(a)
        self.time = 0
        self.lbl = Label(text=str(self.time), font_size=24)
        layout.add_widget(self.lbl)
        self.evt = Clock.schedule_interval(self.increase, 1.0)
        return layout
    
    def increase(self, dt):
        self.time += dt
        self.lbl.text = str(self.time)
        if self.time > 10:
            self.evt.cancel()

def main():
    a = Builder.load_file("kv.kv")
    MuApp().run()

if __name__ == "__main__":
    main()
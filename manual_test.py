from manim import *
from manim_vision import ManimVision

class NoCollisionScene(Scene):
    def construct(self):
        ManimVision.monitor(self)
        circle = Circle().shift(LEFT * 3)
        square = Square().shift(RIGHT * 3)
        self.add(circle, square)
        self.wait(1)
        ManimVision.shutdown(self)

class DirectOverlapScene(Scene):
    def construct(self):
        ManimVision.monitor(self)
        circle = Circle()
        square = Square()  # both centered at origin
        self.add(circle, square)
        self.wait(1)
        ManimVision.shutdown(self)

class ShiftCollisionScene(Scene):
    def construct(self):
        ManimVision.monitor(self)
        circle = Circle().shift(LEFT * 3)
        square = Square()
        self.add(circle, square)
        self.play(circle.animate.shift(RIGHT * 3))  # moves into square
        self.wait(1)
        ManimVision.shutdown(self)
class PerformanceScene(Scene):
    def construct(self):
        ManimVision.monitor(self)
        objects = [Circle(radius=0.3).shift(RIGHT * i * 0.4) for i in range(10)]
        for obj in objects:
            self.add(obj)
        self.play(*[obj.animate.shift(UP) for obj in objects])
        self.wait(1)
        ManimVision.shutdown(self)
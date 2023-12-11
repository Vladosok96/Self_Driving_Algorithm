import linalg


class CAMERA:
    def __init__(self, position=linalg.POINT(0, 0, 0)):
        self.position = position
        self.is_up = False
        self.is_left = False
        self.is_right = False
        self.is_down = False


class PLAYER:

    def __init__(self, position=linalg.POINT(0, 0, 0), velocity=0):
        self.position = position
        self.velocity = velocity
        self.vector = linalg.VECTOR2(0, 0)
        self.steering_angle = 0.0
        self.is_w = False
        self.is_a = False
        self.is_s = False
        self.is_d = False
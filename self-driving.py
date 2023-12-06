import pygame
import math
import numpy as np
import PySimpleGUI as sg
import threading
from time import time
import ReedsShepp
# import qlearning

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def det(arr):
    return (arr[0][0] * arr[1][1]) - (arr[0][1] * arr[1][0])


def vector_angle(vector):
    return math.atan2(vector.y, vector.x)


class POINT:

    def __init__(self, x, y, angle=0, direction = 1):
        self.x = x
        self.y = y
        self.angle = angle
        self.direction = direction


def distance(A, B):
    return math.sqrt((A.x - B.x) ** 2 + (A.y - B.y) ** 2)


def is_between(A, B, C):
    a = distance(B, C)
    b = distance(C, A)
    c = distance(A, B)
    return a ** 2 + b ** 2 >= c ** 2 and a ** 2 + c ** 2 >= b ** 2


class STRAIGHT:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def intersection_bool(self, x3, y3, x4, y4):
        try:
            if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
                Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (
                            x3 * y4 - y3 * x4)) / ((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
            else:
                Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (
                            x3 * y4 - y3 * x4)) / 100

            if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
                Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (y3 - y4) - (self.y1 - self.y2) * (
                            x3 * y4 - y3 * x4)) / ((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
            else:
                Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (
                            x3 * y4 - y3 * x4)) / 100
        except:
            return False

        if is_between(POINT(Px, Py), POINT(x3, y3), POINT(x4, y4)) and is_between(POINT(Px, Py),
                                                                                  POINT(self.x1, self.y1),
                                                                                  POINT(self.x2, self.y2)):
            return True
        else:
            return False

    def intersection(self, x3, y3, x4, y4):
        if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
            Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (x3 * y4 - y3 * x4)) / (
                        (self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
        else:
            Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (x3 * y4 - y3 * x4)) / 100

        if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
            Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 * y4 - y3 * x4)) / (
                        (self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
        else:
            Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (x3 * y4 - y3 * x4)) / 100

        if is_between(POINT(Px, Py), POINT(x3, y3), POINT(x4, y4)) and is_between(POINT(Px, Py),
                                                                                  POINT(self.x1, self.y1),
                                                                                  POINT(self.x2, self.y2)):
            return POINT(Px, Py, int(distance(POINT(x3, y3), POINT(Px, Py))))
        else:
            return POINT(x4, y4, int(distance(POINT(x3, y3), POINT(x4, y4))))


class VECTOR2:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def median(self, x, y, k):
        self.x = (self.x - x) * k + x
        self.y = (self.y - y) * k + y
        return VECTOR2(self.x, self.y)

    def get_length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # reference = VECTOR2(1, 0)

    def AngleOfReference(self, v):
        return self.NormalizeAngle(math.atan2(v.y, v.x) / math.pi * 180)

    def AngleOfVectors(self, first, second):
        return self.NormalizeAngle(self.AngleOfReference(first) - self.AngleOfReference(second))

    def NormalizeAngle(self, angle):
        if angle > -180:
            turn = -360
        else:
            turn = 360
        while not (angle > -180 and angle <= 180):
            angle += turn
        return angle

    def rotate(self, radians):
        x_prime = self.x * math.cos(radians) - self.y * math.sin(radians)
        y_prime = self.x * math.sin(radians) + self.y * math.cos(radians)
        return VECTOR2(x_prime, y_prime)

    def mult(self, scalar):
        return VECTOR2(self.x * scalar, self.y * scalar)

    def __add__(self, other):
        return VECTOR2(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}"


class CAMERA:
    def __init__(self, position=POINT(0, 0, 0)):
        self.position = position
        self.is_up = False
        self.is_left = False
        self.is_right = False
        self.is_down = False


class PLAYER:

    def __init__(self, position=POINT(0, 0, 0), velocity=0):
        self.position = position
        self.velocity = velocity
        self.vector = VECTOR2(0, 0)
        self.steering_angle = 0.0
        self.is_w = False
        self.is_a = False
        self.is_s = False
        self.is_d = False


def solve_angle(angle, add):
    angle += add
    angle %= 360
    return angle


def to_radians(angle):
    angle = float(angle - 90)
    angle = (angle / 180) * 3.14159
    return angle


def blit_rotate(surf, image, pos, originPos, angle):
    angle = -angle
    # calculate the axis aligned bounding box of the rotated image
    w, h = image.get_size()
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # calculate the translation of the pivot
    pivot = pygame.math.Vector2(originPos[0], -originPos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - originPos[0] + min_box[0] - pivot_move[0], pos[1] - originPos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)

    # rotate and blit the image
    surf.blit(rotated_image, origin)


# параметры окна
size = (1200, 800)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
screen.fill((200, 200, 200))
pygame.display.set_caption("Initial N")
pygame.mixer.init()
runGame = True

# изображения
trueno = pygame.image.load("trueno.png")

# Следование до точки
FIRST_POINT_CF = 0.75
STEERING_CF = 2.5
NEXT_POINT_OFFSET = 20
destinations = []
high_destinations = []
is_achieved = True
is_reverse = False
destination = POINT(0, 0, 0)
departure = POINT(0, 0, 0)
destination_angle = 0
pygame.font.init()
heading_font = pygame.font.SysFont("Arial", 20)
current_vector_point = None     # Start direction vector
current_wall_point = None       # Wall first point


# Машинное обучение
class QLearningCar:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.memory = []
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return np.random.uniform(-1, 1, self.action_size)
        q_values = self.model.predict(state)
        return q_values[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0] = action * target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


state_size = 4  # координаты целевой точки (x, y), скорость и угол поворота руля
action_size = 2  # целевой угол поворота руля и целевая скорость
batch_size = 32
# Создаем экземпляр Q-обучения
agent = QLearningCar(state_size, action_size)

# пременные
fps = 60
player = PLAYER(POINT(300, 300, 0), 0)
camera = CAMERA(POINT(0, 0, 0))
show_lines = 1

# карта стен
walls = []

# Путь
last_add_way = 0
way = []


def research_destinations():
    global high_destinations

    high_destinations = []
    path_x, path_y, yaw, directions = [], [], [], []
    tmp_low_destintaions = destinations.copy()

    tmp_low_destintaions.insert(0, POINT(player.position.x,
                                         player.position.y,
                                         angle=np.deg2rad(player.position.angle + 90),
                                         direction=-1))

    for i in range(len(tmp_low_destintaions) - 1):
        s_x = tmp_low_destintaions[i].x
        s_y = tmp_low_destintaions[i].y
        s_yaw = tmp_low_destintaions[i].angle
        g_x = tmp_low_destintaions[i + 1].x
        g_y = tmp_low_destintaions[i + 1].y
        g_yaw = tmp_low_destintaions[i + 1].angle

        path_i = ReedsShepp.calc_optimal_path(s_x, s_y, s_yaw,
                                   g_x, g_y, g_yaw, ReedsShepp.max_c, ReedsShepp.STEP_SIZE)

        for j in range(len(path_i.x)):
            high_destinations.append(POINT(path_i.x[j],
                                           path_i.y[j],
                                           angle=path_i.yaw[j] + 90,
                                           direction=-path_i.directions[j]))


# Окно системы проектирования
layout = [[sg.Text('Автомобиль')],
          [sg.Text('Ширина'), sg.InputText(default_text=ReedsShepp.max_c)],
          [sg.Text('Длина'), sg.InputText(default_text=ReedsShepp.max_c)],
          [sg.HorizontalSeparator()],
          [sg.Text('Алгоритм следования')],
          [sg.Text('Соотношение значимости'), sg.Slider(orientation='h', range=(0, 1), resolution=0.05, default_value=0.5)],
          [sg.Text('Коэффициент руления'), sg.InputText(default_text=STEERING_CF)],
          [sg.Text('Дальность второй точки'), sg.Slider(orientation='h', range=(2, 50), resolution=1, default_value=NEXT_POINT_OFFSET)],
          [sg.HorizontalSeparator()],
          [sg.Text('Алгоритм построения пути')],
          [sg.Text('Минимальный радиус'), sg.InputText(default_text=ReedsShepp.max_c)],
          [sg.Text('Шаг дискретизации'), sg.InputText(default_text=ReedsShepp.STEP_SIZE)],
          [sg.Button('Ok'), sg.Button('Cancel')] ]


def CAD_window():
    global runGame
    global FIRST_POINT_CF
    global STEERING_CF
    global NEXT_POINT_OFFSET
    window = sg.Window('Система проектирования БПТС', layout)
    last_update = {0: '0', 1: '0', 3: '0', 4: '0', 5: '0', 7: '0', 8: '0'}

    while runGame:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            runGame = False
            break

        FIRST_POINT_CF = float(values[3])
        try:
            if float(values[4]) > 0:
                STEERING_CF = float(values[4])
        except:
            pass
        NEXT_POINT_OFFSET = int(values[5])
        try:
            if float(values[7]) > 0:
                ReedsShepp.max_c = float(values[7])
        except:
            pass
        try:
            if float(values[8]) != 0:
                ReedsShepp.STEP_SIZE = float(values[8])
        except:
            pass

        if last_update[7] != values[7] or last_update[8] != values[8]:
            research_destinations()
            last_update = values
            print(values)


CAD_window_task = threading.Thread(target=CAD_window, args=())
CAD_window_task.start()

# отрисовка
while runGame:

    # Данные, относящиеся к Q-обучению
    reward = 1.0
    done = True

    # Создание текущего состояния среды для модели Q-обучения
    current_state = [0, 0, player.steering_angle, player.velocity]
    if len(high_destinations) > 0:

        # Рассчет вектора до точек
        if len(high_destinations) > NEXT_POINT_OFFSET:
            # Векторы, указывающие на напрвления до следующих точек
            first_point_vector = VECTOR2(high_destinations[0].x - player.position.x,
                                         high_destinations[0].y - player.position.y) \
                .rotate(math.radians(player.position.angle))
            second_point_vector = VECTOR2(high_destinations[NEXT_POINT_OFFSET].x - high_destinations[0].x,
                                          high_destinations[NEXT_POINT_OFFSET].y - high_destinations[0].y) \
                .rotate(math.radians(player.position.angle))

            current_state[0] = second_point_vector.x
            current_state[1] = second_point_vector.y
            current_state = np.array(current_state)
            current_state = np.reshape(current_state, [1, state_size])

        done = False

        # Проверка на сход с маршрута
        if VECTOR2(player.position.x - high_destinations[0].x,
                   player.position.y - high_destinations[0].y).get_length() > 50:
            research_destinations()
            reward -= 1

        # Удаление достигнутых точек
        if len(destinations) > 0:
            while len(destinations) > 0 and VECTOR2(player.position.x - destinations[0].x, player.position.y - destinations[0].y).get_length() < 20:
                destinations.pop(0)
        while not is_achieved and VECTOR2(player.position.x - high_destinations[0].x,
                                          player.position.y - high_destinations[0].y).get_length() < 10:
            high_destinations.pop(0)
            departure = POINT(player.position.x, player.position.y, 0)
            reward += 0.1
            if len(high_destinations) == 0:
                is_achieved = True

    # Отслеживание событий для движения
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_reverse = False
                current_vector_point = POINT(pygame.mouse.get_pos()[0] - camera.position.x,
                                             pygame.mouse.get_pos()[1] - camera.position.y, direction = 1)

                departure = POINT(player.position.x, player.position.y, 0)
            if event.button == 2:
                destinations.clear()
                high_destinations.clear()
                is_achieved = True
            if event.button == 3:
                current_wall_point = POINT(pygame.mouse.get_pos()[0] - camera.position.x,
                                           pygame.mouse.get_pos()[1] - camera.position.y, direction=1)

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if current_vector_point != None:
                    current_vector_point.angle = vector_angle(VECTOR2(current_vector_point.x - (pygame.mouse.get_pos()[0] - camera.position.x),
                                                                      current_vector_point.y - (pygame.mouse.get_pos()[1] - camera.position.y)))
                    destinations.append(current_vector_point)
                    current_vector_point = None
            if event.button == 3:
                if current_wall_point != None:
                    walls.append(STRAIGHT(current_wall_point.x,
                                          current_wall_point.y,
                                          pygame.mouse.get_pos()[0] - camera.position.x,
                                          pygame.mouse.get_pos()[1] - camera.position.y))
                    current_wall_point = None

        if event.type == pygame.KEYDOWN:
            if event.key == 99:
                is_achieved = False
            elif event.key == 118:
                is_achieved = True
            elif event.key == 27:
                runGame = False
            elif event.key == 108:
                if show_lines == 3:
                    show_lines = 0
                else:
                    show_lines += 1
            elif event.key == 122:
                walls = []
            elif event.key == 98:
                way = []
            elif event.key == 100:
                player.is_d = True
            elif event.key == 97:
                player.is_a = True
            elif event.key == 119:
                player.is_w = True
            elif event.key == 115:
                player.is_s = True

            elif event.key == 1073741904:
                camera.is_left = True
            elif event.key == 1073741906:
                camera.is_up = True
            elif event.key == 1073741903:
                camera.is_right = True
            elif event.key == 1073741905:
                camera.is_down = True

            elif event.key == 120:
                research_destinations()

        if event.type == pygame.KEYUP:
            if event.key == 27:
                runGame = False
            elif event.key == 100:
                player.is_d = False
            elif event.key == 97:
                player.is_a = False
            elif event.key == 119:
                player.is_w = False
            elif event.key == 115:
                player.is_s = False

            elif event.key == 1073741904:
                camera.is_left = False
            elif event.key == 1073741906:
                camera.is_up = False
            elif event.key == 1073741903:
                camera.is_right = False
            elif event.key == 1073741905:
                camera.is_down = False

    # Рассчет действий для автомобиля
    action = agent.act(current_state)
    destination_angle = action[0]
    velocity = action[1]


    # Применение событий для машины
    if not is_achieved:
        if not is_reverse:
            player.velocity = velocity
            player.steering_angle = max(min(destination_angle * STEERING_CF, 5), -5)
        else:
            if player.velocity > -2:
                player.velocity -= 1 * (1 / 2)
            player.steering_angle = -max(min(destination_angle * STEERING_CF, 5), -5)
    else:
        if not is_reverse:
            if player.velocity > 1:
                player.velocity -= 0.5
        else:
            if player.velocity < 0:
                player.velocity += 0.5
        player.steering_angle = 0

    # Применение событий для камеры
    if camera.is_left:
        camera.position.x += 20
    if camera.is_right:
        camera.position.x -= 20
    if camera.is_up:
        camera.position.y += 20
    if camera.is_down:
        camera.position.y -= 20

    ## перемещение

    # Слежение камеры за машиной
    # camera.position.x -= (camera.position.x + player.position.x - 600) * 0.15 * (1 / 2)
    # camera.position.y -= (camera.position.y + player.position.y - 400) * 0.15 * (1 / 2)

    player.position.angle = solve_angle(player.position.angle,
                                        player.steering_angle * (player.vector.get_length() / 10))

    direction_vector = VECTOR2(math.cos(to_radians(player.position.angle)) * player.velocity * (1 / 2),
                               math.sin(to_radians(player.position.angle)) * player.velocity * (1 / 2))
    player.vector.median(direction_vector.x, direction_vector.y, 0.7)

    player.position.x += player.vector.x
    player.position.y += player.vector.y

    # притормаживание
    if player.velocity > 0:
        player.velocity -= 0.25 * (1 / 2)
    # Возврат руля
    if (not player.is_d) and (not player.is_a) and player.steering_angle > 0:
        player.steering_angle -= 0.1
    if (not player.is_a) and (not player.is_d) and player.steering_angle < 0:
        player.steering_angle += 0.1

    # Рассчет следующего состояния
    next_state = [0, 0, player.steering_angle, player.velocity]
    if len(high_destinations) > NEXT_POINT_OFFSET:
        # Векторы, указывающие на напрвления до следующих точек
        first_point_vector = VECTOR2(high_destinations[0].x - player.position.x,
                                     high_destinations[0].y - player.position.y) \
            .rotate(math.radians(player.position.angle))
        second_point_vector = VECTOR2(high_destinations[NEXT_POINT_OFFSET].x - high_destinations[0].x,
                                      high_destinations[NEXT_POINT_OFFSET].y - high_destinations[0].y) \
            .rotate(math.radians(player.position.angle))

        next_state[0] = second_point_vector.x
        next_state[1] = second_point_vector.y
        next_state = np.array(current_state)
        next_state = np.reshape(current_state, [1, state_size])

    # Сохраняем опыт и обновляем модель
    agent.remember(current_state, action, reward, next_state, done)

    # Обучаем модель на случайном подмножестве опыта
    # agent.replay(batch_size)



    screen.fill((200, 200, 200))
    blit_rotate(screen, trueno, (player.position.x + camera.position.x, player.position.y + camera.position.y),
                (29, 76), player.position.angle)

    point_front = POINT(player.position.x + math.cos(to_radians(player.position.angle)) * 1000,
                        player.position.y + math.sin(to_radians(player.position.angle)) * 1000)
    point_half_left = POINT(player.position.x + math.cos(to_radians(player.position.angle + 35)) * 1000,
                            player.position.y + math.sin(to_radians(player.position.angle + 35)) * 1000)

    point_half_right = POINT(player.position.x + math.cos(to_radians(player.position.angle - 35)) * 1000,
                             player.position.y + math.sin(to_radians(player.position.angle - 35)) * 1000)
    point_left = POINT(player.position.x + math.cos(to_radians(player.position.angle + 90)) * 1000,
                       player.position.y + math.sin(to_radians(player.position.angle + 90)) * 1000)

    point_right = POINT(player.position.x + math.cos(to_radians(player.position.angle - 90)) * 1000,
                        player.position.y + math.sin(to_radians(player.position.angle - 90)) * 1000)

    if (time() - last_add_way > 0.1):
        way.append(POINT(player.position.x, player.position.y))
        last_add_way = time()
    for i in range(1, len(way)):
        pygame.draw.line(screen, pygame.Color(255, 0, 255), [way[i - 1].x + camera.position.x, way[i - 1].y + camera.position.y],
                         [way[i].x + camera.position.x, way[i].y + camera.position.y], 2)

    for wall in walls:
        pygame.draw.line(screen, pygame.Color(255, 0, 0), [wall.x1 + camera.position.x, wall.y1 + camera.position.y],
                         [wall.x2 + camera.position.x, wall.y2 + camera.position.y], 2)

        point_front = wall.intersection(player.position.x, player.position.y,
                                        point_front.x, point_front.y)
        point_half_left = wall.intersection(player.position.x, player.position.y,
                                            point_half_left.x, point_half_left.y)
        point_half_right = wall.intersection(player.position.x, player.position.y,
                                             point_half_right.x, point_half_right.y)
        point_left = wall.intersection(player.position.x, player.position.y,
                                       point_left.x, point_left.y)
        point_right = wall.intersection(player.position.x, player.position.y,
                                        point_right.x, point_right.y)

    if show_lines == 0 or show_lines == 2:
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_front.x + camera.position.x, point_front.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_front.x + camera.position.x, point_front.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_half_left.x + camera.position.x, point_half_left.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_half_left.x + camera.position.x, point_half_left.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_half_right.x + camera.position.x, point_half_right.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_half_right.x + camera.position.x, point_half_right.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_left.x + camera.position.x, point_left.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_left.x + camera.position.x, point_left.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_right.x + camera.position.x, point_right.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_right.x + camera.position.x, point_right.y + camera.position.y), 5.0)

    is_crash = False
    front_corner_l = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle + 20))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle + 20))))
    front_corner_r = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle - 20))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle - 20))))
    rear_corner_l = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle + 160))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle + 160))))
    rear_corner_r = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle - 160))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle - 160))))

    # отрисовка "векторов" направлений

    if current_vector_point != None:
            pygame.draw.line(screen, pygame.Color(255, 255, 0),
                             [current_vector_point.x + camera.position.x, current_vector_point.y + camera.position.y],
                             [pygame.mouse.get_pos()[0],
                              pygame.mouse.get_pos()[1]], 3)

    if current_wall_point != None:
            pygame.draw.line(screen, pygame.Color(0, 255, 100),
                             [current_wall_point.x + camera.position.x, current_wall_point.y + camera.position.y],
                             [pygame.mouse.get_pos()[0],
                              pygame.mouse.get_pos()[1]], 3)

    # Отрисовка линий путей
    if show_lines == 1 or show_lines == 2:

        angle_vector = VECTOR2(math.cos(to_radians(player.position.angle)) * 10 * (1 / 2),
                               math.sin(to_radians(player.position.angle)) * 10 * (1 / 2))

        # отрисовка низкодискретизированного пути
        if len(destinations) > 0:
            pygame.draw.line(screen, pygame.Color(255, 255, 0),
                             [player.position.x + camera.position.x, player.position.y + camera.position.y],
                             [destinations[0].x + camera.position.x,
                              destinations[0].y + camera.position.y], 3)
            for destination_iterator in range(1, len(destinations)):
                pygame.draw.line(screen, pygame.Color(150, 150, 150),
                                 [destinations[destination_iterator].x + camera.position.x,
                                  destinations[destination_iterator].y + camera.position.y],
                                 [destinations[destination_iterator - 1].x + camera.position.x,
                                  destinations[destination_iterator - 1].y + camera.position.y], 3)
            while len(destinations) > 0 and VECTOR2(player.position.x - destinations[0].x, player.position.y - destinations[0].y).get_length() < 20:
                destinations.pop(0)

        # отрисовка высокодискретизированного пути
        if len(high_destinations) > 0:
            for destination_iterator in range(1, len(high_destinations)):
                if high_destinations[destination_iterator].direction == 1:
                    pygame.draw.line(screen, pygame.Color(255, 150, 150),
                                     [high_destinations[destination_iterator].x + camera.position.x,
                                      high_destinations[destination_iterator].y + camera.position.y],
                                     [high_destinations[destination_iterator - 1].x + camera.position.x,
                                      high_destinations[destination_iterator - 1].y + camera.position.y], 3)
                elif high_destinations[destination_iterator].direction == -1:
                    pygame.draw.line(screen, pygame.Color(150, 150, 255),
                                     [high_destinations[destination_iterator].x + camera.position.x,
                                      high_destinations[destination_iterator].y + camera.position.y],
                                     [high_destinations[destination_iterator - 1].x + camera.position.x,
                                      high_destinations[destination_iterator - 1].y + camera.position.y], 3)

    pygame.draw.circle(screen, pygame.Color(255, 0, 0),
                       [destination.x + camera.position.x, destination.y + camera.position.y], 5)

    for wall in walls:
        if wall.intersection_bool(front_corner_l.x, front_corner_l.y,
                                  front_corner_r.x, front_corner_r.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [front_corner_l.x + camera.position.x, front_corner_l.y + camera.position.y],
                             [front_corner_r.x + camera.position.x, front_corner_r.y + camera.position.y], 3)'''

        if wall.intersection_bool(front_corner_l.x, front_corner_l.y,
                                  rear_corner_l.x, rear_corner_l.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [front_corner_l.x + camera.position.x, front_corner_l.y + camera.position.y],
                             [rear_corner_l.x + camera.position.x, rear_corner_l.y + camera.position.y], 3)'''

        if wall.intersection_bool(rear_corner_r.x, rear_corner_r.y,
                                  front_corner_r.x, front_corner_r.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [rear_corner_r.x + camera.position.x, rear_corner_r.y + camera.position.y],
                             [front_corner_r.x + camera.position.x, front_corner_r.y + camera.position.y], 3)'''

        if wall.intersection_bool(rear_corner_r.x, rear_corner_r.y,
                                  rear_corner_l.x, rear_corner_l.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [rear_corner_r.x + camera.position.x, rear_corner_r.y + camera.position.y],
                             [rear_corner_l.x + camera.position.x, rear_corner_l.y + camera.position.y], 3)'''

        if is_crash:
            player = PLAYER(POINT(300, 300, 0), 0)

    pygame.display.update()

    clock.tick(fps)

pygame.quit()

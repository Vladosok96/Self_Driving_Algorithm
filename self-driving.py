import pygame
import math
import numpy as np
import PySimpleGUI as sg
import threading
from time import time
import ReedsShepp
import qlearning
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
STEERING_CF = 3.5
NEXT_POINT_OFFSET = 20
destinations = []
high_destinations = []
is_achieved = True
is_reverse = False
destination = linalg.POINT(0, 0, 0)
departure = linalg.POINT(0, 0, 0)
destination_angle = 0
pygame.font.init()
heading_font = pygame.font.SysFont("Arial", 20)
current_vector_point = None     # Start direction vector
current_wall_point = None       # Wall first point


state_size = 4  # координаты целевой точки (x, y), скорость и угол поворота руля
action_size = 2  # целевой угол поворота руля и целевая скорость
batch_size = 32
agent = qlearning.DQNAgent(state_size, action_size) # Создаем экземпляр Q-обучения
state = [0, 0, 0, 0]
total_reward = 0
done = False
counter = 0
begining_position = linalg.POINT(0, 0, 0)
fit_counter = 0


# пременные
fps = 60
player = PLAYER(linalg.POINT(300, 300, 0), 0)
camera = CAMERA(linalg.POINT(0, 0, 0))
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

    tmp_low_destintaions.insert(0, linalg.POINT(player.position.x,
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
            high_destinations.append(linalg.POINT(path_i.x[j],
                                           path_i.y[j],
                                           angle=path_i.yaw[j] + 90,
                                           direction=-path_i.directions[j]))
    high_destinations = high_destinations[15:]


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


CAD_window_task = threading.Thread(target=CAD_window, args=())
CAD_window_task.start()

# отрисовка
while runGame:

    # Данные, относящиеся к Q-обучению
    reward = 0
    next_state = state.copy()
    action = agent.act(state)
    out_of_way = False

    if len(high_destinations) > 0:

        if high_destinations[0].direction == 1:
            reward += action[1]
        else:
            reward -= action[1]

        # Рассчет вектора до точек
        if len(high_destinations) > NEXT_POINT_OFFSET:
            point_vector = linalg.VECTOR2(high_destinations[NEXT_POINT_OFFSET].x - player.position.x,
                                          high_destinations[NEXT_POINT_OFFSET].y - player.position.y) \
                .rotate(math.radians(player.position.angle))

            next_state[0] = point_vector.x
            next_state[1] = -point_vector.y

        done = False

        # Проверка на сход с маршрута
        if linalg.VECTOR2(player.position.x - high_destinations[0].x,
                   player.position.y - high_destinations[0].y).get_length() > 50:
            reward -= 5
            out_of_way = True

        # Удаление достигнутых точек
        if len(destinations) > 0:
            while len(destinations) > 0 and linalg.VECTOR2(player.position.x - destinations[0].x, player.position.y - destinations[0].y).get_length() < 20:
                destinations.pop(0)
                begining_position = linalg.POINT(player.position.x, player.position.y, player.position.angle)
        while not is_achieved and linalg.VECTOR2(player.position.x - high_destinations[0].x,
                                          player.position.y - high_destinations[0].y).get_length() < 10:
            high_destinations.pop(0)
            departure = linalg.POINT(player.position.x, player.position.y, 0)
            reward += 0.5
            if len(high_destinations) == 0:
                is_achieved = True

    # Отслеживание событий для движения
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_reverse = False
                current_vector_point = linalg.POINT(pygame.mouse.get_pos()[0] - camera.position.x,
                                             pygame.mouse.get_pos()[1] - camera.position.y, direction = 1)

                departure = linalg.POINT(player.position.x, player.position.y, 0)
            if event.button == 2:
                destinations.clear()
                high_destinations.clear()
                is_achieved = True
            if event.button == 3:
                current_wall_point = linalg.POINT(pygame.mouse.get_pos()[0] - camera.position.x,
                                           pygame.mouse.get_pos()[1] - camera.position.y, direction=1)

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if current_vector_point != None:
                    current_vector_point.angle = linalg.vector_angle(linalg.VECTOR2(current_vector_point.x - (pygame.mouse.get_pos()[0] - camera.position.x),
                                                                      current_vector_point.y - (pygame.mouse.get_pos()[1] - camera.position.y)))
                    destinations.append(current_vector_point)
                    current_vector_point = None
            if event.button == 3:
                if current_wall_point != None:
                    walls.append(linalg.STRAIGHT(current_wall_point.x,
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
                begining_position = linalg.POINT(player.position.x, player.position.y, player.position.angle)
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

    destination_angle = action[0]
    velocity = action[1] / 4

    # Применение событий для машины
    if not is_achieved:
        player.velocity = velocity
        if velocity >= 0:
            player.steering_angle = max(min(destination_angle * 5, 3.5), -5)
        else:
            player.steering_angle = -max(min(destination_angle * 5, 3.5), -5)
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

    direction_vector = linalg.VECTOR2(math.cos(to_radians(player.position.angle)) * player.velocity * (1 / 2),
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

    # Сохраняем опыт и обновляем модель
    agent.remember(state, action, reward, next_state, done)
    state = next_state.copy()

    # Обучаем модель на случайном подмножестве опыта
    if out_of_way:
        player.position = linalg.POINT(begining_position.x, begining_position.y, begining_position.angle)
        player.vector = linalg.VECTOR2(0, 0)
        player.velocity = 0
        research_destinations()
        counter += 1
        if counter > 5:
            agent.replay(batch_size)
            counter = 0

    fit_counter += 1
    if fit_counter > 1000:
        agent.replay(batch_size)
        fit_counter = 0

    screen.fill((200, 200, 200))
    blit_rotate(screen, trueno, (player.position.x + camera.position.x, player.position.y + camera.position.y),
                (29, 76), player.position.angle)

    point_front = linalg.POINT(player.position.x + math.cos(to_radians(player.position.angle)) * 1000,
                        player.position.y + math.sin(to_radians(player.position.angle)) * 1000)
    point_half_left = linalg.POINT(player.position.x + math.cos(to_radians(player.position.angle + 35)) * 1000,
                            player.position.y + math.sin(to_radians(player.position.angle + 35)) * 1000)

    point_half_right = linalg.POINT(player.position.x + math.cos(to_radians(player.position.angle - 35)) * 1000,
                             player.position.y + math.sin(to_radians(player.position.angle - 35)) * 1000)
    point_left = linalg.POINT(player.position.x + math.cos(to_radians(player.position.angle + 90)) * 1000,
                       player.position.y + math.sin(to_radians(player.position.angle + 90)) * 1000)

    point_right = linalg.POINT(player.position.x + math.cos(to_radians(player.position.angle - 90)) * 1000,
                        player.position.y + math.sin(to_radians(player.position.angle - 90)) * 1000)

    if (time() - last_add_way > 0.1):
        way.append(linalg.POINT(player.position.x, player.position.y))
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
    front_corner_l = linalg.POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle + 20))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle + 20))))
    front_corner_r = linalg.POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle - 20))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle - 20))))
    rear_corner_l = linalg.POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle + 160))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle + 160))))
    rear_corner_r = linalg.POINT(
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

        angle_vector = linalg.VECTOR2(math.cos(to_radians(player.position.angle)) * 10 * (1 / 2),
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
            while len(destinations) > 0 and linalg.VECTOR2(player.position.x - destinations[0].x, player.position.y - destinations[0].y).get_length() < 20:
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

        if wall.intersection_bool(front_corner_l.x, front_corner_l.y,
                                  rear_corner_l.x, rear_corner_l.y):
            is_crash = True

        if wall.intersection_bool(rear_corner_r.x, rear_corner_r.y,
                                  front_corner_r.x, front_corner_r.y):
            is_crash = True

        if wall.intersection_bool(rear_corner_r.x, rear_corner_r.y,
                                  rear_corner_l.x, rear_corner_l.y):
            is_crash = True

        if is_crash:
            player = PLAYER(linalg.POINT(300, 300, 0), 0)

    pygame.display.update()

    clock.tick(fps)

pygame.quit()

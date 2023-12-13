import pygame
import math
import numpy as np
import PySimpleGUI as sg
import threading
from time import time
import random

import torch

import ReedsShepp
import qlearning
import linalg
import bumpmap
import functions
import structures

import matplotlib.pyplot as plt


# параметры окна
size = (1200, 800)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
screen.fill((200, 200, 200))
pygame.display.set_caption("Initial N")
pygame.mixer.init()
runGame = True

# Шрифты
pygame.font.init()
sysfont = pygame.font.SysFont('Arial', 20)

# изображения
trueno = pygame.image.load("trueno.png")

# Следование до точки
FIRST_POINT_CF = 0.75
STEERING_CF = 3.5
NEXT_POINT_INTERVAL = 50
destinations = []
high_destinations = []
is_achieved = True
is_reverse = False
destination_angle = 0
current_vector_point = None
current_wall_point = None
points_buffer = []
pixel_by_meter = 36.3
difference_angle = 0
next_point_angle = 0
speed_km_h = 0

# Параметры q-обучения
state_size = 9
n_actions = 9
batch_size = 32
state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
n_observations = len(state)
policy_net = qlearning.DQN(n_observations, n_actions).to(qlearning.device)
target_net = qlearning.DQN(n_observations, n_actions).to(qlearning.device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = qlearning.optim.Adam(policy_net.parameters(), lr=qlearning.LR, amsgrad=True)
memory = qlearning.ReplayMemory(10000)
steps_done = 0
done = False
counter = 0
beginning_position = linalg.POINT(0, 0, 0)
beginning_destinations = []
fit_counter = 0
episode_durations = []
is_learning = False

# Карта высот
bump_map = bumpmap.BumpMap(width=2048, height=2048)

# Графики
plot_speeds = []
plot_differences = []
plot_angles = []
plot_times = []
plot_state = 0
plot_update_time = 0
plot_begin_time = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = qlearning.EPS_END + (qlearning.EPS_START - qlearning.EPS_END) * \
        math.exp(-1. * steps_done / qlearning.EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 8)]], device=qlearning.device, dtype=torch.long)


def optimize_model():
    if len(memory) < qlearning.BATCH_SIZE:
        return
    transitions = memory.sample(qlearning.BATCH_SIZE)
    batch = qlearning.Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=qlearning.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(qlearning.BATCH_SIZE, device=qlearning.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * qlearning.GAMMA) + reward_batch

    criterion = qlearning.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Конвертация числа в управляющие команды
def num_control(number):
    if number == 0:
        return [-1, -1]
    if number == 1:
        return [0, -1]
    if number == 2:
        return [1, -1]
    if number == 3:
        return [-1, 0]
    if number == 4:
        return [0, 0]
    if number == 5:
        return [1, 0]
    if number == 6:
        return [-1, 1]
    if number == 7:
        return [0, 1]
    if number == 8:
        return [1, 1]


# пременные
fps = 60
player = structures.PLAYER(linalg.POINT(300, 300, 0), 0)
camera = structures.CAMERA(linalg.POINT(0, 0, 0))
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

        try:
            path_i = ReedsShepp.calc_optimal_path(s_x, s_y, s_yaw,
                                   g_x, g_y, g_yaw, ReedsShepp.max_c, ReedsShepp.STEP_SIZE)
        except:
            return False

        for j in range(len(path_i.x)):
            high_destinations.append(linalg.POINT(path_i.x[j],
                                                  path_i.y[j],
                                                  angle=path_i.yaw[j] + 90,
                                                  direction=-path_i.directions[j]))
    high_destinations = high_destinations[15:]


# Функция рассчета эффективности движения
def engine_efficiency(speed, difference):
    if difference < 0:
        return (-0.0011) * speed * speed + 0.065 * speed
    return (-0.0015) * speed * speed + 0.065 * speed


# Окно системы проектирования
layout = [
    [sg.Text('Автомобиль')],
    [sg.Text('Ширина'), sg.InputText(default_text=ReedsShepp.max_c)],
    [sg.Text('Длина'), sg.InputText(default_text=ReedsShepp.max_c)],
    [sg.HorizontalSeparator()],
    [sg.Text('Алгоритм следования')],
    [sg.Text('Соотношение значимости'), sg.Slider(orientation='h', range=(0, 1), resolution=0.05, default_value=0.5)],
    [sg.Text('Коэффициент руления'), sg.InputText(default_text=STEERING_CF)],
    [sg.Text('Дальность второй точки'), sg.Slider(orientation='h', range=(2, 150), resolution=1, default_value=NEXT_POINT_INTERVAL)],
    [sg.HorizontalSeparator()],
    [sg.Text('Алгоритм построения пути')],
    [sg.Text('Минимальный радиус'), sg.InputText(default_text=ReedsShepp.max_c)],
    [sg.Text('Шаг дискретизации'), sg.InputText(default_text=ReedsShepp.STEP_SIZE)],
    [sg.Button('Ok'), sg.Button('Cancel')],
    [sg.HorizontalSeparator()],
    [sg.Button('Начало графика', key='-begin_plot-'), sg.Button('Конец графика', key='-end_plot-')],
    [sg.HorizontalSeparator()],
    [sg.Button('Включить обучение', key='-learning_switch-')],
    [sg.SaveAs('Сохранить нейросеть', key='-save_net-', file_types=(("Self driving model Files", "*.model"),), change_submits=True)],
    [sg.FileBrowse('Загрузить нейросеть', key='-load_net-', file_types=(("Self driving model Files", "*.model"),), change_submits=True)]
]


def CAD_window():
    global runGame
    global FIRST_POINT_CF
    global STEERING_CF
    global NEXT_POINT_INTERVAL
    global plot_state
    global plot_begin_time
    global plot_speeds
    global plot_differences
    global plot_angles
    global plot_times
    global policy_net
    global target_net
    global optimizer
    global is_learning

    window = sg.Window('Средство проектирования БПТС', layout)
    last_update = {0: '0', 1: '0', 3: '0', 4: '0', 5: '0', 7: '0', 8: '0'}

    while runGame:
        event, values = window.read(timeout=500)
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            runGame = False
            break

        if event == '-begin_plot-':
            plot_begin_time = time()
            plot_state = 1
            plot_speeds = []
            plot_differences = []
            plot_angles = []
            plot_times = []
        if event == '-end_plot-':
            plot_state = 0
            fig, axs = plt.subplots(3)
            axs[0].plot(plot_times, plot_speeds)
            axs[0].set_title('График скорости автомобиля')
            axs[0].set_ylabel('Км/ч')
            axs[0].set_xlabel('с')
            axs[1].plot(plot_times, plot_differences)
            axs[1].set_title('График высоты рельефа')
            axs[1].set_ylabel('M')
            axs[1].set_xlabel('с')
            axs[2].plot(plot_times, plot_angles)
            axs[2].set_title('График отклонения курса автомобиля от маршрута')
            axs[2].set_ylabel('Угол°')
            axs[2].set_xlabel('с')
            plt.show()
        if event == '-learning_switch-':
            if is_learning:
                window['-learning_switch-'].update('Включить обучение')
                is_learning = False
            else:
                window['-learning_switch-'].update('Выключить обучение')
                is_learning = True
        if event == '-save_net-':
            torch.save({
                'polici_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, values['-save_net-'])
        if event == '-load_net-':
            checkpoint = torch.load(values['-load_net-'])
            policy_net.load_state_dict(checkpoint['polici_net'])
            target_net.load_state_dict(checkpoint['target_net'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        FIRST_POINT_CF = float(values[3])
        try:
            if float(values[4]) > 0:
                STEERING_CF = float(values[4])
        except:
            pass
        NEXT_POINT_INTERVAL = int(values[5])
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
            last_update = values.copy()


CAD_window_task = threading.Thread(target=CAD_window, args=())
CAD_window_task.start()

state = torch.tensor(state, dtype=torch.float32, device=qlearning.device).unsqueeze(0)

# отрисовка
while runGame:
    # Обновление экрана
    screen.fill((200, 200, 200))
    bump_map.draw_normals_map(screen, camera.position.x, camera.position.y)

    points_buffer.clear()

    # Данные, относящиеся к Q-обучению
    reward = 0
    next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    out_of_way = False

    # Первая половина алгоритма машинного обучения
    action = select_action(state)
    control = num_control(action)
    if len(high_destinations) > 0:
        # Рассчет векторов до точек
        if len(high_destinations) > 0:
            for i in range(3):
                calulate_offset = min(len(high_destinations) - 1, NEXT_POINT_INTERVAL * (i))
                point_vector = linalg.VECTOR2(high_destinations[calulate_offset].x - player.position.x,
                                              high_destinations[calulate_offset].y - player.position.y) \
                    .rotate(math.radians(-player.position.angle + 90))
                points_buffer.append(linalg.POINT(high_destinations[calulate_offset].x, high_destinations[calulate_offset].y))
                next_state[2 * i] = point_vector.x / (NEXT_POINT_INTERVAL * 2 * (i + 1))
                next_state[2 * i + 1] = - point_vector.y / (100 * (i + 1))

        done = False

        # Проверка на сход с маршрута
        if linalg.VECTOR2(player.position.x - high_destinations[0].x,
                          player.position.y - high_destinations[0].y).get_length() > 20:
            reward -= 5
            out_of_way = True

        # Удаление достигнутых точек
        if len(destinations) > 0:
            while len(destinations) > 0 and linalg.VECTOR2(player.position.x - destinations[0].x, player.position.y - destinations[0].y).get_length() < 20:
                destinations.pop(0)
                reward += 5
        while not is_achieved and linalg.VECTOR2(player.position.x - high_destinations[0].x,
                                                 player.position.y - high_destinations[0].y).get_length() < 10:
            high_destinations.pop(0)
            reward += 0.5 * engine_efficiency(speed_km_h, difference_angle)
            if len(high_destinations) == 0:
                is_achieved = True

    # Отслеживание событий для движения
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_reverse = False
                current_vector_point = linalg.POINT(pygame.mouse.get_pos()[0] - camera.position.x,
                                                    pygame.mouse.get_pos()[1] - camera.position.y, direction=1)
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
                beginning_position = linalg.POINT(player.position.x, player.position.y, player.position.angle, player.position.direction)
                beginning_destinations = destinations.copy()
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

    # Применение событий для камеры
    if camera.is_left:
        camera.position.x += 20
    if camera.is_right:
        camera.position.x -= 20
    if camera.is_up:
        camera.position.y += 20
    if camera.is_down:
        camera.position.y -= 20

    # Применение событий для машины
    if not is_achieved:
        destination_angle = control[0] * 7
        player.velocity += control[1] / 5
        player.velocity = min(10, max(-3, player.velocity))
        if player.steering_angle < destination_angle:
            player.steering_angle += 0.5
        elif player.steering_angle > destination_angle:
            player.steering_angle -= 0.5
    else:
        if player.is_w:
            if player.velocity < 10:
                player.velocity += 0.2
        elif player.is_s:
            if player.velocity > -3:
                player.velocity -= 0.5
        elif player.velocity >= 0.2:
            player.velocity -= 0.2
        elif player.velocity <= -0.2:
            player.velocity += 0.2
        else:
            player.velocity = 0
        if player.is_a:
            player.steering_angle += -0.5
        if player.is_d:
            player.steering_angle += 0.5
    player.steering_angle = max(min(player.steering_angle, 7), -7)
    player.velocity = min(20, max(-10, player.velocity))

    # Рассчет высот
    player_altitude = 0
    next_altitude = 0
    next_altitude_offset = linalg.VECTOR2(10 * math.cos(math.radians(player.position.angle - 90)), 10 * math.sin(math.radians(player.position.angle - 90)))
    next_position = player.position.add(next_altitude_offset)
    if 0 < player.position.x < bump_map.width and 0 < player.position.y < bump_map.height:
        player_altitude = bump_map.get_color(player.position.x, player.position.y)
    if 0 < next_position.x < bump_map.width and 0 < next_position.y < bump_map.height:
        next_altitude = bump_map.get_color(next_position.x, next_position.y)
    altitude_difference = (next_altitude - player_altitude) / 20

    # перемещение
    if player.velocity >= 0:
        player.position.angle = linalg.solve_angle(player.position.angle,
                                            player.steering_angle * (player.vector.get_length() / 10))
    else:
        player.position.angle = linalg.solve_angle(player.position.angle,
                                            -player.steering_angle * (player.vector.get_length() / 10))

    direction_vector = linalg.VECTOR2(math.cos(math.radians(player.position.angle - 90)) * player.velocity * (1 / 2),
                                      math.sin(math.radians(player.position.angle - 90)) * player.velocity * (1 / 2))
    direction_vector = direction_vector.mult(1 - altitude_difference)
    player.vector.median(direction_vector.x, direction_vector.y, 0.9)

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

    # Вторая половина части с машинным обучением:
    next_state[6] = player.vector.get_length() / 6
    next_state[7] = player.steering_angle / 10
    next_state[8] = altitude_difference
    next_state_str = functions.float_array_to_str(next_state)
    reward_str = str(reward)
    if not is_achieved:
        # Рассчет следующего состояния

        next_state = torch.tensor(next_state, dtype=torch.float32, device=qlearning.device).unsqueeze(0)
        reward = torch.tensor([reward], device=qlearning.device)

        # Сохраняем опыт и обновляем модель
        memory.push(state, action, next_state, reward)
        state = next_state

        if is_learning:
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * qlearning.TAU + target_net_state_dict[key] * (1 - qlearning.TAU)
            target_net.load_state_dict(target_net_state_dict)

        if out_of_way:
            player.position = linalg.POINT(beginning_position.x, beginning_position.y, beginning_position.angle, beginning_position.direction)
            destinations = beginning_destinations.copy()
            player.vector = linalg.VECTOR2(0, 0)
            player.velocity = 0
            player.steering_angle = 0
            research_destinations()

    # Рассчет угла отклонения от траектории
    if len(high_destinations) > NEXT_POINT_INTERVAL:
        next_point_angle = (math.degrees(high_destinations[NEXT_POINT_INTERVAL].angle) - 206) % 360
        difference_angle = linalg.angle_difference(player.position.angle, next_point_angle)
    else:
        difference_angle = 0
        next_point_angle = 0

    # Вывод информации в виде текста:
    #   - Алгоритм обучения
    text_reward = sysfont.render(f'reward: {reward_str}', False, (0, 0, 0))
    text_action = sysfont.render(f'action: {str(control)}', False, (0, 0, 0))
    text_states = sysfont.render(f'states: {str(next_state_str)}', False, (0, 0, 0))
    screen.blit(text_reward, (10, 0))
    screen.blit(text_action, (10, 20))
    screen.blit(text_states, (10, 40))
    #   - Карта высот
    text_current_altitude = sysfont.render(f'current alt: {player_altitude:.2f}', False, (0, 0, 0))
    text_next_altitude = sysfont.render(f'next alt: {next_altitude:.2f}', False, (0, 0, 0))
    text_altitude_difference = sysfont.render(f'alt diff: {altitude_difference:.2f}', False, (0, 0, 0))
    screen.blit(text_current_altitude, (size[0] - 140, 0))
    screen.blit(text_next_altitude, (size[0] - 140, 20))
    screen.blit(text_altitude_difference, (size[0] - 140, 40))
    #   - Одометрия
    speed_km_h = player.vector.get_length() * fps / pixel_by_meter * 3.6
    text_speed = sysfont.render(f'speed: {speed_km_h:.2f} km/h [{player.vector.get_length():.2f}]', False, (0, 0, 0))
    text_difference = sysfont.render(f'difference: {difference_angle:.2f}', False, (0, 0, 0))
    text_yaw = sysfont.render(f'yaw: {player.position.angle:.2f}', False, (0, 0, 0))
    text_next_angle = sysfont.render(f'next angle: {next_point_angle:.2f}', False, (0, 0, 0))
    text_efficiency = sysfont.render(f'efficiency: {engine_efficiency(speed_km_h, altitude_difference):.2f}', False, (0, 0, 0))
    screen.blit(text_speed, (10, size[1] - 30))
    screen.blit(text_difference, (10, size[1] - 50))
    screen.blit(text_yaw, (10, size[1] - 70))
    screen.blit(text_next_angle, (10, size[1] - 90))
    screen.blit(text_efficiency, (10, size[1] - 110))

    # Запись данных в графике
    if plot_state == 1:
        if time() - plot_update_time > 0.1:
            plot_update_time = time()
            plot_times.append(time() - plot_begin_time)
            plot_speeds.append(speed_km_h)
            plot_angles.append(difference_angle)
            plot_differences.append(player_altitude / 100)

    # Отрисовка автомобиля
    functions.blit_rotate(screen, trueno, (player.position.x + camera.position.x, player.position.y + camera.position.y),
                (29, 76), player.position.angle)

    # Отрисовка и запись пройденного пути
    if time() - last_add_way > 0.1:
        way.append(linalg.POINT(player.position.x, player.position.y))
        last_add_way = time()
        if len(way) > 2000:
            way = way[-1000:]
    for i in range(1, len(way)):
        pygame.draw.line(screen, pygame.Color(255, 0, 255), [way[i - 1].x + camera.position.x, way[i - 1].y + camera.position.y],
                         [way[i].x + camera.position.x, way[i].y + camera.position.y], 2)

    # Отрисовка стен
    for wall in walls:
        pygame.draw.line(screen, pygame.Color(255, 0, 0), [wall.x1 + camera.position.x, wall.y1 + camera.position.y],
                         [wall.x2 + camera.position.x, wall.y2 + camera.position.y], 2)

    is_crash = False
    front_corner_l = linalg.POINT(
        player.position.x + (75 * math.cos(math.radians(player.position.angle + 20))),
        player.position.y + (75 * math.sin(math.radians(player.position.angle + 20))))
    front_corner_r = linalg.POINT(
        player.position.x + (75 * math.cos(math.radians(player.position.angle - 20))),
        player.position.y + (75 * math.sin(math.radians(player.position.angle - 20))))
    rear_corner_l = linalg.POINT(
        player.position.x + (75 * math.cos(math.radians(player.position.angle + 160))),
        player.position.y + (75 * math.sin(math.radians(player.position.angle + 160))))
    rear_corner_r = linalg.POINT(
        player.position.x + (75 * math.cos(math.radians(player.position.angle - 160))),
        player.position.y + (75 * math.sin(math.radians(player.position.angle - 160))))

    # отрисовка вектора направления
    pygame.draw.line(screen, pygame.Color(0, 255, 0),
                     [player.position.x + camera.position.x, player.position.y + camera.position.y],
                     [player.position.x + camera.position.x + next_altitude_offset.x, player.position.y + camera.position.y + next_altitude_offset.y], 2)

    # Отрисовка линий редакторов карты
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

        angle_vector = linalg.VECTOR2(math.cos(math.radians(player.position.angle)) * 10 * (1 / 2),
                                      math.sin(math.radians(player.position.angle)) * 10 * (1 / 2))

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

    # Отрисовка направляющих точек
    for i in range(len(points_buffer)):
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           [points_buffer[i].x + camera.position.x, points_buffer[i].y + camera.position.y], 5)

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
            player = structures.PLAYER(linalg.POINT(300, 300, 0), 0)

    pygame.display.update()

    clock.tick(fps)

pygame.quit()
import pygame
import random
import noise

# Инициализация Pygame
pygame.init()

# Параметры окна
WIDTH, HEIGHT = 800, 800
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Константы и переменные
num_obstacles = 40
obstacles = []
car_pos = [0, HEIGHT]
car_speed = 10  # Константная скорость для начала
obstacle_radius = 15  # Радиус препятствия


# Функция для генерации карты нормалей с использованием перлинового шума
def generate_normals_map(scale=0.009, octaves=6, persistence=0.6, lacunarity=2.0):
    return [[noise.pnoise2(x * scale, y * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
             for x in range(WIDTH)] for y in range(HEIGHT)]

# Функция для отрисовки карты нормалей
def draw_normals_map(normals_map):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Получаем значение высоты и нормализуем его в диапазон [0, 1]
            height = (normals_map[y][x] + 1) / 2
            # Преобразуем высоту в оттенок серого
            color_value = int(height * 255)
            color = (color_value, color_value, color_value)
            window.set_at((x, y), color)

# Функция для планирования пути (простая линейная траектория для примера)
def plan_path():
    # Здесь можно использовать более сложный алгоритм планирования пути
    path = []
    for i in range(WIDTH):
        path.append((i, HEIGHT - i))
    return path

# Функция для визуализации пути
def draw_path(path):
    for i in range(len(path) - 1):
        pygame.draw.line(window, BLUE, path[i], path[i + 1], 1)

# Основной цикл программы
running = True
clock = pygame.time.Clock()
normals_map = generate_normals_map()
path = plan_path()
path_index = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    window.fill(WHITE)

    # Отрисовка карты нормалей
    draw_normals_map(normals_map)
    # Визуализация пути
    draw_path(path)

    # Движение автомобиля по пути
    if path_index < len(path):
        car_pos = path[path_index]
        path_index += 1

    # Отрисовка автомобиля
    pygame.draw.circle(window, BLACK, car_pos, 10)

    pygame.display.update()
    clock.tick(60)  # 60 FPS

pygame.quit()
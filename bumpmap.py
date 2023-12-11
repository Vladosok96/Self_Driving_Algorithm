import pygame
import numpy as np
import noise


class BumpMap:
    # Функция для генерации карты нормалей с использованием перлинового шума
    def __init__(self, width=800, height=800, scale=0.003, octaves=6, persistence=0.3, lacunarity=2.0):
        self.width = width
        self.height = height
        self.normals_map = [[0 for i in range(width)] for j in range(height)]

        image_buffer = bytearray(height * width * 3)
        i = 0
        for y in range(height):
            for x in range(width):
                color = (noise.pnoise2(x * scale, y * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity) + 1) / 2 * 255
                image_buffer[i] = int(color)
                image_buffer[i + 1] = int(color)
                image_buffer[i + 2] = int(color)
                i += 3
                self.normals_map[y][x] = color
        self.image = pygame.image.frombuffer(image_buffer, (width, height), 'RGB')
        self.rect = self.image.get_rect()

    # Функция для отрисовки карты нормалей
    def draw_normals_map(self, window, x, y):
        window.blit(self.image, (x, y))

    def get_color(self, x, y):
        return self.normals_map[int(y)][int(x)]

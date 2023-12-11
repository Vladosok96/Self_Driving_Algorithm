import pygame
import numpy as np
import noise


class BumpMap:
    # Функция для генерации карты нормалей с использованием перлинового шума
    def __init__(self, width=800, height=800, scale=0.003, octaves=6, persistence=0.1, lacunarity=2.0):
        self.width = width
        self.height = height
        self.normals_map = bytearray(height * width * 3)
        i = 0
        for y in range(height):
            for x in range(width):
                self.normals_map[i] = int((noise.pnoise2(x * scale, y * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity) + 1) / 2 * 255)
                self.normals_map[i + 1] = int((noise.pnoise2(x * scale, y * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity) + 1) / 2 * 255)
                self.normals_map[i + 2] = int((noise.pnoise2(x * scale, y * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity) + 1) / 2 * 255)
                i += 3
        self.image = pygame.image.frombuffer(self.normals_map, (width, height), 'RGB')
        self.rect = self.image.get_rect()

    # Функция для отрисовки карты нормалей
    def draw_normals_map(self, window, x, y):
        window.blit(self.image, (x, y))

import pygame
import GPS

window = pygame.display.set_mode((500,500))
red = (200,0,0)

points = []

active = True

while active:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            active = False

    point_coordinates = input('inp>>>').split(', ')

    try:
        points.append(GPS.gps_to_rect(float(point_coordinates[0]), float(point_coordinates[1])))
    except:
        pass

    for i in range(len(points)):
        local_position = [points[i][0] - points[0][0], points[i][1] - points[0][1]]
        pygame.draw.circle(window, red, (250 + (local_position[0] * 10), 250 - (local_position[1] * 10)), 5)

    pygame.display.update()
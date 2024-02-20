import pygame
import GPS

window = pygame.display.set_mode((500,500))
red = (200,0,0)

points = []
gps = GPS.GPSReader('COM6', 3000000)
gps.read_data()

active = True

while active:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            active = False

    gps.read_data()
    points.append([gps.lat_dec, gps.lon_dec])
    print(gps.lat_dec, gps.lon_dec)

    # try:
    #     points.append(GPS.gps_to_rect(float(point_coordinates[0]), float(point_coordinates[1])))
    # except:
    #     pass

    for i in range(len(points)):
        local_position = [points[i][0] - points[0][0], points[i][1] - points[0][1]]
        pygame.draw.circle(window, red, (250 + (local_position[0] * 100), 250 - (local_position[1] * 100)), 5)

    pygame.display.update()
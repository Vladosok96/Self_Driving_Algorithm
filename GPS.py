#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import time
import csv
import sys
import struct
from math import sin, cos, tan, pi


def bytes_to_int(offset, paket):
    r = 0
    for i in range(4):
        d = i * 8
        r += int.from_bytes(paket[offset + i], 'little') << d
    return r


gps_positions = {'gps_state_status': 0,
                 'gps_int_longitude': 8,
                 'gps_int_latitude': 4,
                 'gps_altitude': 12,
                 'gps_velocity': 16,
                 'gps_yaw': 20}


class GPSReader:
    @property
    def lat_dec(self):
        return self.__lat_dec

    @property
    def lon_dec(self):
        return self.__lon_dec

    @property
    def velocity(self):
        return self.__velocity

    @property
    def yaw(self):
        return self.__yaw

    def to_decimal(self, lat, lon):
        lat_dd = int(float(lat) / 100)
        lat_mm = float(lat) - lat_dd * 100
        lat_dec = lat_dd + lat_mm / 60
        lon_dd = int(float(lon) / 100)
        lon_mm = float(lon) - lon_dd * 100
        lon_dec = lon_dd + lon_mm / 60
        return lat_dec, lon_dec

    def save_vals(self, count, filename):
        try:
            with open(filename, 'w') as gpscsv:
                wr = csv.writer(gpscsv, quoting=csv.QUOTE_ALL)
                i = 0
                while i < count:
                    prev_lat = float(self.__lat_dec)
                    self.read_data()
                    if abs(prev_lat - self.__lat_dec) > 10 ** (-7):
                        print("i: {}; lat:{}; lon:{}; velocity: {}".format(i, self.__lat_dec, self.__lon_dec,
                                                                                   self.__velocity))
                        wr.writerow([self.__lat_dec, self.__lon_dec])
                        i += 1
        except KeyboardInterrupt:
            pass

    def disconnect(self):
        self.__ser.close()  # close port

    def read_data(self):
        while not self.__ser.is_open:
            print("GPS device not found")
            time.sleep(2)

        self.__paket = [b'\x00']
        self.__paket[0] = self.__ser.read()

        while self.__paket[0] != b'\xff':
            self.__paket[0] = self.__ser.read()

        for byte in range(3):
            self.__paket.append(self.__ser.read())

        for byte in range(int.from_bytes(self.__paket[3], byteorder=sys.byteorder) + 4):
            self.__paket.append(self.__ser.read())

        self.__status = bool(bytes_to_int(gps_positions['gps_state_status'] + 4, self.__paket) & (2 ** 16)) - 1

        if self.__status == -1:
            print('GPS NMEA data received but not confidential')
        else:

            self.__lon_dec = bytes_to_int(gps_positions['gps_int_longitude'] + 4, self.__paket) * 360 / 4294967296
            self.__lat_dec = bytes_to_int(gps_positions['gps_int_latitude'] + 4, self.__paket) * 360 / 4294967296
            self.__altitude = self.__paket[gps_positions['gps_altitude'] + 4] + self.__paket[gps_positions['gps_altitude'] + 5] + \
                              self.__paket[gps_positions['gps_altitude'] + 6] + self.__paket[gps_positions['gps_altitude'] + 7]
            self.__altitude = struct.unpack('f', self.__altitude)[0]
            self.__velocity = self.__paket[gps_positions['gps_velocity'] + 4] + self.__paket[gps_positions['gps_velocity'] + 5] + \
                              self.__paket[gps_positions['gps_velocity'] + 6] + self.__paket[gps_positions['gps_velocity'] + 7]
            self.__velocity = struct.unpack('f', self.__velocity)[0]
            self.__yaw = self.__paket[gps_positions['gps_yaw'] + 4] + self.__paket[gps_positions['gps_yaw'] + 5] + self.__paket[
                gps_positions['gps_yaw'] + 6] + self.__paket[gps_positions['gps_yaw'] + 7]
            self.__yaw = struct.unpack('f', self.__yaw)[0]
            # print(self.__status, self.__lat_dec, self.__lon_dec, self.__altitude, self.__velocity, self.__yaw)

    def __init__(self, port, baudrate):
        self.__ser = serial.Serial(port, baudrate)
        print(self.__ser.name)  # check which port was really used
        self.__init_lat_dec, self.__init_lon_dec = 0, 0
        self.__lat_dec, self.__lon_dec = 0, 0
        self.__velocity = 0
        self.__yaw = 0
        self.__paket = [b'\x00']
        self.read_data()
        self.read_data()


def gps_to_rect(dLon, dLat):
    # Номер зоны Гаусса-Крюгера
    zone = int(dLon / 6.0 + 1)

    # Параметры эллипсоида Красовского
    a = 6378245.0  # Большая (экваториальная) полуось
    b = 6356863.019  # Малая (полярная) полуось
    e2 = (a ** 2 - b ** 2) / a ** 2  # Эксцентриситет
    n = (a - b) / (a + b)  # Приплюснутость

    # Параметры зоны Гаусса-Крюгера
    F = 1.0  # Масштабный коэффициент
    Lat0 = 0.0  # Начальная параллель (в радианах)
    Lon0 = (zone * 6 - 3) * pi / 180  # Центральный меридиан (в радианах)
    N0 = 0.0  # Условное северное смещение для начальной параллели
    E0 = zone * 1e6 + 500000.0  # Условное восточное смещение для центрального меридиана

    # Перевод широты и долготы в радианы
    Lat = dLat * pi / 180.0
    Lon = dLon * pi / 180.0

    # Вычисление переменных для преобразования
    v = a * F * (1 - e2 * (sin(Lat) ** 2)) ** -0.5
    p = a * F * (1 - e2) * (1 - e2 * (sin(Lat) ** 2)) ** -1.5
    n2 = v / p - 1
    M1 = (1 + n + 5.0 / 4.0 * n ** 2 + 5.0 / 4.0 * n ** 3) * (Lat - Lat0)
    M2 = (3 * n + 3 * n ** 2 + 21.0 / 8.0 * n ** 3) * sin(Lat - Lat0) * cos(Lat + Lat0)
    M3 = (15.0 / 8.0 * n ** 2 + 15.0 / 8.0 * n ** 3) * sin(2 * (Lat - Lat0)) * cos(2 * (Lat + Lat0))
    M4 = 35.0 / 24.0 * n ** 3 * sin(3 * (Lat - Lat0)) * cos(3 * (Lat + Lat0))
    M = b * F * (M1 - M2 + M3 - M4)
    I = M + N0
    II = v / 2 * sin(Lat) * cos(Lat)
    III = v / 24 * sin(Lat) * (cos(Lat)) ** 3 * (5 - (tan(Lat) ** 2) + 9 * n2)
    IIIA = v / 720 * sin(Lat) * (cos(Lat) ** 5) * (61 - 58 * (tan(Lat) ** 2) + (tan(Lat) ** 4))
    IV = v * cos(Lat)
    V = v / 6 * (cos(Lat) ** 3) * (v / p - (tan(Lat) ** 2))
    VI = v / 120 * (cos(Lat) ** 5) * (5 - 18 * (tan(Lat) ** 2) + (tan(Lat) ** 4) + 14 * n2 - 58 * (tan(Lat) ** 2) * n2)

    # Вычисление северного и восточного смещения (в метрах)
    N = I + II * (Lon - Lon0) ** 2 + III * (Lon - Lon0) ** 4 + IIIA * (Lon - Lon0) ** 6
    E = E0 + IV * (Lon - Lon0) + V * (Lon - Lon0) ** 3 + VI * (Lon - Lon0) ** 5

    return N, E

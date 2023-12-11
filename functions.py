import random
import pygame


def custom_random_choice(a, size=None, replace=True, p=None):
    """
    Выбирает случайные элементы из массива 'a'.

    Параметры:
    - a: одномерный массив или последовательность
    - size: int или tuple[int], опционально, размер возвращаемой выборки
    - replace: bool, опционально, указывает, может ли элемент повторяться в выборке
    - p: одномерный массив, опционально, вероятности для каждого элемента в 'a'

    Возвращает:
    - Выборка из 'a'
    """
    if not isinstance(a, (list, tuple)):
        raise ValueError("Массив 'a' должен быть одномерным.")

    if p is not None:
        if len(a) != len(p):
            raise ValueError("Длины массивов 'a' и 'p' должны совпадать.")
        if not all(0 <= prob <= 1 for prob in p):
            raise ValueError("Вероятности 'p' должны находиться в диапазоне от 0 до 1.")
        if abs(sum(p) - 1.0) > 1e-9:
            raise ValueError("Сумма вероятностей 'p' должна быть равна 1.")

    if not isinstance(replace, bool):
        raise ValueError("Параметр 'replace' должен быть типа bool.")

    if size is not None:
        if isinstance(size, int):
            size = (size,)
        elif not isinstance(size, tuple) or not all(isinstance(dim, int) for dim in size):
            raise ValueError("Параметр 'size' должен быть int или tuple[int].")

    if not replace and size is not None:
        if sum(size) > len(a):
            raise ValueError("Размер выборки превышает количество элементов в массиве 'a'.")

    # Генерация выборки
    if p is not None:
        indices = random.choices(range(len(a)), weights=p, k=sum(size) if size is not None else 1)
    else:
        indices = random.sample(range(len(a)), sum(size) if size is not None else 1)

    # Построение выборки
    if size is not None:
        return [a[i] for i in indices]
    else:
        return a[indices[0]]


def float_array_to_str(arr):
    out = ''
    for element in arr:
        out += f'{element:.2f}, '
    return out


def blit_rotate(surf, image, pos, origin_pos, angle):
    angle = -angle
    # calculate the axis aligned bounding box of the rotated image
    w, h = image.get_size()
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # calculate the translation of the pivot
    pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)

    # rotate and blit the image
    surf.blit(rotated_image, origin)


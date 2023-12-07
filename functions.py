import random


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


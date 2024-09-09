import numpy as np

def TMA(a, b, c, f = None):
    '''
    Входные массивы должны быть одномерными numpy массивами (float)
    Размерность b, f = n (главная диагональ матрицы и свободный член)
    Размерность a, c = n - 1 (поддиагональ и наддиагональ соответственно)
    '''
    n = len(b)
    if f is None:
        f = np.zeros(n)
    # приводим к матрице с 0 на поддиагонали
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        f[i] -= m * f[i-1]
    result = np.zeros(n)
    # обратный ход метода Гаусса
    result[-1] = f[-1] / b[-1]
    for i in range(n-2, -1, -1):
        result[i] = (f[i] - c[i] * result[i+1]) / b[i]
    return result
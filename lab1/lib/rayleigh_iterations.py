import numpy as np
from numpy import linalg as LA
from .TMA import TMA

def rayleigh_method_for_TM(a, b, c, lambd, v = None, epsilon = 1e-8, max_iterations = 50):
    '''
    Метод предполагает использование для случая оператора с трехдиагональной матрицей
    a, b, c - numpy массивы float n-1, n, n-1 (поддиагональ, главная диагональ, и наддиагональ),
    lambd - скаляр (начальное приближение собственного значения),
    v - numpy массив float (начальное приближение собственного вектора),
    epsilon - точность,
    max_iterations - ограничение итераций

    Возвращает найденное собственное значение и соответствующий ему вектор
    '''
    n= len(b)
    iteration_counter = 0

    while np.any(b - lambd == 0.0):
        lambd += 0.00000001
    
    if v is None:
        v = np.ones(n)
    else:
        v = v.copy()

    v /= LA.norm(v)
    v_new = TMA(a, b - lambd, c, v)
    mu = v_new.dot(v)
    lambd =  lambd + 1 / mu
    lambd_old = lambd
    err =  2 * epsilon

    while err > epsilon and iteration_counter < max_iterations:
        v = v_new / LA.norm(v_new)
        v_new = TMA(a, b - lambd, c, v)
        mu = v_new.dot(v)
        lambd =  lambd + 1 / mu
        err = np.abs(lambd - lambd_old)
        lambd_old = lambd
        iteration_counter += 1

    return lambd, v_new / LA.norm(v_new)



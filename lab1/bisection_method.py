import numpy as np

def bisection_method(f, a, b, epsilon = 1e-8):
    '''
    На вход подается функция func,
    a, b - границы поиска корня

    Возвращает решение уравнения f(x) = 0 на отрезке [a, b]
    '''
    x = (a + b)/2
    while abs(b-a) > 2*epsilon:
        if f(x) * f(a) < 0:
            b = x
        else:
            a = x
        x = (a + b)/2
    return x
import numpy as np
from .rayleigh_iterations import rayleigh_method_for_TM
from .bisection_method import bisection_method

def get_eigs(l, h, init_eigenvalues_appr):
    '''
    Функция под конкретную задачу поиска собственных функций в неглубокой прямоугольной потенциальной яме.

    Получает на вход радиус ямы l, шаг сетки h и начальные приближения собственных значений

    Возвращает списки из собственных значений и собственных векторов в формате numpy массивов
    '''
    u0 = 400/ l**2
    x = np.arange(-2*l, 2*l, h) + 0.5 * h
    u = u0 * np.heaviside(np.abs(x) - l, 1)
    n = int(l * 4 / h)
    # трехдиагональну матрицу симметричного оператора можно хранить всего в 2 массивах 
    b = np.ones(n) * 2 # главная диагональ
    b[0], b[-1] = 3, 3 # считая на границах решения задачи [-2l, 2l] волновую функцию равной 0
                       # получаем граничные условия 1 рода, за счет них получается 3
    b /= h**2
    b += u
    a = -1 * np.ones(n-1) / h**2 # наддиагональ и поддиагональ
    eigenvalues_appr = []
    eigenvectors_appr = []
    for i, lambd in enumerate(init_eigenvalues_appr):
        init_func_appr = np.cos(x) if i % 2 == 0 else np.sin(x)
        cur_eigenvalue, cur_eigenvector = rayleigh_method_for_TM(a, b, a, lambd, v=init_func_appr, max_iterations=1000)
        cur_eigenvector /= np.abs(cur_eigenvector).max() # нормируем амлитуду на 1
        eigenvalues_appr.append(cur_eigenvalue)
        eigenvectors_appr.append(cur_eigenvector)
    return eigenvalues_appr, eigenvectors_appr

def get_exact_eigfuncs(l, h, lambds):
    '''
    Функция под конкретную задачу поиска собственных функций в неглубокой прямоугольной потенциальной яме.

    Получает на вход радиус ямы l, шаг сетки h и массив собственных значений, для которых нужно получить функции
    Собственные значения должны начинаться с минимального (в этой задаче это четный уровень энергии) и идти подряд

    Возвращает список из собственных функций в формате numpy массивов
    '''
    u0 = 400/ l**2
    x = np.arange(-2*l, 2*l, h) + 0.5 * h
    def ksi_even(x, lambd):
        return np.exp(-np.sqrt(u0 - lambd) * l) * np.cos(np.sqrt(lambd) * x) * np.heaviside(x + l, 0) * np.heaviside(l - x, 0) + \
            np.cos(np.sqrt(lambd) * l) * np.exp(-np.sqrt(u0 - lambd) * np.abs(x)) * (np.heaviside(x - l, 1) + np.heaviside( - x - l, 1))

    def ksi_odd(x, lambd):
        return np.exp(-np.sqrt(u0 - lambd) * l) * np.sin(np.sqrt(lambd) * x) * np.heaviside(x + l, 0) * np.heaviside(l - x, 0) + \
            np.sin(np.sqrt(lambd) * l) * np.exp(-np.sqrt(u0 - lambd) * np.abs(x)) * np.sign(x) * (np.heaviside(x - l, 1) + np.heaviside( - x - l, 1))
    
    func_anal = []
    for i, lambd in enumerate(lambds):
        if (i % 2 == 0):
            func_anal.append(ksi_even(x, lambd) / np.abs(ksi_even(x, lambd)).max())
        else:
            func_anal.append(ksi_odd(x, lambd) / np.abs(ksi_odd(x, lambd)).max())
    return func_anal

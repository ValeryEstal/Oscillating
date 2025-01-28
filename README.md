 
import numpy as np
import matplotlib.pyplot as plt

# Определение параметров системы (их можно изменять для исследования различных режимов)
delta = 0.1  # Коэффициент затухания (изменять для разных режимов)
omega_0 = 1.0  # Собственная частота системы
A = 1.0  # Амплитуда внешней силы
omega = 1.5  # Частота внешней силы

# Параметры времени
t_end = 100  # Конечное время моделирования
dt = 0.01  # Шаг по времени
t = np.arange(0, t_end, dt)  # Массив времени

# Функция для вычисления производных для системы дифференциальных уравнений
def system_dynamics(x, t, delta, omega_0, A, omega):
    """
    Описывает динамику системы с учетом затухания, собственной частоты, внешней силы.
    Вход:
        x: массив, содержащий текущее положение и скорость (x1 = положение, x2 = скорость)
        t: текущий момент времени
        delta: коэффициент затухания
        omega_0: собственная частота системы
        A: амплитуда внешней силы
        omega: частота внешней силы
    Выход:
        Массив производных (скорость и ускорение)
    """
    x1 = x[0]  # Положение (x)
    x2 = x[1]  # Скорость (v = dx/dt)
   
    dx1_dt = x2  # Скорость
    dx2_dt = -2 * delta * x2 - omega_0**2 * np.sin(x1) + A * np.cos(omega * t)  # Ускорение
   
    return np.array([dx1_dt, dx2_dt])  # Возвращаем массив производных

# Метод Рунге-Кутта 4-го порядка для численного решения уравнений
def runge_kutta_4(f, x0, t, delta, omega_0, A, omega):
    """
    Численное решение дифференциального уравнения методом Рунге-Кутта 4-го порядка.
    Вход:
        f: функция, описывающая динамику системы (system_dynamics)
        x0: начальное состояние системы (положение и скорость)
        t: массив временных шагов
        delta: коэффициент затухания
        omega_0: собственная частота системы
        A: амплитуда внешней силы
        omega: частота внешней силы
    Выход:
        Массив состояний системы (положение и скорость) для каждого временного шага
    """
    x = np.zeros((len(t), 2))  # Массив с сотояниями
    x[0] = x0  # Устанавливаем начальное состояние
    for i in range(len(t) - 1):
        k1 = f(x[i], t[i], delta, omega_0, A, omega)
        k2 = f(x[i] + dt / 2 * k1, t[i] + dt / 2, delta, omega_0, A, omega)
        k3 = f(x[i] + dt / 2 * k2, t[i] + dt / 2, delta, omega_0, A, omega)
        k4 = f(x[i] + dt * k3, t[i] + dt, delta, omega_0, A, omega)
        x[i + 1] = x[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # Четвертый порядок точности
    return x  # Возвращаем массив состояний

# Начальные условия и параметры системы для разных фазовых портретов
cases = {
    "Точка типа Узел": {"delta": 0.5, "omega_0": 1.0, "A": 0, "omega": 0, "x0": [1, 0]},  # Сильное затухание
    "Точка типа СЕДЛО": {"delta": 0.0, "omega_0": 1.0, "A": 0, "omega": 0, "x0": [0.1, 0.1]},  # Без затухания
    "ЦЕНТР": {"delta": 0.0, "omega_0": 1.0, "A": 0, "omega": 0, "x0": [1, 0]},  # Периодическое движение
    "ФОКУС": {"delta": 0.1, "omega_0": 1.0, "A": 0.5, "omega": 1.5, "x0": [1, 0]}  # Слабое затухание и внешняя сила
}
# Фазовые портреты
plt.figure(figsize=(12, 12))
for idx, (title, case) in enumerate(cases.items(), 1):
    x = runge_kutta_4(system_dynamics, case["x0"], t, case["delta"], case["omega_0"], case["A"], case["omega"])
    plt.subplot(2, 2, idx)
    plt.plot(x[:, 0], x[:, 1])
    plt.xlabel("Положение (X)")
    plt.ylabel("Скорость (X')")
    plt.title(f"Фазовый портрет - {title}")
    plt.grid(True)

plt.tight_layout()
plt.show()

# Зависимости X(t) и X'(t)
x = runge_kutta_4(system_dynamics, [1, 0], t, delta, omega_0, A, omega)
x_t = x[:, 0]  # Положение как функция времени
v_t = x[:, 1]  # Скорость как функция времени

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x_t)
plt.xlabel('Время (t)')
plt.ylabel('Положение X(t)')
plt.title('Зависимость положения от времени')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, v_t)
plt.xlabel('Время (t)')
plt.ylabel('Скорость X\'(t)')
plt.title('Зависимость скорости от времени')
plt.grid(True)

plt.tight_layout()
plt.show()

# Изображения/Отображения Пуанкаре (выбор временных шагов на основе частоты внешней силы)
t_poincare = np.arange(10, t_end, 2 * np.pi / omega)
x_poincare = x_t[np.searchsorted(t, t_poincare)]
v_poincare = v_t[np.searchsorted(t, t_poincare)]

plt.figure()
plt.scatter(x_poincare, v_poincare, s=10)
plt.xlabel('Положение (X)')
plt.ylabel('Скорость (X\')')
plt.title('Изображение Пуанкаре')
plt.grid(True)
plt.show()

# Построение амплитудно-частотной характеристики
omega_range = np.linspace(0.1, 3, 100)
amplitudes = []

for omega_val in omega_range:
    x = runge_kutta_4(system_dynamics, [1, 0], t, delta, omega_0, A, omega_val)
    x_steady = x[int(len(x) / 2):]  # Вторая половина решения
    amplitude = np.max(x_steady[:, 0]) - np.min(x_steady[:, 0])  # Вычисление амплитуды
    amplitudes.append(amplitude)

plt.figure()
plt.plot(omega_range, amplitudes)
plt.xlabel('Частота (ω)')
plt.ylabel('Амплитуда колебаний')
plt.title('Амплитудно-частотная характеристика')
plt.grid(True)
plt.show()

# Построение фазо-частотной характеристики
phases = []

for omega_val in omega_range:
    x = runge_kutta_4(system_dynamics, [1, 0], t, delta, omega_0, A, omega_val)
    x_steady = x[int(len(x) / 2):]  # Вторая половина решения
    peak_idx = np.argmax(x_steady[:, 0])
    min_idx = np.argmin(x_steady[:, 0])
    phase = (t[min_idx] - t[peak_idx]) / (omega_val * 2 * np.pi) * 2 * np.pi  # Вычисление фазы
    phases.append(phase)

plt.figure()
plt.plot(omega_range, phases)
plt.xlabel('Частота (ω)')
plt.ylabel('Фаза колебаний (радианы)')
plt.title('Фазо-частотная характеристика')
plt.grid(True)
plt.show()


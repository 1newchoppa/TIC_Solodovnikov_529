import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


def generate_random_signal(mean, std_dev, num_elements):
    random_signal = np.random.normal(mean, std_dev, num_elements)
    return random_signal


def plot_signal(time_values, signal_values, title, x_label, y_label, save_path=None):
    plt.figure(figsize=(21, 14))
    plt.plot(time_values, signal_values, linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    else:
        plt.show()


# Створення папки для збереження
output_directory = './figures/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Параметри генерації сигналу
mean_value = 0
std_deviation = 10
num_elements_to_generate = 500
sampling_rate = 1000  # Частота дискретизації

# Генерація випадкового сигналу
random_signal = generate_random_signal(mean_value, std_deviation, num_elements_to_generate)

# Визначення відліків часу
time_values = np.arange(num_elements_to_generate) / sampling_rate

# Розрахунок параметрів фільтру
F_max = 23  # Максимальна частота сигналу в Гц
w = F_max / (sampling_rate / 2)
order = 3  # Порядок фільтру
sos = signal.butter(order, w, 'low', output='sos')

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(sos, random_signal)

# Відображення та збереження результатів
output_path_before = os.path.join(output_directory, 'random_signal_before.png')
output_path_after = os.path.join(output_directory, 'filtered_signal_after.png')

plot_signal(time_values, random_signal, 'Випадковий сигнал (до фільтрації)', 'Час, сек', 'Значення сигналу',
            output_path_before)
plot_signal(time_values, filtered_signal, 'Сигнал з максимальною частотою F_max = 23 Гц', 'Час, сек', 'Значення сигналу після фільтрації',
            output_path_after)

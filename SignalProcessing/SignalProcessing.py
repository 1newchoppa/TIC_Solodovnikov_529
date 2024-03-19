import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, fftfreq
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

# Создание списка для сохранения дискретизованных сигналов
discrete_signals = []

# Цикл дискретизации с различными шагами
for Dt in [2, 4, 8, 16]:
    # Создание нового сигнала для каждого шага дискретизации
    discrete_signal = np.zeros(num_elements_to_generate)

    # Процедура прореживания сигнала
    for i in range(0, round(num_elements_to_generate/Dt)):
        discrete_signal[i*Dt] = filtered_signal[i*Dt]

    # Добавление дискретизованного сигнала в список
    discrete_signals += [list(discrete_signal)]

    # Відображення та збереження результатів
    # output_path_discrete = os.path.join(output_directory, f'discrete_signal_{Dt}.png')
    # plot_signal(time_values, discrete_signal, f'Дискретизований сигнал, Dt = {Dt}', 'Час, сек', 'Значення сигналу після дискретизації',
    #             output_path_discrete)

# Відображення та збереження результатів
output_path_before = os.path.join(output_directory, '!random_signal_before.png')
output_path_after = os.path.join(output_directory, '!filtered_signal_after.png')

plot_signal(time_values, random_signal, 'Випадковий сигнал (до фільтрації)', 'Час, сек', 'Значення сигналу',
            output_path_before)
plot_signal(time_values, filtered_signal, 'Сигнал з максимальною частотою F_max = 23 Гц', 'Час, сек', 'Значення сигналу після фільтрації',
            output_path_after)

# Параметр дискретизации
Dt = 30  # Крок дискретизації

# Создание дополнительной переменной для сохранения дискретизованного сигнала
discrete_signal = np.zeros(num_elements_to_generate)

# Процедура дискретизации
for i in range(0, round(num_elements_to_generate/Dt)):
    discrete_signal[i*Dt] = filtered_signal[i*Dt]

# Відображення та збереження результатів
output_path_discrete = os.path.join(output_directory, 'discrete_signal.png')

plot_signal(time_values, discrete_signal, 'Дискретизований сигнал', 'Час, сек', 'Значення сигналу після дискретизації',
            output_path_discrete)

# Параметры графика
line_width = 1
font_size = 14

# Создание фигуры и осей
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

# Построение графиков
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_values, discrete_signals[s], linewidth=line_width)
        s += 1

# Подписи осей
fig.supxlabel('Час, сек', fontsize=font_size)
fig.supylabel('Значення сигналу після дискретизації', fontsize=font_size)

# Заголовок
fig.suptitle('Дискретизовані сигнали з кроком 2, 4, 8, 16', fontsize=font_size)

# Сохранение графика
fig.savefig('./figures/Дискретизовані сигнали.png', dpi=600)

#Этап 5,6

# Створення списку для збереження спектрів дискретизованих сигналів
discrete_spectrums = []

# Цикл дискретизації з різними кроками
for Dt in [2, 4, 8, 16]:
    # Створення нового сигналу для кожного кроку дискретизації
    discrete_signal = np.zeros(num_elements_to_generate)

    # Процедура проріджування сигналу
    for i in range(0, round(num_elements_to_generate/Dt)):
        discrete_signal[i*Dt] = filtered_signal[i*Dt]

    # Додавання дискретизованого сигналу до списку
    discrete_signals += [list(discrete_signal)]

    # Розрахунок спектру сигналу
    spectrum = np.abs(fftshift(fft(discrete_signal)))
    discrete_spectrums += [list(spectrum)]

# Розрахунок частотних відліків спектру
freqs = fftshift(fftfreq(num_elements_to_generate, 1/sampling_rate))

# Параметри графіка
line_width = 1
font_size = 14

# Створення фігури та осей
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

# Побудова графіків
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(freqs, discrete_spectrums[s], linewidth=line_width)
        s += 1

# Підписи осей
fig.supxlabel('Частота, Гц', fontsize=font_size)
fig.supylabel('Амплітуда', fontsize=font_size)

# Заголовок
fig.suptitle('Спектри дискретизованих сигналів з кроком 2, 4, 8, 16', fontsize=font_size)

# Збереження графіка
fig.savefig('./figures/Спектри дискретизованих сигналів.png', dpi=600)

# Этап 7,8

# Створення списку для збереження відновлених сигналів
recovered_signals = []

# Цикл відновлення аналогового сигналу з дискретного
for index, discrete_signal in enumerate(discrete_signals):
    # Визначення нормованої частоти фільтру
    w = F_max/(sampling_rate/2)

    # Розрахунок параметрів фільтру
    sos = signal.butter(3, w, 'low', output='sos')

    # Фільтрація дискретизованого сигналу
    recovered_signal = signal.sosfiltfilt(sos, discrete_signal)

    # Додавання відновленого сигналу до списку
    recovered_signals += [list(recovered_signal)]

# Параметри графіка
line_width = 1
font_size = 14

# Створення фігури та осей
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

# Побудова графіків
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_values, recovered_signals[s], linewidth=line_width)
        s += 1

# Підписи осей
fig.supxlabel('Час, сек', fontsize=font_size)
fig.supylabel('Значення відновленого сигналу', fontsize=font_size)

# Заголовок
fig.suptitle('Відновлені сигнали з кроком 2, 4, 8, 16', fontsize=font_size)

# Збереження графіка
fig.savefig('./figures/Відновлені сигнали.png', dpi=600)

#Этап 9,10

# Змінні для збереження дискретних сигналів, спектрів, відновлених сигналів, дисперсії та співвідношення сигнал-шум
discrete_signals = []
discrete_spectrums = []
recovered_signals = []
variances = []
snr_ratios = []

# Цикл Дт по [2, 4, 8, 16]
for Dt in [2, 4, 8, 16]:
    # Створення змінної для дискретного сигналу сформована з нулів
    discrete_signal = np.zeros(num_elements_to_generate)

    # Цикл для прорідження початкового сигналу
    for i in range(0, round(num_elements_to_generate / Dt)):
        # Формування дискретизованого сигналу з певним кроком
        discrete_signal[i * Dt] = filtered_signal[i * Dt]

    # Збереження дискретизованого сигналу у список
    discrete_signals.append(list(discrete_signal))

    # Розрахунок спектру для дискретизованого сигналу
    spectrum = np.abs(fftshift(fft(discrete_signal)))
    discrete_spectrums.append(list(spectrum))

    # Розрахунок параметрів фільтру ФНЧ
    sos = signal.butter(order, w, 'low', output='sos')

    # Відновлення аналогового сигналу шляхом фільтрації дискретизованого
    recovered_signal = signal.sosfiltfilt(sos, discrete_signal)

    # Збереження відновленного сигналу у список
    recovered_signals.append(list(recovered_signal))

    # Розрахунок різниці між початковим та відновленим сигналами
    E1 = recovered_signal - random_signal

    # Розрахунок дисперсії початкового сигналу
    signal_variance = np.var(random_signal)

    # Розрахунок дисперсії різниці між початковим та відновленим сигналами
    E1_variance = np.var(E1)

    # Розрахунок співвідношення сигнал-шум як відношення дисперсій
    snr = signal_variance / E1_variance

    # Збереження значень дисперсії різниці та співвідношення сигнал-шум
    variances.append(E1_variance)
    snr_ratios.append(snr)

# Тут можна продовжити збереження результатів, будувати графіки і виводити значення дисперсії та snr_ratios
# Побудова графіку для дисперсії різниці
plt.figure(figsize=(10, 6))
plt.plot([2, 4, 8, 16], variances, marker='o', linestyle='-', color='b')
plt.title('Залежність дисперсії від кроку дискретизації')
plt.xlabel('Крок дискретизації')
plt.ylabel('Дисперсія різниці')
plt.grid(True)
output_path_variance = os.path.join(output_directory, 'dependence_variance.png')
plt.savefig(output_path_variance)
plt.show()

# Побудова графіку для співвідношення сигнал-шум
plt.figure(figsize=(10, 6))
plt.plot([2, 4, 8, 16], snr_ratios, marker='o', linestyle='-', color='b')
plt.title('Залежність співвідношення сигнал-шум від кроку дискретизації')
plt.xlabel('Крок дискретизації')
plt.ylabel('Співвідношення сигнал-шум')
plt.grid(True)
output_path_snr = os.path.join(output_directory, 'dependence_snr.png')
plt.savefig(output_path_snr)
plt.show()

#---
#ПР 4

# Задані рівні квантування
quantization_levels = [4, 16, 64, 256]

# Змінні для збереження результатів
quantized_signals = []
variance_values = []
snr_values = []

# Цикл для квантування на різну кількість рівнів
for M in quantization_levels:

    bits = []
    # Розрахунок кроку квантування
    delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)

    # Квантування сигналу
    quantized_signal = delta * np.round(filtered_signal / delta)

    # Розрахунок квантованих рівнів сигналу
    quantize_levels = np.arange(np.min(quantized_signal), np.max(quantized_signal) + 1, delta)

    # Збереження результатів квантування
    quantized_signals.append(quantized_signal)

    # Генерація необхідного діапазону--------
    quantize_bit = np.arange(0, M)

    # Перетворення десяткових значень у бітові послідовності
    quantize_bit = [format(bits, '0' + str(int(np.log2(M))) + 'b') for bits in quantize_bit]

    # Об'єднання масивів квантованих рівнів та бітів
    quantize_table = np.c_[quantize_levels[:M], quantize_bit[:M]]

    # Розміри фігури
    fig, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))

    # Визначення властивостей таблиці
    table_properties = {
        'cellLoc': 'center',
        'bbox': [0, 0, 1, 1],
        'edges': 'closed'
    }

    # Створення таблиці
    table = ax.table(cellText=quantize_table,
                     colLabels=['Значення сигналу', 'Бітова послідовність'],
                     **table_properties)

    # Відключення вісей
    ax.axis('off')

    # Виведення таблиці
    plt.show()

    # Збереження отриманого рисунка
    fig.savefig(f'./figures/Table_quantization_M_{M}.png', dpi=600)

    # Перебір усіх значень отриманого цифрового сигналу
    for signal_value in quantized_signal:
        # Перебір значень з масиву квантованих рівнів
        for index, value in enumerate(quantize_levels[:M]):
            # Перевірка, чи модуль різниці округленої до нуля
            if np.round(np.abs(signal_value - value), 0) == 0:
                # Запис бітової послідовності у масив
                bits.append(quantize_bit[index])
                break

    # Об'єднання усіх елементів масиву в одну строку
    bits_combined = [int(item) for item in list(''.join(bits))]

    # Побудова графіку бітових послідовностей
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    x_values = np.arange(0, len(bits_combined))
    ax.step(x_values, bits_combined, linewidth=0.1)

    # Встановлення підписів та заголовку
    ax.set_title('Кодова послідовність сигналу при кількості рівнів квантування 4', fontsize=14)
    ax.set_xlabel('Біти', fontsize=14)
    ax.set_ylabel('Амплітуда сигналу', fontsize=14)

    # Збереження графіку у директорії проекту у папку figures
    # fig.savefig('./figures/bit_sequence_plot.png', dpi=600)
    fig.savefig('./figures/bit_sequence_plot.png', dpi=600)

    # Відображення графіку
    plt.show()

    # Розрахунок та збереження дисперсії
    variance = np.var(quantized_signal-filtered_signal)
    variance_values.append(variance)

    # Розрахунок та збереження співвідношення сигнал-шум
    snr = np.var(filtered_signal) / variance
    snr_values.append(snr)


fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time_values, quantized_signals[s], linewidth=1)
        s += 1

fig.supxlabel('Час(секунди)', fontsize=14)
fig.supylabel('Амплітуда сигналу', fontsize=14)
fig.suptitle('Цифрові сигнали з рівнями квантування (4, 16, 64, 256)', fontsize=14)
fig.savefig('./figures/' + 'Цифрові сигнали з рівнями квантування (4, 16, 64, 256)' + '.png', dpi=600)
plt.close(fig)

fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], variance_values, linewidth=1 , marker='o', linestyle='-', color='b')
fig.supxlabel('Кількість рівнів квантування', fontsize=14)
fig.supylabel('Дисперсія', fontsize=14)
fig.suptitle('Залежність дисперсії від кількості рівніів квантування', fontsize=14)
fig.savefig('./figures/' + 'Залежність дисперсії від кількості рівніів квантування' + '.png', dpi=600)
plt.grid(True)
plt.close(fig)

fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], snr_values, linewidth=1 , marker='o', linestyle='-', color='b')
fig.supxlabel('Кількість рівнів квантування', fontsize=14)
fig.supylabel('ССШ', fontsize=14)
fig.suptitle('Залежність співвідношення сигнал-шум від кількості рівніів квантування', fontsize=14)
fig.savefig('./figures/' + 'Залежність співвідношення сигнал-шум від кількості рівніів квантування' + '.png', dpi=600)
plt.grid(True)
plt.close(fig)
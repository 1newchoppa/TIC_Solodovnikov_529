from scipy import signal, fft
import numpy as np
import matplotlib.pyplot as plt


def generate_random_signal(mean, std_dev, num_elements):
    random_signal = np.random.normal(mean, std_dev, num_elements)
    return random_signal


def plot_signal(time_values, signal_values, title, x_label, y_label):
    plt.figure(figsize=(21, 14))
    plt.plot(time_values, signal_values, linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True)
    plt.show()


def plot_spectrum(signal, sampling_rate, max_amplitude):
    spectrum = fft.fft(signal)
    shifted_spectrum = fft.fftshift(spectrum)
    freqs = fft.fftfreq(len(signal), 1 / sampling_rate)
    shifted_freqs = fft.fftshift(freqs)

    # Обмеження амплітуди спектру до max_amplitude
    limited_spectrum = np.clip(np.abs(shifted_spectrum), 0, max_amplitude)

    plt.figure(figsize=(21, 14))
    plt.plot(shifted_freqs, limited_spectrum, linewidth=1)
    plt.title('Спектр сигналу з максимальною частотою F_max = 23 Гц', fontsize=14)
    plt.xlabel('Частота, Гц', fontsize=14)
    plt.ylabel('Амплітуда', fontsize=14)
    plt.xlim(-200, 200)
    plt.grid(True)
    plt.show()


def save_spectrum_plot(signal, sampling_rate, max_amplitude, save_path):
    spectrum = fft.fft(signal)
    shifted_spectrum = fft.fftshift(spectrum)
    freqs = fft.fftfreq(len(signal), 1 / sampling_rate)
    shifted_freqs = fft.fftshift(freqs)

    # Обмеження амплітуди спектру до max_amplitude
    limited_spectrum = np.clip(np.abs(shifted_spectrum), 0, max_amplitude)

    plt.figure(figsize=(21, 14))
    plt.plot(shifted_freqs, limited_spectrum, linewidth=1)
    plt.title('Спектр сигналу', fontsize=14)
    plt.xlabel('Частота, Гц', fontsize=14)
    plt.ylabel('Амплітуда', fontsize=14)
    plt.xlim(-200, 200)
    plt.grid(True)

    # Збереження зображення
    plt.savefig(save_path, dpi=600)

    # Відображення зображення
    plt.show()


# Параметри генерації сигналу
mean_value = 0
std_deviation = 10
num_elements_to_generate = 500
sampling_rate = 1000  # Частота дискретизації

# Генерація випадкового сигналу
random_signal = generate_random_signal(mean_value, std_deviation, num_elements_to_generate)

# Розрахунок параметрів фільтру
F_max = 23  # Максимальна частота сигналу в Гц
w = F_max / (sampling_rate / 2)
order = 3  # Порядок фільтру
sos = signal.butter(order, w, 'low', output='sos')

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(sos, random_signal)

# Відображення результатів фільтрації
plot_signal(np.arange(num_elements_to_generate) / sampling_rate, random_signal,
            'Випадковий сигнал (до фільтрації)', 'Час, сек', 'Значення сигналу')
plot_signal(np.arange(num_elements_to_generate) / sampling_rate, filtered_signal,
            'Фільтрований сигнал', 'Час, сек', 'Значення сигналу після фільтрації')

# Розрахунок та відображення спектру сигналу
plot_spectrum(filtered_signal, sampling_rate, max_amplitude=350)

save_path_spectrum = './figures/spectrum_plot.png'
save_spectrum_plot(filtered_signal, sampling_rate, max_amplitude=350, save_path=save_path_spectrum)

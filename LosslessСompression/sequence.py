import random
import string
import collections
import math

from matplotlib import pyplot as plt

# Тестова послідовність №1
# Номер студента у журналі
student_number = 11

# Розмірність послідовності
N_sequence = 100

# Створення списку з N1 елементами "1"
list1 = ["1"] * student_number

# Створення списку з N0 елементами "0"
list0 = ["0"] * (N_sequence - student_number)

# Об'єднання списків та перемішування їх елементів
original_sequence_1 = list1 + list0
random.shuffle(original_sequence_1)

# Обчислення розмірності алфавіту
Sequence_alphabet_size = len(set(original_sequence_1))

# Обчислення розміру послідовності в байтах та бітах
Original_sequence_size_bytes = len(original_sequence_1)
Original_sequence_size_bits = Original_sequence_size_bytes * 8


# Тестова послідовність №2

N_sequence = 100

list1 = ["s", "o", "l", "o", "d", "o", "v", "n", "i", "k", "o", "v"]
list0 = ["0"] * (N_sequence - len(list1))

original_sequence_2 = list1 + list0
Sequence_alphabet_size = len(set(original_sequence_2))

Original_sequence_size_bytes = len(original_sequence_2)
Original_sequence_size_bits = Original_sequence_size_bytes * 8


# Тестова послідовність №3

N_sequence = 100

list1 = ["s", "o", "l", "o", "d", "o", "v", "n", "i", "k", "o", "v"]
N1 = len(list1)
list0 = ["0"] * (N_sequence - N1)

# global original_sequence_3
original_sequence_3 = list1 + list0
random.shuffle(original_sequence_3)

Sequence_alphabet_size = len(set(original_sequence_3))

Original_sequence_size_bytes = len(original_sequence_3)
Original_sequence_size_bits = Original_sequence_size_bytes * 8


# Тестова послідовність №4

N_sequence = 100

letters = ["s", "o", "l", "o", "d", "o", "v", "n", "i", "k", "o", "v", "5", "2", "9"]
n_letters = len(letters)

n_repeats = (N_sequence // n_letters)
remainder = N_sequence % n_letters
list = letters * n_repeats
list += letters[:remainder]

original_sequence_4 = ''.join(map(str, list))

Sequence_alphabet_size = len(set(original_sequence_4))

Original_sequence_size_bytes = len(original_sequence_4)
Original_sequence_size_bits = Original_sequence_size_bytes * 8


# Тестова послідовність №5

N_sequence = 100

Pi = 0.2

sequence_list = ["s", "o", "5", "2", "9"]

original_sequence_5 = 20*sequence_list
# for _ in range(N_sequence):
#     if random.random() < Pi:
#         original_sequence_5.append(random.choice(sequence_list))
#     else:
#         original_sequence_5.append(random.choice(sequence_list))

random.shuffle(original_sequence_5)

Sequence_alphabet_size = len(set(original_sequence_5))

Original_sequence_size_bytes = len(original_sequence_5)
Original_sequence_size_bits = Original_sequence_size_bytes * 8


# Тестова послідовність №6

N_sequence = 100

P_let = 0.7
P_nums = 0.3

letters = ["s", "o"]
digits = ["5", "2", "9"]

n_lettets = int(P_let * N_sequence)
n_digits = int(P_nums * N_sequence)

list_100 = []
for i in range(n_lettets):
    list_100.append(random.choice(letters))
for i in range(n_digits):
    list_100.append(random.choice(digits))
random.shuffle(list_100)

original_sequence_6 = list_100

Sequence_alphabet_size = len(set(original_sequence_6))

Original_sequence_size_bytes = len(original_sequence_6)
Original_sequence_size_bits = Original_sequence_size_bytes * 8


# Тестова послідовність №7

N_sequence = 100
# elements = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
#                  't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]
elements = string.ascii_lowercase + string.digits

list_100 = [random.choice(elements) for _ in range(N_sequence)]
original_sequence_7 = list_100

Sequence_alphabet_size = len(set(original_sequence_7))

Original_sequence_size_bytes = len(original_sequence_7)
Original_sequence_size_bits = Original_sequence_size_bytes * 8

# Тестова послідовність №8

N_sequence = 100

list_digits = ["1"]

list_100 = list_digits * 100
original_sequence_8 = list_100

Sequence_alphabet_size = len(set(original_sequence_8))

Original_sequence_size_bytes = len(original_sequence_8)
Original_sequence_size_bits = Original_sequence_size_bytes * 8

#END
original_sequence_1 = ''.join(original_sequence_1)
original_sequence_2 = ''.join(original_sequence_2)
original_sequence_3 = ''.join(original_sequence_3)
original_sequence_4 = ''.join(original_sequence_4)
original_sequence_5 = ''.join(original_sequence_5)
original_sequence_6 = ''.join(original_sequence_6)
original_sequence_7 = ''.join(original_sequence_7)
original_sequence_8 = ''.join(original_sequence_8)

original_sequences = [original_sequence_1, original_sequence_2, original_sequence_3, original_sequence_4, original_sequence_5, original_sequence_6
                      , original_sequence_7, original_sequence_8]
results = []
for sequence in original_sequences:
    unique_chars = set(sequence)
    Sequence_alphabet_size = len(unique_chars)

    counts = collections.Counter(sequence)
    probability = {symbol: count / N_sequence for symbol, count in counts.items()}

    mean_probability = sum(probability.values()) / len(probability)

    equal = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in
    probability.values())

    uniformity = "рівна" if equal else "нерівна"

    entropy = -sum(p * math.log2(p) for p in probability.values())

    if Sequence_alphabet_size > 1:
        source_excess = 1 - entropy / math.log2(Sequence_alphabet_size)
    else:
        source_excess = 1

    probability_str = ', '.join([f"{symbol}={prob:.4f}" for symbol, prob in probability.items()])


    results.append ([Sequence_alphabet_size, round((entropy),2), round((source_excess),2), uniformity])

    with open("results_sequence.txt", "a") as file:
        file.write("Послідовність: {}\n".format(''.join(sequence)))
        file.write("Розмір послідовності (bytes): {}\n".format(Original_sequence_size_bytes))
        file.write("Розмір алфавіту: {}\n".format(Sequence_alphabet_size))
        file.write("Ймовірність появи символів: {}\n".format(probability))
        file.write("Середне арифметичне ймовірностей: {}\n".format(mean_probability))
        file.write("Ймовірність розподілу символів: {}\n".format(uniformity))
        file.write("Ентропія: {}\n".format(entropy))
        file.write("Надмірність: {}\n".format(source_excess))
        file.write("\n")


N = 8
fig,ax = plt.subplots(figsize=(14/1.54, N/1.54))

headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4',
       'Послідовність 5', 'Послідовність 6', 'Послідовність 7', 'Послідовність 8']


fig, ax = plt.subplots(figsize=(14/1.54, 8/1.54))
ax.axis('off')
table = ax.table(cellText=results, colLabels=headers, rowLabels=row, loc='center', cellLoc='center')

table.set_fontsize(14)
table.scale(0.8, 2)
fig.savefig('Характеристики сформованих послідовностей')

with open("sequence.txt", "w") as file:
    print(original_sequences, file=file)

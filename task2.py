import numpy as np

array = np.random.randint(low=-100, high=101, size=200)
print("1. Початковий масив (перші 20 елементів):")
print(array[:20], "...\n")

positive_mask = array > 0
positive_numbers = array[positive_mask]

print("2. Фільтрація додатних чисел за допомогою маски:")
print(f"Знайдено додатних чисел: {len(positive_numbers)}")
print(positive_numbers[:20], "...\n")

negative_mask = array < 0
array[negative_mask] = 0

print("3. Масив після заміни всіх від'ємних чисел на нулі (перші 20 елементів):")
print(array[:20], "...\n")

mean_value = np.mean(array)

print("4. Обчислення середнього значення:")
print(f"Середнє значення отриманого масиву: {mean_value:.2f}")
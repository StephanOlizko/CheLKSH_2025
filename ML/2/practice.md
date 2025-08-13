# Практикум по NumPy и Matplotlib
## Занятие для 9-10 классов (90 минут)

# Импорт необходимых библиотек
```python
import numpy as np
import matplotlib.pyplot as plt
```

# Настройка для корректного отображения русского текста
```python
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("Добро пожаловать на практикум по NumPy и Matplotlib!")
print("Сегодня мы изучим основы работы с массивами и построения графиков")
```

---

## Блок 1: Операции с векторами и матрицами (15 минут)

### Задание 1.1: Создание и операции с векторами

**Подробное описание и подсказки:**
1. Создайте два одномерных массива (вектора) с элементами `[1, 2, 3, 4, 5]` и `[6, 7, 8, 9, 10]` с помощью `np.array()`.
2. Выполните поэлементное сложение и умножение с помощью операторов `+` и `*`.
3. Найдите скалярное произведение с помощью `np.dot()` или оператора `@`.
4. Для вычисления длины (нормы) используйте `np.linalg.norm()`.

**Подсказки:**
- Для создания векторов используйте `np.array([1, 2, 3, 4, 5])` и `np.array([6, 7, 8, 9, 10])`.
- Для поэлементных операций используйте обычные арифметические операторы.
- Для скалярного произведения используйте `np.dot(A, B)`.
- Для длины вектора используйте `np.linalg.norm(A)`.

```python
# Ваш код здесь
# 1. Создайте два вектора A и B
# 2. Сложите их поэлементно
# 3. Перемножьте их поэлементно
# 4. Найдите скалярное произведение
# 5. Найдите длину каждого вектора
```

### Решение 1.1:
```python
# Создаем векторы
A = np.array([1, 2, 3, 4, 5])
B = np.array([6, 7, 8, 9, 10])

print("Вектор A:", A)
print("Вектор B:", B)

# Операции с векторами
sum_vectors = A + B
print("A + B =", sum_vectors)

mult_vectors = A * B
print("A * B (поэлементное) =", mult_vectors)

dot_product = np.dot(A, B)
print("Скалярное произведение A · B =", dot_product)

# Длина векторов (норма)
length_A = np.linalg.norm(A)
length_B = np.linalg.norm(B)
print(f"Длина вектора A: {length_A:.2f}")
print(f"Длина вектора B: {length_B:.2f}")
```

### Задание 1.2: Работа с матрицами

**Подробное описание и подсказки:**
1. Создайте матрицу C размером 3x3, заполненную числами от 1 до 9 с помощью `np.arange(1, 10).reshape(3, 3)`.
2. Создайте единичную матрицу D с помощью `np.eye(3)`.
3. Выполните матричное умножение C × D с помощью `np.dot()` или оператора `@`.
4. Найдите определитель матрицы C с помощью `np.linalg.det()`.

**Подсказки:**
- Для создания матрицы используйте `np.arange(1, 10).reshape(3, 3)`.
- Для единичной матрицы используйте `np.eye(3)`.
- Для матричного умножения используйте `np.dot(C, D)` или `C @ D`.
- Для определителя используйте `np.linalg.det(C)`.

```python
# Ваш код здесь
# 1. Создайте матрицу C (3x3) с числами от 1 до 9
# 2. Создайте единичную матрицу D
# 3. Выполните матричное умножение C × D
# 4. Найдите определитель матрицы C
```

### Решение 1.2:
```python
# Создаем матрицы
C = np.arange(1, 10).reshape(3, 3)
D = np.eye(3)  # единичная матрица

print("Матрица C:")
print(C)
print("\nМатрица D (единичная):")
print(D)

# Матричное умножение
product = np.dot(C, D)
print("\nC × D =")
print(product)

# Определитель матрицы C
det_C = np.linalg.det(C)
print(f"\nОпределитель матрицы C: {det_C:.2f}")
```

---

## Блок 2: Статистический анализ данных (15 минут)

### Задание 2.1: Анализ оценок класса

**Подробное описание и подсказки:**
1. Используйте массив оценок: `[5, 4, 3, 5, 4, 4, 3, 5, 4, 2, 5, 4, 3, 4, 5, 3, 4, 5, 4, 3]`.
2. Найдите среднее арифметическое с помощью `np.mean()`.
3. Найдите медиану с помощью `np.median()`.
4. Найдите стандартное отклонение с помощью `np.std()`.
5. Найдите минимальную и максимальную оценки с помощью `np.min()` и `np.max()`.
6. Подсчитайте количество каждой оценки с помощью `np.unique(grades, return_counts=True)`.

**Подсказки:**
- Используйте функции из шпаргалки для каждого пункта.
- Для подсчёта количества каждой оценки используйте цикл по уникальным значениям.

```python
# Ваш код здесь
# 1. Найдите среднее арифметическое
# 2. Найдите медиану
# 3. Найдите стандартное отклонение
# 4. Найдите минимальную и максимальную оценки
# 5. Подсчитайте количество каждой оценки
```

### Решение 2.1:
```python
grades = np.array([5, 4, 3, 5, 4, 4, 3, 5, 4, 2, 5, 4, 3, 4, 5, 3, 4, 5, 4, 3])

print("Анализ оценок класса:")
print("Оценки:", grades)

# Основные статистики
mean_grade = np.mean(grades)
median_grade = np.median(grades)
std_grade = np.std(grades)
min_grade = np.min(grades)
max_grade = np.max(grades)

print(f"\nСреднее арифметическое: {mean_grade:.2f}")
print(f"Медиана: {median_grade}")
print(f"Стандартное отклонение: {std_grade:.2f}")
print(f"Минимальная оценка: {min_grade}")
print(f"Максимальная оценка: {max_grade}")

# Подсчет каждой оценки
unique_grades, counts = np.unique(grades, return_counts=True)
print("\nКоличество каждой оценки:")
for grade, count in zip(unique_grades, counts):
    print(f"Оценка {grade}: {count} учеников")
```

---

## Блок 3: Генерация случайных данных и их обработка (10 минут)

### Задание 3.1: Моделирование бросков кубика

**Подробное описание и подсказки:**
1. Сгенерируйте массив из 1000 случайных целых чисел от 1 до 6 с помощью `np.random.randint()`. Используйте фиксированный seed: `np.random.seed(42)`.
2. Подсчитайте частоту каждого числа с помощью `np.unique(..., return_counts=True)`.
3. Найдите среднее значение с помощью `np.mean()`.

**Подсказки:**
- Для генерации используйте `np.random.randint(1, 7, size=1000)`.
- Для анализа используйте функции из шпаргалки.

```python
# Ваш код здесь
# 1. Сгенерируйте 1000 бросков кубика
# 2. Подсчитайте частоту каждого числа
# 3. Найдите среднее значение
```

### Решение 3.1:
```python
# Устанавливаем seed для воспроизводимости
np.random.seed(42)

# Генерируем 1000 бросков кубика
dice_rolls = np.random.randint(1, 7, size=1000)

print("Моделирование 1000 бросков кубика:")
print("Первые 20 бросков:", dice_rolls[:20])

# Анализ результатов
unique_values, counts = np.unique(dice_rolls, return_counts=True)
frequencies = counts / len(dice_rolls)

print(f"\nСреднее значение: {np.mean(dice_rolls):.3f}")
print("Частота каждого числа:")
for value, count, freq in zip(unique_values, counts, frequencies):
    print(f"Число {value}: {count} раз ({freq:.3f})")

# Проверим, насколько равномерно распределение
print(f"\nСтандартное отклонение частот: {np.std(frequencies):.4f}")
print("(Чем меньше, тем равномернее распределение)")
```

---

## Блок 4: Фильтрация и сортировка массивов (10 минут)

### Задание 4.1: Работа с температурными данными

**Подробное описание и подсказки:**
1. Сгенерируйте массив температур из 30 элементов с помощью `np.random.normal(20, 5, 30)`. Используйте фиксированный seed: `np.random.seed(123)`.
2. Отсортируйте температуры с помощью `np.sort()`.
3. Найдите дни с температурой выше 25°C с помощью булевой маски.
4. Найдите дни с температурой ниже 15°C с помощью булевой маски.
5. Создайте массив только с температурами в диапазоне 18-22°C с помощью комбинированной маски.

**Подсказки:**
- Для фильтрации используйте булевое индексирование: `temperatures[temperatures > 25]`.
- Для диапазона используйте: `(temperatures >= 18) & (temperatures <= 22)`.

```python
# Ваш код здесь
# 1. Сгенерируйте температурные данные
# 2. Отсортируйте температуры
# 3. Найдите дни с температурой выше 25°C
# 4. Найдите дни с температурой ниже 15°C
# 5. Выберите температуры в диапазоне 18-22°C
```

### Решение 4.1:
```python
# Генерируем температурные данные
np.random.seed(123)
temperatures = np.random.normal(20, 5, 30)

print("Анализ температурных данных за месяц:")
print(f"Температуры (первые 10 дней): {temperatures[:10]}")

# 1. Сортировка
sorted_temps = np.sort(temperatures)
print(f"\nОтсортированные температуры:")
print(f"Самая низкая: {sorted_temps[0]:.1f}°C")
print(f"Самая высокая: {sorted_temps[-1]:.1f}°C")

# 2. Дни с температурой выше 25°C
hot_days = temperatures > 25
hot_temperatures = temperatures[hot_days]
print(f"\nГорячие дни (>25°C): {len(hot_temperatures)} дней")
if len(hot_temperatures) > 0:
    print(f"Температуры в горячие дни: {hot_temperatures}")

# 3. Дни с температурой ниже 15°C
cold_days = temperatures < 15
cold_temperatures = temperatures[cold_days]
print(f"\nХолодные дни (<15°C): {len(cold_temperatures)} дней")
if len(cold_temperatures) > 0:
    print(f"Температуры в холодные дни: {cold_temperatures}")

# 4. Комфортные температуры (18-22°C)
comfortable_mask = (temperatures >= 18) & (temperatures <= 22)
comfortable_temps = temperatures[comfortable_mask]
print(f"\nКомфортные дни (18-22°C): {len(comfortable_temps)} дней")
print(f"Процент комфортных дней: {len(comfortable_temps)/len(temperatures)*100:.1f}%")
```

---

## Блок 5: Построение и анализ математических функций (15 минут)

### Задание 5.1: Исследование функций

**Подробное описание и подсказки:**
1. Создайте массив x на интервале [-2π, 2π] с помощью `np.linspace(-2*np.pi, 2*np.pi, 1000)`.
2. Вычислите значения функций: `np.sin(x)`, `np.cos(x)`, `x**2 - 4`, `np.sin(x) + np.cos(x)`.
3. Постройте графики всех функций с помощью `plt.plot()` и `plt.subplot()`.
4. Найдите точки пересечения синуса и косинуса (подсказка: где `sin(x) == cos(x)`, используйте формулу из шпаргалки).

**Подсказки:**
- Для построения графиков используйте примеры из шпаргалки.
- Для точек пересечения используйте формулу: `x = π/4 + nπ`.

```python
# Ваш код здесь
# 1. Создайте массив x
# 2. Вычислите значения функций
# 3. Постройте графики
# 4. Найдите точки пересечения sin(x) и cos(x)
```

### Решение 5.1:
```python
# Создаем массив x от -2π до 2π
x = np.linspace(-2*np.pi, 2*np.pi, 1000)

# Вычисляем функции
y_sin = np.sin(x)
y_cos = np.cos(x)
y_parabola = x**2 - 4
y_sum = y_sin + y_cos

# Строим графики
plt.figure(figsize=(12, 8))

# График 1: Синус и косинус
plt.subplot(2, 2, 1)
plt.plot(x, y_sin, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, y_cos, 'r-', label='cos(x)', linewidth=2)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Синус и косинус')
plt.xlabel('x')
plt.ylabel('y')

# График 2: Парабола
plt.subplot(2, 2, 2)
plt.plot(x, y_parabola, 'g-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.title('Парабола: y = x² - 4')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# График 3: Сумма синуса и косинуса
plt.subplot(2, 2, 3)
plt.plot(x, y_sum, 'm-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.title('y = sin(x) + cos(x)')
plt.xlabel('x')
plt.ylabel('y')

# График 4: Все функции вместе (масштабированные)
plt.subplot(2, 2, 4)
plt.plot(x, y_sin, 'b-', label='sin(x)', alpha=0.7)
plt.plot(x, y_cos, 'r-', label='cos(x)', alpha=0.7)
plt.plot(x, y_parabola/5, 'g-', label='(x²-4)/5', alpha=0.7)
plt.plot(x, y_sum, 'm-', label='sin(x)+cos(x)', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Сравнение функций')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# Найдем точки пересечения синуса и косинуса
# sin(x) = cos(x) => tan(x) = 1 => x = π/4 + nπ
intersections_x = []
for n in range(-2, 3):
    x_int = np.pi/4 + n*np.pi
    if -2*np.pi <= x_int <= 2*np.pi:
        intersections_x.append(x_int)

print("Точки пересечения sin(x) и cos(x):")
for x_int in intersections_x:
    y_int = np.sin(x_int)
    print(f"x = {x_int:.3f}, y = {y_int:.3f}")
```

---

## Блок 6: Столбчатые и круговые диаграммы (10 минут)

### Задание 6.1: Анализ популярности предметов

**Подробное описание и подсказки:**
1. Используйте массивы: предметы `['Математика', 'Физика', 'Химия', 'Биология', 'История', 'Литература']` и голоса `[45, 32, 28, 38, 25, 30]`.
2. Постройте столбчатую диаграмму с помощью `plt.bar()`.
3. Постройте круговую диаграмму с помощью `plt.pie()`.
4. Проанализируйте результаты: найдите самый и наименее популярный предмет, среднее количество голосов.

**Подсказки:**
- Для построения диаграмм используйте примеры из шпаргалки.
- Для анализа используйте функции `max()`, `min()`, `sum()`.

```python
# Ваш код здесь
# 1. Создайте массивы предметов и голосов
# 2. Постройте столбчатую диаграмму
# 3. Постройте круговую диаграмму
# 4. Проанализируйте результаты
```

### Решение 6.1:
```python
# Данные опроса
subjects = ['Математика', 'Физика', 'Химия', 'Биология', 'История', 'Литература']
votes = [45, 32, 28, 38, 25, 30]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

# Создаем фигуру с двумя подграфиками
plt.figure(figsize=(14, 6))

# Столбчатая диаграмма
plt.subplot(1, 2, 1)
bars = plt.bar(subjects, votes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
plt.title('Популярность школьных предметов\n(столбчатая диаграмма)', fontsize=14, fontweight='bold')
plt.ylabel('Количество голосов')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for bar, vote in zip(bars, votes):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{vote}', ha='center', va='bottom', fontweight='bold')

# Круговая диаграмма
plt.subplot(1, 2, 2)
wedges, texts, autotexts = plt.pie(votes, labels=subjects, colors=colors, 
                                   autopct='%1.1f%%', startangle=90,
                                   explode=[0.05 if vote == max(votes) else 0 for vote in votes])
plt.title('Популярность школьных предметов\n(круговая диаграмма)', fontsize=14, fontweight='bold')

# Улучшаем читаемость процентов
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()

# Статистический анализ
total_votes = sum(votes)
most_popular = subjects[votes.index(max(votes))]
least_popular = subjects[votes.index(min(votes))]

print(f"Общее количество голосов: {total_votes}")
print(f"Самый популярный предмет: {most_popular} ({max(votes)} голосов)")
print(f"Наименее популярный предмет: {least_popular} ({min(votes)} голосов)")
print(f"Средняя популярность: {total_votes/len(subjects):.1f} голосов на предмет")
```

---

## Блок 7: Гистограммы и диаграммы рассеяния (15 минут)

### Задание 7.1: Анализ роста и веса учеников

**Подробное описание и подсказки:**
1. Сгенерируйте массивы роста и веса с помощью `np.random.normal(165, 10, 50)` и `weights = 0.7 * heights + np.random.normal(0, 5, 50) - 50`. Используйте фиксированный seed: `np.random.seed(42)`.
2. Постройте гистограмму роста с помощью `plt.hist()`.
3. Постройте гистограмму веса с помощью `plt.hist()`.
4. Постройте диаграмму рассеяния с помощью `plt.scatter()`.
5. Найдите корреляцию между ростом и весом с помощью `np.corrcoef()`.

**Подсказки:**
- Для построения графиков используйте примеры из шпаргалки.
- Для корреляции используйте `np.corrcoef(heights, weights)[0, 1]`.

```python
# Ваш код здесь
# 1. Сгенерируйте данные о росте и весе
# 2. Постройте гистограмму роста
# 3. Постройте гистограмму веса
# 4. Постройте диаграмму рассеяния
# 5. Найдите корреляцию между ростом и весом
```

### Решение 7.1:
```python
# Генерируем данные
np.random.seed(42)
heights = np.random.normal(165, 10, 50)  # рост в см
weights = 0.7 * heights + np.random.normal(0, 5, 50) - 50  # вес зависит от роста + шум

# Убеждаемся, что все значения положительные и реалистичные
heights = np.clip(heights, 145, 190)
weights = np.clip(weights, 40, 90)

print(f"Данные о {len(heights)} учениках:")
print(f"Рост: от {min(heights):.1f} до {max(heights):.1f} см")
print(f"Вес: от {min(weights):.1f} до {max(weights):.1f} кг")

# Создаем фигуру с тремя подграфиками
plt.figure(figsize=(15, 5))

# 1. Гистограмма роста
plt.subplot(1, 3, 1)
plt.hist(heights, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, label=f'Среднее: {np.mean(heights):.1f} см')
plt.title('Распределение роста учеников')
plt.xlabel('Рост (см)')
plt.ylabel('Количество учеников')
plt.legend()
plt.grid(alpha=0.3)

# 2. Гистограмма веса
plt.subplot(1, 3, 2)
plt.hist(weights, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(weights), color='red', linestyle='--', linewidth=2, label=f'Среднее: {np.mean(weights):.1f} кг')
plt.title('Распределение веса учеников')
plt.xlabel('Вес (кг)')
plt.ylabel('Количество учеников')
plt.legend()
plt.grid(alpha=0.3)

# 3. Диаграмма рассеяния
plt.subplot(1, 3, 3)
plt.scatter(heights, weights, alpha=0.6, s=50, c='green', edgecolors='black')

# Добавляем линию тренда
z = np.polyfit(heights, weights, 1)
p = np.poly1d(z)
plt.plot(heights, p(heights), "r--", linewidth=2, label=f'Тренд: y = {z[0]:.2f}x + {z[1]:.1f}')

plt.title('Зависимость веса от роста')
plt.xlabel('Рост (см)')
plt.ylabel('Вес (кг)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ корреляции
correlation = np.corrcoef(heights, weights)[0, 1]
print(f"\nСтатистический анализ:")
print(f"Корреляция рост-вес: {correlation:.3f}")
print(f"Средний рост: {np.mean(heights):.1f} ± {np.std(heights):.1f} см")
print(f"Средний вес: {np.mean(weights):.1f} ± {np.std(weights):.1f} кг")

if correlation > 0.7:
    print("Сильная положительная корреляция!")
elif correlation > 0.3:
    print("Умеренная положительная корреляция")
else:
    print("Слабая корреляция")
```

---

## Дополнительное задание: Комплексный анализ (Бонус)

### Задание 8.1: Анализ успеваемости класса

**Подробное описание и подсказки:**
1. Сгенерируйте матрицу оценок для 25 учеников по 5 предметам с помощью `np.random.choice([2, 3, 4, 5], size=(25, 5), p=[0.05, 0.25, 0.45, 0.25])`. Используйте фиксированный seed: `np.random.seed(100)`.
2. Найдите средние оценки по предметам с помощью `np.mean(..., axis=0)`.
3. Постройте столбчатую диаграмму средних оценок.
4. Постройте гистограмму всех оценок.
5. Найдите средний балл каждого ученика с помощью `np.mean(..., axis=1)` и постройте график.
6. Постройте тепловую карту оценок с помощью `plt.imshow()`.
7. Найдите корреляцию между предметами с помощью `np.corrcoef()`.
8. Постройте круговую диаграмму структуры оценок.
9. Проанализируйте статистику: среднее, медиану, стандартное отклонение, лучших и худших учеников, качество знаний.

**Подсказки:**
- Для всех пунктов используйте функции и примеры из шпаргалки.
- Для визуализации используйте `plt.bar()`, `plt.hist()`, `plt.plot()`, `plt.imshow()`, `plt.pie()`.

```python
# Ваш код здесь
# 1. Сгенерируйте матрицу оценок
# 2. Найдите средние оценки по предметам
# 3. Постройте столбчатую диаграмму
# 4. Постройте гистограмму всех оценок
# 5. Найдите средний балл каждого ученика и постройте график
# 6. Постройте тепловую карту оценок
# 7. Найдите корреляцию между предметами
# 8. Постройте круговую диаграмму структуры оценок
# 9. Проанализируйте статистику класса
```

### Решение 8.1:
```python
# Генерируем данные об успеваемости
np.random.seed(100)
students = [f"Ученик_{i+1}" for i in range(25)]
subjects = ['Математика', 'Физика', 'Русский язык', 'История', 'Биология']

# Создаем матрицу оценок (25 учеников × 5 предметов)
grades_matrix = np.random.choice([2, 3, 4, 5], size=(25, 5), p=[0.05, 0.25, 0.45, 0.25])

print("Комплексный анализ успеваемости класса")
print("=" * 50)

# Анализ по предметам
plt.figure(figsize=(16, 12))

# 1. Средние оценки по предметам (столбчатая диаграмма)
plt.subplot(2, 3, 1)
subject_means = np.mean(grades_matrix, axis=0)
bars = plt.bar(subjects, subject_means, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
plt.title('Средние оценки по предметам')
plt.ylabel('Средняя оценка')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 5)
for bar, mean in zip(bars, subject_means):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')

# 2. Распределение оценок (гистограмма)
plt.subplot(2, 3, 2)
all_grades = grades_matrix.flatten()
plt.hist(all_grades, bins=[1.5, 2.5, 3.5, 4.5, 5.5], alpha=0.7, color='lightblue', edgecolor='black')
plt.title('Распределение всех оценок')
plt.xlabel('Оценка')
plt.ylabel('Количество')
plt.xticks([2, 3, 4, 5])

# 3. Успеваемость каждого ученика (средний балл)
plt.subplot(2, 3, 3)
student_means = np.mean(grades_matrix, axis=1)
plt.plot(range(1, 26), student_means, 'o-', color='green', markersize=4)
plt.axhline(np.mean(student_means), color='red', linestyle='--', label=f'Общее среднее: {np.mean(student_means):.2f}')
plt.title('Средний балл каждого ученика')
plt.xlabel('Номер ученика')
plt.ylabel('Средний балл')
plt.legend()
plt.grid(alpha=0.3)

# 4. Тепловая карта оценок
plt.subplot(2, 3, 4)
im = plt.imshow(grades_matrix.T, cmap='RdYlGn', aspect='auto', vmin=2, vmax=5)
plt.colorbar(im, label='Оценка')
plt.title('Тепловая карта оценок')
plt.xlabel('Ученики')
plt.ylabel('Предметы')
plt.yticks(range(5), subjects)

# 5. Корреляция между предметами
plt.subplot(2, 3, 5)
correlation_matrix = np.corrcoef(grades_matrix.T)
im2 = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im2, label='Корреляция')
plt.title('Корреляция между предметами')
plt.xticks(range(5), subjects, rotation=45, ha='right')
plt.yticks(range(5), subjects)

# 6. Круговая диаграмма качества знаний
plt.subplot(2, 3, 6)
grade_counts = np.bincount(all_grades)[2:]  # считаем с оценки 2
grade_labels = ['Двойки', 'Тройки', 'Четвёрки', 'Пятёрки']
colors_pie = ['#FF4444', '#FFA500', '#4169E1', '#32CD32']
plt.pie(grade_counts, labels=grade_labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
plt.title('Структура оценок в классе')

plt.tight_layout()
plt.show()

# Детальная статистика
print(f"\nДетальная статистика:")
print(f"Общее количество оценок: {len(all_grades)}")
print(f"Средняя оценка в классе: {np.mean(all_grades):.2f}")
print(f"Медиана: {np.median(all_grades)}")
print(f"Стандартное отклонение: {np.std(all_grades):.2f}")

print(f"\nПо предметам:")
for i, subject in enumerate(subjects):
    subject_grades = grades_matrix[:, i]
    print(f"{subject}: среднее {np.mean(subject_grades):.2f}, медиана {np.median(subject_grades)}")

print(f"\nЛучшие и худшие ученики:")
best_student_idx = np.argmax(student_means)
worst_student_idx = np.argmin(student_means)
print(f"Лучший ученик: {students[best_student_idx]} (средний балл: {student_means[best_student_idx]:.2f})")
print(f"Слабейший ученик: {students[worst_student_idx]} (средний балл: {student_means[worst_student_idx]:.2f})")

# Качество знаний
excellent_students = np.sum(student_means >= 4.5)
good_students = np.sum((student_means >= 3.5) & (student_means < 4.5))
weak_students = np.sum(student_means < 3.5)

print(f"\nКачество знаний:")
print(f"Отличники (≥4.5): {excellent_students} учеников ({excellent_students/25*100:.1f}%)")
print(f"Хорошисты (3.5-4.5): {good_students} учеников ({good_students/25*100:.1f}%)")
print(f"Слабые ученики (<3.5): {weak_students} учеников ({weak_students/25*100:.1f}%)")
```


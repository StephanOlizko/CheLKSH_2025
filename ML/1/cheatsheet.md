# Полная шпаргалка по NumPy, Matplotlib и Pandas
# Jupyter Notebook с подробными объяснениями всех базовых команд

# =============================================================================
# NUMPY - Работа с многомерными массивами и математическими операциями
# =============================================================================

# Импорт библиотеки NumPy с стандартным алиасом np
import numpy as np

# Создание одномерного массива из списка Python
# np.array() преобразует последовательность (список, кортеж) в ndarray
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D массив:", arr_1d)

# Создание двумерного массива из вложенных списков
# Каждый вложенный список становится строкой в 2D массиве
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D массив:\n", arr_2d)

# Создание трёхмерного массива из вложенных списков
# Первое измерение - глубина, второе - строки, третье - столбцы
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3D массив:\n", arr_3d)

# Проверка формы (размерности) массива
# .shape возвращает кортеж с размерами по каждой оси
print("Форма 1D массива:", arr_1d.shape)
print("Форма 2D массива:", arr_2d.shape)
print("Форма 3D массива:", arr_3d.shape)

# Проверка количества измерений массива
# .ndim возвращает количество осей (измерений) массива
print("Размерность 1D массива:", arr_1d.ndim)
print("Размерность 2D массива:", arr_2d.ndim)

# Проверка типа данных элементов массива
# .dtype показывает тип данных (int64, float64, и т.д.)
print("Тип данных массива:", arr_1d.dtype)

# Проверка общего количества элементов в массиве
# .size возвращает произведение всех размерностей
print("Общее количество элементов:", arr_2d.size)

# Создание массива из нулей указанной формы
# np.zeros() создаёт массив заполненный нулями
zeros_array = np.zeros((3, 4))
print("Массив из нулей:\n", zeros_array)

# Создание массива из единиц указанной формы
# np.ones() создаёт массив заполненный единицами
ones_array = np.ones((2, 3))
print("Массив из единиц:\n", ones_array)

# Создание массива заполненного определённым значением
# np.full() создаёт массив заданной формы с указанным значением
full_array = np.full((2, 3), 7)
print("Массив заполненный семёрками:\n", full_array)

# Создание единичной матрицы (диагональная матрица с единицами)
# np.eye() создаёт квадратную матрицу с 1 на диагонали и 0 везде
identity_matrix = np.eye(3)
print("Единичная матрица 3x3:\n", identity_matrix)

# Создание последовательности чисел с указанным шагом
# np.arange(start, stop, step) аналог range() для массивов NumPy
range_array = np.arange(0, 10, 2)
print("Массив с шагом 2:", range_array)

# Создание массива из равномерно распределённых чисел
# np.linspace(start, stop, num) создаёт num точек от start до stop включительно
linspace_array = np.linspace(0, 1, 5)
print("Линейно распределённые числа:", linspace_array)

# Создание массива случайных чисел от 0 до 1
# np.random.random() генерирует случайные числа из равномерного распределения
random_array = np.random.random((2, 3))
print("Случайные числа:\n", random_array)

# Создание массива случайных целых чисел в диапазоне
# np.random.randint(low, high, size) генерирует целые числа от low до high-1
random_int = np.random.randint(1, 10, (2, 3))
print("Случайные целые числа:\n", random_int)

# Создание массива из нормального (гауссова) распределения
# np.random.normal(mean, std, size) генерирует числа с заданным средним и стандартным отклонением
normal_array = np.random.normal(0, 1, (2, 3))
print("Нормальное распределение:\n", normal_array)

# Доступ к элементу массива по индексу (начиная с 0)
# Для многомерных массивов используется [row, column] или [axis0, axis1, ...]
element = arr_2d[1, 2]  # строка 1, столбец 2
print("Элемент [1,2]:", element)

# Срезы массива (slicing) - получение подмассива
# [start:stop:step] работает как в списках Python
slice_1d = arr_1d[1:4]  # элементы с индекса 1 по 3
print("Срез 1D массива:", slice_1d)

# Срезы для многомерных массивов
# [row_start:row_stop, col_start:col_stop]
slice_2d = arr_2d[0:2, 1:3]  # первые 2 строки, столбцы 1-2
print("Срез 2D массива:\n", slice_2d)

# Получение всей строки или столбца
# : означает "все элементы по этой оси"
row = arr_2d[1, :]  # вторая строка целиком
print("Вторая строка:", row)

column = arr_2d[:, 1]  # второй столбец целиком
print("Второй столбец:", column)

# Булевое индексирование - фильтрация по условию
# Создаём булевый массив и используем его как маску
condition = arr_1d > 3
filtered = arr_1d[condition]
print("Элементы больше 3:", filtered)

# Fancy indexing - индексирование массивом индексов
# Можно передать список/массив индексов для получения элементов
indices = [0, 2, 4]
fancy_indexed = arr_1d[indices]
print("Элементы по индексам [0,2,4]:", fancy_indexed)

# Изменение формы массива без изменения данных
# .reshape() возвращает новый вид массива с другой формой
reshaped = arr_1d.reshape(5, 1)  # из (5,) в (5,1)
print("Изменённая форма:\n", reshaped)

# Преобразование многомерного массива в одномерный
# .flatten() создаёт копию в виде 1D массива
flattened = arr_2d.flatten()
print("Сплющенный массив:", flattened)

# Альтернативный способ сплющивания (возвращает view, не копию)
# .ravel() более эффективен, но изменения влияют на оригинал
raveled = arr_2d.ravel()
print("Ravel массив:", raveled)

# Транспонирование массива (поворот матрицы)
# .T меняет местами строки и столбцы
transposed = arr_2d.T
print("Транспонированный массив:\n", transposed)

# Альтернативный способ транспонирования
# .transpose() может менять порядок осей для многомерных массивов
transposed_alt = arr_2d.transpose()
print("Transpose альтернативный:\n", transposed_alt)

# Поэлементное сложение массивов одинакового размера
# Операции выполняются поэлементно (element-wise)
arr_a = np.array([1, 2, 3])
arr_b = np.array([4, 5, 6])
addition = arr_a + arr_b
print("Сложение массивов:", addition)

# Поэлементное вычитание массивов
subtraction = arr_b - arr_a
print("Вычитание массивов:", subtraction)

# Поэлементное умножение массивов
# Это НЕ матричное произведение, а поэлементное
multiplication = arr_a * arr_b
print("Поэлементное умножение:", multiplication)

# Поэлементное деление массивов
division = arr_b / arr_a
print("Деление массивов:", division)

# Возведение в степень поэлементно
power = arr_a ** 2
print("Возведение в квадрат:", power)

# Операции с скаляром (broadcasting)
# Скаляр применяется ко всем элементам массива
scalar_add = arr_a + 10
print("Добавление скаляра:", scalar_add)

scalar_mult = arr_a * 3
print("Умножение на скаляр:", scalar_mult)

# Матричное произведение двух массивов
# np.dot() или @ выполняют матричное умножение
mat_a = np.array([[1, 2], [3, 4]])
mat_b = np.array([[5, 6], [7, 8]])
dot_product = np.dot(mat_a, mat_b)
print("Матричное произведение:\n", dot_product)

# Альтернативный синтаксис для матричного умножения
dot_product_alt = mat_a @ mat_b
print("Матричное произведение (@):\n", dot_product_alt)

# Нахождение суммы всех элементов массива
# np.sum() или .sum() суммирует элементы
total_sum = np.sum(arr_1d)
print("Сумма всех элементов:", total_sum)

# Сумма по определённой оси в многомерном массиве
# axis=0 - по строкам (сумма столбцов), axis=1 - по столбцам (сумма строк)
sum_axis0 = np.sum(arr_2d, axis=0)  # сумма по столбцам
print("Сумма по столбцам:", sum_axis0)

sum_axis1 = np.sum(arr_2d, axis=1)  # сумма по строкам
print("Сумма по строкам:", sum_axis1)

# Нахождение среднего арифметического
# np.mean() вычисляет среднее значение
mean_value = np.mean(arr_1d)
print("Среднее значение:", mean_value)

# Нахождение медианы (среднего элемента при сортировке)
median_value = np.median(arr_1d)
print("Медиана:", median_value)

# Нахождение стандартного отклонения
std_value = np.std(arr_1d)
print("Стандартное отклонение:", std_value)

# Нахождение дисперсии
var_value = np.var(arr_1d)
print("Дисперсия:", var_value)

# Нахождение минимального элемента
min_value = np.min(arr_1d)
print("Минимальное значение:", min_value)

# Нахождение максимального элемента
max_value = np.max(arr_1d)
print("Максимальное значение:", max_value)

# Нахождение индекса минимального элемента
# np.argmin() возвращает индекс первого минимального элемента
min_index = np.argmin(arr_1d)
print("Индекс минимума:", min_index)

# Нахождение индекса максимального элемента
max_index = np.argmax(arr_1d)
print("Индекс максимума:", max_index)

# Сортировка массива по возрастанию
# np.sort() возвращает отсортированную копию
unsorted = np.array([3, 1, 4, 1, 5, 9, 2])
sorted_array = np.sort(unsorted)
print("Отсортированный массив:", sorted_array)

# Получение индексов для сортировки
# np.argsort() возвращает индексы, которые отсортируют массив
sort_indices = np.argsort(unsorted)
print("Индексы сортировки:", sort_indices)

# Нахождение уникальных элементов
# np.unique() возвращает отсортированный массив уникальных значений
unique_elements = np.unique(unsorted)
print("Уникальные элементы:", unique_elements)

# Проверка условий для каждого элемента
# np.where() возвращает элементы по условию или применяет функцию
condition_result = np.where(arr_1d > 3, arr_1d, 0)  # если > 3, то значение, иначе 0
print("Условная замена:", condition_result)

# Объединение массивов по горизонтали (столбцы)
# np.concatenate() с axis=1 или np.hstack()
arr_left = np.array([[1, 2], [3, 4]])
arr_right = np.array([[5, 6], [7, 8]])
h_concat = np.hstack([arr_left, arr_right])
print("Горизонтальное объединение:\n", h_concat)

# Объединение массивов по вертикали (строки)
# np.concatenate() с axis=0 или np.vstack()
arr_top = np.array([[1, 2, 3]])
arr_bottom = np.array([[4, 5, 6]])
v_concat = np.vstack([arr_top, arr_bottom])
print("Вертикальное объединение:\n", v_concat)

# Разделение массива на части по горизонтали
# np.hsplit() делит массив на равные части по столбцам
to_split = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
h_split = np.hsplit(to_split, 3)  # на 3 части
print("Горизонтальное разделение:")
for i, part in enumerate(h_split):
    print(f"Часть {i}:\n", part)

# Разделение массива на части по вертикали
# np.vsplit() делит массив на равные части по строкам
to_split_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v_split = np.vsplit(to_split_v, 2)  # на 2 части
print("Вертикальное разделение:")
for i, part in enumerate(v_split):
    print(f"Часть {i}:\n", part)

# =============================================================================
# MATPLOTLIB - Создание графиков и визуализации данных
# =============================================================================

import matplotlib.pyplot as plt

# Создание простого линейного графика
# plt.plot() строит линейный график по точкам (x, y)
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Синусоида")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()

# Создание графика с несколькими линиями
# Можно вызвать plot() несколько раз или передать несколько наборов данных
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.title("Синус и косинус")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()  # показать легенду
plt.grid(True)  # показать сетку
plt.show()

# Настройка стиля линий
# Можно задать цвет, стиль линии, маркеры
plt.plot(x, y, 'r--', linewidth=2)  # красная пунктирная линия
plt.plot(x, np.cos(x), 'b:', marker='o', markersize=3)  # синяя точечная с кружками
plt.title("Стили линий")
plt.show()

# Создание точечного графика (scatter plot)
# plt.scatter() рисует точки без соединяющих линий
x_scatter = np.random.normal(0, 1, 100)
y_scatter = np.random.normal(0, 1, 100)
plt.scatter(x_scatter, y_scatter, alpha=0.6)  # alpha - прозрачность
plt.title("Точечный график")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Создание гистограммы
# plt.hist() показывает распределение значений в виде столбцов
data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30, alpha=0.7, color='green')  # bins - количество столбцов
plt.title("Гистограмма нормального распределения")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.show()

# Создание столбчатой диаграммы
# plt.bar() создаёт вертикальные столбцы
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values, color=['red', 'green', 'blue', 'orange'])
plt.title("Столбчатая диаграмма")
plt.ylabel("Значения")
plt.show()

# Создание горизонтальной столбчатой диаграммы
plt.barh(categories, values, color='purple')  # barh - horizontal bar
plt.title("Горизонтальная диаграмма")
plt.xlabel("Значения")
plt.show()

# Создание круговой диаграммы
# plt.pie() создаёт круговую диаграмму с долями
sizes = [15, 30, 45, 10]
labels = ['Группа A', 'Группа B', 'Группа C', 'Группа D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Круговая диаграмма")
plt.axis('equal')  # делает круг круглым
plt.show()

# Создание подграфиков (subplots)
# plt.subplot() позволяет разместить несколько графиков на одной фигуре
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 сетка графиков

# График в позиции (0,0)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title("sin(x)")

# График в позиции (0,1)  
axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title("cos(x)")

# График в позиции (1,0)
axes[1, 0].scatter(x_scatter[:50], y_scatter[:50])
axes[1, 0].set_title("Scatter plot")

# График в позиции (1,1)
axes[1, 1].hist(data, bins=20)
axes[1, 1].set_title("Histogram")

plt.tight_layout()  # автоматически подгоняет отступы
plt.show()

# Сохранение графика в файл
# plt.savefig() сохраняет текущую фигуру в различных форматах
plt.plot(x, y)
plt.title("График для сохранения")
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')  # высокое качество, обрезать поля
plt.show()

# Настройка размера фигуры
# plt.figure() создаёт новую фигуру с заданными параметрами
plt.figure(figsize=(12, 6))  # ширина 12, высота 6 дюймов
plt.plot(x, y)
plt.title("Широкий график")
plt.show()

# Настройка осей
# Можно задать пределы осей, метки, шкалу
plt.plot(x, y)
plt.xlim(0, 5)  # пределы по x
plt.ylim(-1.5, 1.5)  # пределы по y
plt.xticks([0, 1, 2, 3, 4, 5])  # конкретные метки по x
plt.yticks([-1, -0.5, 0, 0.5, 1])  # конкретные метки по y
plt.title("График с настроенными осями")
plt.show()

# Добавление аннотаций и стрелок
# plt.annotate() добавляет текст с указанием на точку
plt.plot(x, y)
plt.annotate('Максимум', xy=(np.pi/2, 1), xytext=(2, 1.2),
            arrowprops=dict(arrowstyle='->', color='red'))
plt.title("График с аннотацией")
plt.show()

# Создание тепловой карты (heatmap)
# plt.imshow() может отображать 2D массивы как изображения
data_2d = np.random.random((10, 10))
plt.imshow(data_2d, cmap='hot', interpolation='nearest')
plt.colorbar()  # добавить цветовую шкалу
plt.title("Тепловая карта")
plt.show()

# Создание 3D графика (требует дополнительный импорт)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D поверхность
x_3d = np.linspace(-5, 5, 50)
y_3d = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_3d, y_3d)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("3D поверхность")
plt.show()

# Настройка стиля графиков
# plt.style.use() применяет предустановленные стили
available_styles = plt.style.available
print("Доступные стили:", available_styles[:5])  # показать первые 5

plt.style.use('seaborn-v0_8')  # используем стиль seaborn
plt.plot(x, y)
plt.title("График в стиле seaborn")
plt.show()

# =============================================================================  
# PANDAS - Работа с табличными данными и анализ данных
# =============================================================================

import pandas as pd

# Создание Series (одномерная структура данных)
# Series - это одномерный массив с индексами
series = pd.Series([1, 2, 3, 4, 5])
print("Простая Series:")
print(series)

# Создание Series с пользовательским индексом
series_indexed = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print("\nSeries с индексом:")
print(series_indexed)

# Создание Series из словаря
# Ключи становятся индексами, значения - данными
dict_data = {'apple': 5, 'banana': 3, 'orange': 8, 'grape': 2}
series_dict = pd.Series(dict_data)
print("\nSeries из словаря:")
print(series_dict)

# Создание DataFrame из словаря
# DataFrame - это двумерная таблица с именованными столбцами и строками
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['Moscow', 'SPb', 'Moscow', 'Kazan'],
    'Salary': [50000, 60000, 75000, 55000]
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)

# Создание DataFrame из списка словарей
# Каждый словарь представляет одну строку
list_of_dicts = [
    {'Name': 'Eve', 'Age': 32, 'City': 'Moscow'},
    {'Name': 'Frank', 'Age': 28, 'City': 'SPb'},
    {'Name': 'Grace', 'Age': 29, 'City': 'Kazan'}
]
df_from_list = pd.DataFrame(list_of_dicts)
print("\nDataFrame из списка словарей:")
print(df_from_list)

# Просмотр первых n строк DataFrame
# .head(n) показывает первые n строк (по умолчанию 5)
print("\nПервые 3 строки:")
print(df.head(3))

# Просмотр последних n строк DataFrame  
# .tail(n) показывает последние n строк (по умолчанию 5)
print("\nПоследние 2 строки:")
print(df.tail(2))

# Получение информации о DataFrame
# .info() показывает типы данных, количество непустых значений, память
print("\nИнформация о DataFrame:")
df.info()

# Получение описательной статистики
# .describe() показывает count, mean, std, min, quartiles, max для числовых столбцов
print("\nОписательная статистика:")
print(df.describe())

# Просмотр формы DataFrame (количество строк и столбцов)
print(f"\nРазмер DataFrame: {df.shape}")

# Просмотр названий столбцов
print(f"Столбцы: {df.columns.tolist()}")

# Просмотр индекса (названий строк)
print(f"Индекс: {df.index.tolist()}")

# Выбор одного столбца (возвращает Series)
# Можно использовать df['column'] или df.column
names = df['Name']
print("\nСтолбец Name:")
print(names)
print(f"Тип: {type(names)}")

# Выбор нескольких столбцов (возвращает DataFrame)
# Передаём список названий столбцов
subset = df[['Name', 'Age']]
print("\nВыбранные столбцы:")
print(subset)

# Выбор строк по индексу с помощью .loc
# .loc[row_indexer, column_indexer] использует названия индексов
row_by_loc = df.loc[1]  # вторая строка (индекс 1)
print("\nСтрока с индексом 1:")
print(row_by_loc)

# Выбор строк по позиции с помощью .iloc
# .iloc[row_position, column_position] использует числовые позиции
row_by_iloc = df.iloc[2]  # третья строка (позиция 2)
print("\nСтрока на позиции 2:")  
print(row_by_iloc)

# Выбор конкретного элемента
# Можно указать строку и столбец
element = df.loc[1, 'Name']  # строка 1, столбец Name
print(f"\nЭлемент [1, 'Name']: {element}")

# Выбор диапазона строк и столбцов
# Используем слайсы с .loc или .iloc
slice_data = df.loc[1:3, 'Name':'City']  # строки 1-3, столбцы от Name до City
print("\nСрез данных:")
print(slice_data)

# Фильтрация данных по условию
# Создаём булевую маску и применяем её
age_filter = df['Age'] > 28
filtered_df = df[age_filter]
print("\nЛюди старше 28:")
print(filtered_df)

# Множественные условия фильтрации
# Используем & (и), | (или), ~ (не). Условия в скобках!
complex_filter = (df['Age'] > 25) & (df['City'] == 'Moscow')
complex_filtered = df[complex_filter]
print("\nМосквичи старше 25:")
print(complex_filtered)

# Сортировка по одному столбцу
# .sort_values() сортирует по указанному столбцу
sorted_by_age = df.sort_values('Age')
print("\nСортировка по возрасту:")
print(sorted_by_age)

# Сортировка по убыванию
sorted_desc = df.sort_values('Age', ascending=False)
print("\nСортировка по возрасту (убывание):")
print(sorted_desc)
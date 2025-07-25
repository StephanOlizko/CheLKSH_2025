# Занятие: Современные IDE и VS Code  
**Аудитория:** 6 класс  
**Время:** 90 минут (2 академических часа)  
**Уровень:** Базовые знания ПК, опыт написания простых программ

---

## 0:00–0:10 — Вступление и знакомство

- Приветствие, рассказ о теме занятия.
- Кратко: "Сегодня мы узнаем, что такое IDE, зачем они нужны, и научимся работать в одной из самых популярных сред — VS Code."
- Вопрос аудитории: "Кто уже писал код? В какой программе? Какие трудности были?"

---

## 0:10–0:25 — Что такое IDE и зачем они нужны?

### (0:10–0:17) — Погружение в мир разработки

- Процесс программирования — очень непростой. Есть множество уровней абстракции: от низких, где инженеры собирают электрические схемы, до высоких, где программисты пишут промышленные приложения.
- Скорее всего, ни один человек не понимает, как всё работает полностью, но, объединяя усилия, мы можем запускать Fortnite на "магическом куске камня".
- Благо есть множество инструментов, о которых уже подумали другие люди, они существенно упрощают жизнь разработчикам ПО.
- По сути, всё, что делает средний разработчик — это работа с текстом. Это не сильно отличается от работы с документами в Word.
- Можно было бы писать программы в Word или на листочке, но это жутко неудобно. Почему?
    - Много работы мы отдаём интерпретатору (например, в Python), который сообщает об ошибках, хотя мог бы просто выполняться дальше. Было бы круто? Нет, потому что тогда появляется риск затронуть рабочие программы и другой код, вплоть до полной "смерти" компьютера.
    - Есть анализаторы синтаксиса, которые подсказывают, где не закрыта скобка и т.д.

### (0:17–0:25) — IDE: зачем и какие бывают

- IDE (интегрированная среда разработки) — это "умный" редактор, который помогает писать код быстрее и безопаснее.
- Примеры IDE: Geany, VS Code, PyCharm, IntelliJ IDEA, Eclipse.
- Geany — неплохой базовый редактор, но сильно ограниченный.
- Есть знакомый, который даже на 3-4 курсе университета писал программы в Geany — и вот только что закончил бакалавриат!
- Мейнстрим в IDE сейчас — это VS Code от Microsoft или редакторы от JetBrains.
- Лично я использую VS Code, и вот почему:
    - Он проще, чем редакторы от JetBrains.
    - Универсальный: можно писать практически на любом языке, установив нужные расширения.
    - Много полезных аддонов: контроль версий, автодополнение, интеграция с терминалом и многое другое.
- Сегодня мы научимся работать в VS Code, разберём базовые вещи и полезные фишки.

---

## 0:25–0:45 — Интерфейс VS Code: основные элементы и практика

### (0:25–0:33) — Структура окна

- Левая боковая панель — навигация по проекту (файлы, поиск, контроль версий, расширения).
- Левая выдвигающаяся панель — дополнительные инструменты (например, поиск по проекту).
- Нижняя выдвигающаяся панель — терминал, вывод, отладка.
- Окно редактора текста — основное рабочее пространство.
- Демонстрация: открываем проект, показываем структуру.

### (0:33–0:38) — Шорткаты и управление окнами

- Быстрое открытие/закрытие боковых панелей: `Ctrl+B`
- Открытие терминала: `` Ctrl+` ``
- Переключение между файлами: `Ctrl+Tab`
- Поиск по файлу: `Ctrl+F`
- Поиск по проекту: `Ctrl+Shift+F`
- Замена по файлу: `Ctrl+H`
- Быстрый переход к файлу: `Ctrl+P`
- Разделение редактора: `Ctrl+\`
- Демонстрация: показать, как быстро искать и переключаться.

### (0:38–0:45) — Практика: редактирование кода

- VS Code подсказывает при написании кода: автодополнение, подсветка ошибок, описание функций при наведении.
- Подчеркивание ошибок — сразу видно, где опечатка или синтаксическая ошибка.
- Наведение мыши — всплывающие подсказки по функциям и переменным.
- Поиск по файлу (`Ctrl+F`), по проекту (`Ctrl+Shift+F`), замена всех вхождений (`Ctrl+H`).
- Практика: откройте файл, попробуйте найти и заменить слово, используйте автодополнение и подсказки.

---

#### **Добавление: Работа с объемным примером**

- Откройте папку `examples` и найдите файл `contacts.py` (пример программы — простая записная книжка на Python).
- Прочитайте код, разберитесь, как он устроен: есть функции для добавления, поиска и вывода контактов.
- Используйте поиск по проекту, чтобы найти все места, где используется функция `add_contact`.
- Измените название функции на `add_new_contact` во всех местах с помощью поиска и массовой замены.
- Добавьте новую функцию для удаления контакта (можно вместе с классом).
- Проверьте, как работает автодополнение при вызове новых функций.
- Запустите программу через терминал, попробуйте добавить и вывести контакт.
- Если есть ошибки — используйте подсказки VS Code для их исправления.
- Откройте два файла рядом (например, `contacts.py` и `utils.py`), попробуйте скопировать функцию из одного файла в другой.
- Используйте "Go to Definition" (F12), чтобы быстро перейти к определению функции.
- Попробуйте поставить точку останова и запустить дебаггер для пошагового выполнения программы.

---

## 0:45–1:00 — Терминал в VS Code и работа с проектом

### (0:45–0:50) — Терминал как часть IDE

- Терминал — один из самых базовых способов взаимодействия с компьютером с помощью текста.
- Большая часть действий в графическом интерфейсе — это "обёртка" для команд терминала.
- В VS Code терминал встроен прямо в окно редактора.
- Демонстрация: запуск Python-скрипта через кнопку и через терминал.

### (0:50–1:00) — Практика: работа с проектом

- Откройте проект в VS Code.
- Создайте новую папку и файл через терминал.
- Напишите небольшой код (например, "Hello, world!").
- Используйте подсказки, автодополнение, поиск.
- Запустите код через терминал.
- Попробуйте найти ошибку и исправить её с помощью поиска и автозамены.

---

## 1:00–1:25 — Мини-квест: "IDE-детектив"

### Легенда (1:00–1:02)

Вы — разработчик, которому нужно быстро разобраться в чужом проекте и внести исправления.

### Задание 1 (1:02–1:08): Навигация и поиск

- Откройте проект в VS Code.
- Найдите файл с функцией, которая вызывает ошибку (например, функция с опечаткой).
- Используйте поиск по проекту, чтобы найти все вхождения этой функции.

### Задание 2 (1:08–1:13): Исправление ошибок

- Исправьте опечатку в названии функции.
- Используйте автодополнение, чтобы убедиться, что функция теперь вызывается правильно во всех местах.

### Задание 3 (1:13–1:18): Массовая замена

- Используйте функцию "замена по проекту", чтобы заменить все старые вхождения на новое имя.

### Задание 4 (1:18–1:25): Проверка и запуск

- Запустите проект или отдельный скрипт через терминал.
- Убедитесь, что ошибок больше нет.

---

## 1:25–1:40 — Дополнительные задания и фишки VS Code

1. Откройте настройки VS Code и измените цветовую тему редактора.
2. Установите расширение для работы с Python или другим языком.
3. Настройте автосохранение файлов.
4. Используйте встроенный терминал для запуска нескольких команд подряд (например, создать папку, создать файл, вывести содержимое).
5. Попробуйте открыть два файла рядом и редактировать их одновременно (разделение редактора).
6. Используйте функцию "Go to Definition" (F12) для перехода к определению функции или переменной.
7. Попробуйте использовать встроенный дебаггер для запуска и пошагового выполнения кода.
8. Настройте горячие клавиши под себя (File → Preferences → Keyboard Shortcuts).

---

## 1:40–1:45 — Заключение и домашнее задание

- IDE — это ваш главный инструмент для эффективной разработки.
- VS Code — универсальный, бесплатный и мощный редактор.
- Терминал — незаменимая часть работы программиста.

**Домашнее задание:**  
- Установить VS Code дома.
- Открыть свой проект или создать новый.
- Попробовать все основные шорткаты и функции поиска.
- Написать небольшой скрипт и запустить его через терминал в VS Code.
- Поменять цветовую тему и установить хотя бы одно расширение.

---

## Материалы для подготовки

- Установленный VS Code на компьютерах
- Проектор для демонстрации
- Раздаточные материалы с шорткатами и командами
- Пример проекта для практики

---
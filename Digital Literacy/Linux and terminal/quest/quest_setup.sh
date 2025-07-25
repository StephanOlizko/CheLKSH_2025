#!/bin/bash
# quest_setup.sh - подготовка квеста

# Создаем основные директории для квеста
mkdir -p /tmp/tech_records
mkdir -p /secret/archives
mkdir -p /var/tmp/gaming_data

# Рекорд 1: Программирование
cat > /tmp/tech_records/coding_record.data << 'EOF'
ТЕХНОЛОГИЧЕСКИЙ РЕКОРД: Самый длинный марафон программирования

Рекордсмен: Алвин Ашкрафт (США)
Продолжительность: 135 часов 14 минут 38 секунд
Дата установления: 2014 год

Во время марафона программист создал игру "Asteroids" с нуля,
используя только язык C и библиотеку SDL. За почти 6 дней
непрерывной работы он написал более 3000 строк кода.

Код доступа: MARATHON135
EOF

# Рекорд 2: Видеоигры
cat > /var/tmp/gaming_data/gaming_record.data << 'EOF'
ИГРОВОЙ РЕКОРД: Самая дорогостоящая видеоигра в разработке

Проект: Grand Theft Auto V
Бюджет разработки: 265 миллионов долларов США
Время разработки: 5 лет
Размер команды: более 1000 специалистов

Интересный факт: бюджет превысил затраты на производство
большинства голливудских блокбастеров того времени.

Код доступа: GTAV265M
EOF

# Рекорд 3: Интернет технологии
cat > /secret/archives/internet_record.data << 'EOF'
СЕТЕВОЙ РЕКОРД: Максимальная скорость передачи данных

Достижение: 178 терабит в секунду
Место: Япония, 2020 год
Технология: оптоволоконные каналы связи

При такой скорости можно передать весь каталог Netflix
за одну секунду. Это в миллион раз быстрее обычного
домашнего интернет-соединения.

Код доступа: FIBER178TB
EOF

# Финальный рекорд
cat > /secret/archives/final_record.txt << 'EOF'
СЕКРЕТНЫЙ РЕКОРД: Самая молодая команда разработчиков приложения

Проект: мобильная игра "Pocket Pixels" для iOS
Возраст участников: от 10 до 14 лет
Дата: 2019 год
Команда: 4 школьника из Канады

Их приложение было скачано более 100,000 раз в первый месяц.
Этот случай доказывает, что программированию действительно
нет возрастных ограничений.

Поздравляем! Теперь вы знаете, что даже школьники могут
устанавливать технологические рекорды.

Сертификат исследователя: RECORD_HUNTER_2025
EOF

# Создаем проверочный скрипт
cat > quest_checker.sh << 'EOF'
#!/bin/bash
# Проверяльщик заданий квеста

USER_QUEST_DIR="./record_hunt"

check_research_base() {
    echo "Проверка исследовательской базы..."
    
    if [ -d "$USER_QUEST_DIR" ] && [ -d "$USER_QUEST_DIR/equipment" ] && [ -d "$USER_QUEST_DIR/database" ] && [ -d "$USER_QUEST_DIR/records" ] && [ -d "$USER_QUEST_DIR/reports" ]; then
        echo "Исследовательская база готова к работе."
        echo ""
        echo "ПЕРВАЯ ПОДСКАЗКА:"
        echo "Данные о рекордах находятся в трех локациях:"
        echo "• /tmp/tech_records/ - технические достижения"
        echo "• /var/tmp/gaming_data/ - игровые рекорды"  
        echo "• /secret/archives/ - засекреченная информация"
        echo ""
        echo "ЗАДАНИЕ: Найдите все файлы с расширением .data и скопируйте их в папку records"
        return 0
    else
        echo "Ошибка: создайте структуру record_hunt/{equipment,database,records,reports}"
        echo "в текущей директории квеста"
        return 1
    fi
}

check_records_found() {
    echo "Проверка собранных данных..."
    
    local records_count=$(find "$USER_QUEST_DIR/records" -name "*.data" 2>/dev/null | wc -l)
    
    if [ "$records_count" -ge 3 ]; then
        echo "Все записи о рекордах успешно найдены и каталогизированы."
        echo ""
        echo "ВТОРАЯ ПОДСКАЗКА:"
        echo "Каждая запись содержит уникальный код доступа."
        echo "Изучите содержимое файлов и извлеките коды."
        echo ""
        echo "ЗАДАНИЕ: Создайте файл 'access_codes.txt' в папке reports"
        echo "Запишите туда все найденные коды (по одному на строку)"
        return 0
    else
        echo "Ошибка: найдено записей $records_count из 3. Продолжите поиск файлов .data"
        return 1
    fi
}

check_codes_analyzed() {
    echo "Проверка анализа кодов доступа..."
    
    if [ -f "$USER_QUEST_DIR/reports/access_codes.txt" ]; then
        local codes_count=$(wc -l < "$USER_QUEST_DIR/reports/access_codes.txt")
        if [ "$codes_count" -ge 3 ]; then
            echo "Коды доступа успешно извлечены и систематизированы."
            echo ""
            echo "ТРЕТЬЯ ПОДСКАЗКА:"
            echo "Профессиональные исследователи ведут детальную документацию."
            echo "Составьте аналитический отчет о найденных рекордах."
            echo ""
            echo "ЗАДАНИЕ: Создайте файл 'record_analysis.txt' в папке reports"
            echo "Опишите каждый рекорд (минимум 3 строки текста)"
            return 0
        fi
    fi
    
    echo "Ошибка: создайте файл access_codes.txt в папке reports и внесите все коды"
    return 1
}

check_analysis_complete() {
    echo "Проверка аналитического отчета..."
    
    if [ -f "$USER_QUEST_DIR/reports/record_analysis.txt" ]; then
        local lines_count=$(wc -l < "$USER_QUEST_DIR/reports/record_analysis.txt")
        if [ "$lines_count" -ge 3 ]; then
            echo "Аналитический отчет составлен на профессиональном уровне."
            echo ""
            echo "ФИНАЛЬНАЯ ПОДСКАЗКА:"
            echo "В секретных архивах /secret/archives/ хранится особая запись."
            echo "Найдите файл с самым вдохновляющим технологическим достижением."
            echo ""
            echo "ФИНАЛЬНОЕ ЗАДАНИЕ: Получите доступ к секретному архиву"
            return 0
        fi
    fi
    
    echo "Ошибка: создайте record_analysis.txt с описанием рекордов (минимум 3 строки)"
    return 1
}

check_secret_record() {
    echo "Проверка доступа к секретному архиву..."
    
    if [ -f "$USER_QUEST_DIR/final_record.txt" ]; then
        echo "КВЕСТ ЗАВЕРШЕН УСПЕШНО"
        echo ""
        echo "Выполненные задачи:"
        echo "• Создана исследовательская база данных"
        echo "• Найдены все технологические рекорды"  
        echo "• Извлечены секретные коды доступа"
        echo "• Составлен профессиональный отчет"
        echo "• Получен доступ к секретному архиву"
        echo ""
        echo "Поздравляем! Вы получили сертификат исследователя технологических рекордов."
        return 0
    else
        echo "Ошибка: найдите и скопируйте секретную запись в папку record_hunt"
        return 1
    fi
}

# Главная логика проверки
case $1 in
    "1"|"base") check_research_base ;;
    "2"|"records") check_records_found ;;
    "3"|"codes") check_codes_analyzed ;;
    "4"|"analysis") check_analysis_complete ;;
    "5"|"secret") check_secret_record ;;
    *)
        echo "Квест Linux и терминал"
        echo ""
        echo "Команды проверки:"
        echo "  ./quest_checker.sh 1  - проверить исследовательскую базу"
        echo "  ./quest_checker.sh 2  - проверить сбор рекордов"
        echo "  ./quest_checker.sh 3  - проверить коды доступа"
        echo "  ./quest_checker.sh 4  - проверить аналитический отчет"
        echo "  ./quest_checker.sh 5  - проверить секретный архив"
        ;;
esac
EOF

chmod +x quest_checker.sh
chmod +r /tmp/tech_records/*
chmod +r /var/tmp/gaming_data/*
chmod +r /secret/archives/*

# Генерируем скрипт для очистки всех файлов квеста
cat > quest_cleanup.sh << 'EOF'
#!/bin/bash
# quest_cleanup.sh - удаляет все артефакты и результаты квеста

echo "Удаление файлов квеста..."

# Удаляем артефакты из системных директорий
rm -rf /tmp/tech_records
rm -rf /var/tmp/gaming_data
rm -rf /secret/archives

# Удаляем рабочую папку участника (если запускать из папки квеста)
rm -rf ./record_hunt
rm ./quest_checker.sh
rm ./quest_tasks.txt
rm ./quest_cleanup.sh

echo "Очистка завершена."
EOF

chmod +x quest_cleanup.sh

# Генерируем файл с текстами заданий для студентов
cat > quest_tasks.txt << 'EOF'
ЗАДАНИЯ КВЕСТА "ОХОТНИКИ ЗА РЕКОРДАМИ"

=== Задание 1: Создание исследовательской базы ===
Ваша роль: Вы — исследователь технологических рекордов, которому поручено создать базу для изучения удивительных достижений в мире IT.

Что нужно сделать:
1. Создайте папку с названием record_hunt — это будет ваша исследовательская база.
2. Перейдите в эту папку.
3. Создайте внутри четыре подпапки:
   - equipment — для инструментов исследования
   - database — для систематизации данных
   - records — для хранения найденных записей
   - reports — для аналитических отчетов
4. Проверьте, что структура создана правильно.
5. Вернитесь в основную папку квеста.
6. Запустите проверку: ./quest_checker.sh 1

Подсказка: Используйте команды mkdir, cd, ls -la.

=== Задание 2: Поиск технологических рекордов ===
Ваша миссия: В системе спрятаны файлы с данными о трех выдающихся технологических рекордах. Найдите их!

Что нужно сделать:
1. Найдите все файлы с расширением .data в системе.
2. Проверьте три основные локации: /tmp/, /var/tmp/ и /secret/.
3. Скопируйте каждый найденный файл в папку record_hunt/records/.
4. Проверьте, что все файлы успешно скопированы.
5. Запустите проверку: ./quest_checker.sh 2

Подсказка: Команда find поможет найти файлы по шаблону, а cp — скопировать их.

=== Задание 3: Извлечение секретных кодов ===
Детективная работа: Каждая запись о рекорде содержит секретный код доступа. Найдите их все!

Что нужно сделать:
1. Прочитайте содержимое каждого файла .data в папке records.
2. Найдите в каждом файле строку "Код доступа:" и запомните код.
3. Создайте файл access_codes.txt в папке reports.
4. Запишите в него все найденные коды (каждый код на отдельной строке).
5. Проверьте содержимое созданного файла.
6. Запустите проверку: ./quest_checker.sh 3

Подсказка: cat покажет содержимое файла, echo с > и >> поможет записать данные.

=== Задание 4: Составление аналитического отчета ===
Научная работа: Профессиональные исследователи всегда документируют свои открытия.

Что нужно сделать:
1. Создайте файл record_analysis.txt в папке reports.
2. Напишите в него анализ каждого найденного рекорда:
   - Опишите рекорд марафона программирования (135 часов).
   - Проанализируйте бюджет разработки GTA V ($265 млн).
   - Объясните значение рекорда скорости интернета (178 терабит/с).
3. Каждое описание должно содержать минимум одну строку.
4. Проверьте количество строк в файле.
5. Запустите проверку: ./quest_checker.sh 4

Подсказка: Используйте nano для удобного редактирования или echo для быстрой записи. Команда wc -l покажет количество строк.

=== Задание 5: Секретный архив ===
Финальное открытие: Существует один особый рекорд, который может вас вдохновить!

Что нужно сделать:
1. Найдите в папке /secret/archives/ файл с расширением .txt.
2. Прочитайте его содержимое — это самый вдохновляющий рекорд!
3. Скопируйте этот файл в папку record_hunt (в корень, не в подпапку).
4. Запустите финальную проверку: ./quest_checker.sh 5

Награда: Узнаете о рекорде самой молодой команды разработчиков и получите сертификат исследователя!

=== Теория: grep и конвейер (|) ===

Команда grep используется для поиска строк в файлах по заданному шаблону (регулярному выражению).

Основные флаги grep:
    -i   — игнорировать регистр букв при поиске (например, "Рекорд" и "рекорд" будут найдены одинаково)
    -o   — выводить только совпадающие фрагменты, а не всю строку
    -h   — не выводить имя файла перед найденной строкой (удобно при поиске в нескольких файлах)
    -E   — использовать расширенные регулярные выражения (например, для поиска по шаблону с +, {n,})
    -r   — искать рекурсивно во всех вложенных папках
    -v   — выводить только строки, которые НЕ соответствуют шаблону

Конвейер (|) — это оператор, который позволяет передать вывод одной команды на вход другой.
Например: команда1 | команда2

Примеры использования grep и конвейера:

1. Поиск всех строк с "код доступа" во всех .data-файлах:
    grep "Код доступа" *.data

2. Поиск без учёта регистра:
    grep -i "рекорд" *.data

3. Подсчёт количества файлов, где встречается слово "игра":
    grep -l "игра" *.data | wc -l

4. Вывод только самих кодов доступа (после двоеточия):
    grep "Код доступа" *.data | cut -d':' -f2

5. Поиск всех строк, не содержащих слово "рекорд":
    grep -v "рекорд" *.data

6. Рекурсивный поиск слова "бюджет" во всех файлах в папке:
    grep -r "бюджет" .

7. Использование расширенных регулярных выражений:
    grep -E "код|бюджет" *.data

8. Поиск и подсчёт количества уникальных слов "Рекорд" в файлах:
    grep -o "Рекорд" *.data | wc -l

9. Поиск строк с числом (цифрами) в файле:
    grep -E "[0-9]+" файл.txt

10. Поиск строк, где слово "игра" встречается в начале строки:
    grep "^игра" *.data

Примеры использования конвейеров (|):

1. Найти строки с "Код доступа" и вывести только коды:
    grep "Код доступа" *.data | cut -d':' -f2

2. Найти все строки с "рекорд", привести к нижнему регистру и отсортировать:
    grep "рекорд" *.data | tr '[:upper:]' '[:lower:]' | sort

3. Подсчитать количество уникальных слов "игра" во всех файлах:
    grep -o "игра" *.data | sort | uniq -c

4. Найти строки с числами и вывести только уникальные числа:
    grep -Eo "[0-9]+" *.data | sort -n | uniq

5. Найти строки с "бюджет", вывести только суммы (цифры):
    grep "бюджет" *.data | grep -Eo "[0-9]+"

6. Найти все строки с "код", отсортировать и убрать дубликаты:
    grep -i "код" *.data | sort | uniq

7. Найти строки с "игра", вывести только имена файлов:
    grep -l "игра" *.data | sort

8. Найти строки с "рекорд", посчитать количество таких строк:
    grep "рекорд" *.data | wc -l

Команды можно объединять с помощью | для последовательной обработки данных.


EOF

echo "Квест настроен. Можно начинать с команды ./quest_checker.sh"
echo "Для удаления всех файлов квеста используйте ./quest_cleanup.sh"
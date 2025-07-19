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
        echo "Квест 'Охотники за рекордами'"
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

echo "Квест настроен. Студенты могут начинать с команды ./quest_checker.sh"
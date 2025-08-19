#!/bin/bash
# quest_cleanup.sh - удаляет все артефакты и результаты квеста

echo "Удаление файлов квеста..."

# Удаляем артефакты из домашней директории
rm -rf ~/temp/tech_records
rm -rf ~/var_temp/gaming_data
rm -rf ~/secret/archives

# Удаляем рабочую папку участника (если запускать из папки квеста)
rm -rf ./record_hunt
rm ./quest_checker.sh
rm ./quest_tasks.txt
rm ./quest_cleanup.sh

echo "Очистка завершена."

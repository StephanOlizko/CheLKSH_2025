"""
Простая записная книжка с меню, поиском, удалением и сохранением в файл.
В программе специально есть опечатка в названии функции (add_contatc вместо add_contact).
"""

contacts = []

def add_contatc(name, phone, email=""):
    """Добавить контакт (с опечаткой в названии функции!)"""
    contacts.append({"name": name, "phone": phone, "email": email})

def find_contact(name):
    """Найти контакт по имени"""
    for c in contacts:
        if c["name"].lower() == name.lower():
            return c
    return None

def show_contacts():
    """Показать все контакты"""
    if not contacts:
        print("Список контактов пуст.")
        return
    print("Контакты:")
    for idx, c in enumerate(contacts, 1):
        print(f"{idx}. {c['name']} | {c['phone']} | {c['email']}")

def delete_contact(name):
    """Удалить контакт по имени"""
    global contacts
    before = len(contacts)
    contacts = [c for c in contacts if c["name"].lower() != name.lower()]
    if len(contacts) < before:
        print(f"Контакт '{name}' удалён.")
    else:
        print(f"Контакт '{name}' не найден.")

def update_contact(name, new_phone=None, new_email=None):
    """Обновить телефон или email контакта"""
    c = find_contact(name)
    if c:
        if new_phone:
            c["phone"] = new_phone
        if new_email:
            c["email"] = new_email
        print(f"Контакт '{name}' обновлён.")
    else:
        print(f"Контакт '{name}' не найден.")

def save_contacts(filename):
    """Сохранить контакты в файл"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for c in contacts:
                f.write(f"{c['name']},{c['phone']},{c['email']}\n")
        print(f"Контакты сохранены в {filename}")
    except Exception as e:
        print("Ошибка при сохранении:", e)

def load_contacts(filename):
    """Загрузить контакты из файла"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                name, phone, email = line.strip().split(",", 2)
                contacts.append({"name": name, "phone": phone, "email": email})
        print(f"Контакты загружены из {filename}")
    except FileNotFoundError:
        print("Файл не найден, начинаем с пустого списка.")
    except Exception as e:
        print("Ошибка при загрузке:", e)

def menu():
    """Главное меню"""
    while True:
        print("\nМеню:")
        print("1. Показать все контакты")
        print("2. Добавить контакт")
        print("3. Найти контакт")
        print("4. Удалить контакт")
        print("5. Обновить контакт")
        print("6. Сохранить контакты в файл")
        print("7. Загрузить контакты из файла")
        print("0. Выйти")
        choice = input("Выберите действие: ")
        if choice == "1":
            show_contacts()
        elif choice == "2":
            name = input("Имя: ")
            phone = input("Телефон: ")
            email = input("Email (необязательно): ")
            add_contatc(name, phone, email)  # <-- опечатка!
        elif choice == "3":
            name = input("Имя для поиска: ")
            c = find_contact(name)
            if c:
                print(f"Найден: {c['name']} | {c['phone']} | {c['email']}")
            else:
                print("Контакт не найден.")
        elif choice == "4":
            name = input("Имя для удаления: ")
            delete_contact(name)
        elif choice == "5":
            name = input("Имя для обновления: ")
            phone = input("Новый телефон (или Enter): ")
            email = input("Новый email (или Enter): ")
            update_contact(name, phone if phone else None, email if email else None)
        elif choice == "6":
            filename = input("Имя файла для сохранения: ")
            save_contacts(filename)
        elif choice == "7":
            filename = input("Имя файла для загрузки: ")
            load_contacts(filename)
        elif choice == "0":
            print("До свидания!")
            break
        else:
            print("Неизвестная команда.")

if __name__ == "__main__":
    print("Добро пожаловать в записную книжку!")
    menu()
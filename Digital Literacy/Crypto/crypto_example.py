import hashlib
import random
import math

def simple_hash(text):
    """
    Простая хэш-функция, возвращающая число
    
    Математическое обоснование:
    Это полиномиальная хэш-функция с основанием 31.
    Для строки s = s₀s₁s₂...sₙ₋₁ вычисляется:
    h(s) = s₀×31^(n-1) + s₁×31^(n-2) + ... + sₙ₋₁×31^0
    
    Почему 31?
    - 31 простое число, что уменьшает коллизии
    - 31 = 32 - 1, что позволяет компилятору оптимизировать умножение
    - Экспериментально доказано, что дает хорошее распределение
    
    Свойства:
    - Детерминирована: одинаковый вход → одинаковый выход
    - Учитывает порядок символов: "AB" ≠ "BA"
    - Быстрая вычислимость: O(n) времени
    """
    hash_value = 0
    for char in text:
        hash_value = hash_value * 31 + ord(char)
    return hash_value

def better_hash(text):
    """
    Улучшенная хэш-функция с использованием SHA-256
    
    Математическое обоснование:
    SHA-256 основан на криптографических принципах:
    1. Использует операции XOR, сдвиги, модульное сложение
    2. Применяет 64 раунда преобразований с константами
    3. Каждый бит выхода зависит от каждого бита входа
    
    Криптографические свойства:
    - Односторонность: практически невозможно найти x для данного h(x)
    - Устойчивость к коллизиям: сложно найти x₁ ≠ x₂ где h(x₁) = h(x₂)
    - Эффект лавины: изменение 1 бита входа меняет ~50% битов выхода
    - Равномерное распределение: все значения хэша равновероятны
    
    Почему берем только первые 8 символов:
    - Полный SHA-256 дает 256 бит (64 hex символа)
    - 8 hex символов = 32 бита = 4,294,967,296 возможных значений
    - Достаточно для демонстрации, но компромисс между скоростью и безопасностью
    """
    sha256_hash = hashlib.sha256(text.encode()).hexdigest()
    # Преобразуем первые 8 символов hex в число
    return int(sha256_hash[:8], 16)

def gcd(a, b):
    """
    Алгоритм Евклида для нахождения НОД
    
    Математическое обоснование:
    Теорема: gcd(a, b) = gcd(b, a mod b)
    
    Доказательство:
    Пусть d = gcd(a, b) и d' = gcd(b, a mod b)
    
    1) a = bq + r, где r = a mod b
    2) Любой общий делитель a и b также делит r = a - bq
    3) Любой общий делитель b и r также делит a = bq + r
    4) Следовательно, множества общих делителей (a,b) и (b,r) совпадают
    5) Значит, их наибольшие общие делители равны: d = d'
    
    Сложность: O(log(min(a, b))) - очень быстро даже для больших чисел
    
    Пример:
    gcd(48, 18):
    48 = 18×2 + 12  →  gcd(18, 12)
    18 = 12×1 + 6   →  gcd(12, 6)
    12 = 6×2 + 0    →  gcd(6, 0) = 6
    """
    while b:
        a, b = b, a % b
    return a

def mod_inverse(e, phi):
    """
    Нахождение модульного обратного числа
    
    Математическое обоснование:
    Ищем число d такое, что: e × d ≡ 1 (mod φ)
    
    Расширенный алгоритм Евклида:
    Для любых a, b существуют x, y такие что: ax + by = gcd(a, b)
    
    Если gcd(e, φ) = 1, то существуют x, y: ex + φy = 1
    Это означает: ex ≡ 1 (mod φ), где x и есть искомое d
    
    Почему это работает в RSA:
    1) Выбираем e взаимно простое с φ(n) = (p-1)(q-1)
    2) Находим d = e⁻¹ (mod φ)
    3) Тогда ed ≡ 1 (mod φ)
    4) По теореме Эйлера: m^(ed) ≡ m^1 ≡ m (mod n)
    5) Это обеспечивает правильное шифрование/расшифровку
    
    Пример:
    e = 3, φ = 40
    Ищем d: 3d ≡ 1 (mod 40)
    Расширенный Евклид: 3×27 + 40×(-2) = 1
    Значит d = 27, проверка: 3×27 = 81 ≡ 1 (mod 40) ✓
    """
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(e, phi)
    if gcd != 1:
        raise ValueError("Модульное обратное не существует")
    return (x % phi + phi) % phi

def is_prime(n):
    """
    Проверка числа на простоту (детерминированный алгоритм)
    
    Математическое обоснование:
    Теорема: Если n составное, то у него есть делитель ≤ √n
    
    Доказательство:
    Пусть n = a × b, где a, b > 1
    Если a > √n и b > √n, то a × b > √n × √n = n
    Противоречие! Значит, хотя бы один из делителей ≤ √n
    
    Алгоритм:
    1) Проверяем все числа от 2 до √n
    2) Если нашли делитель - число составное
    3) Если не нашли - число простое
    
    Сложность: O(√n) - для больших чисел медленно, но точно
    
    Оптимизации (не реализованы здесь):
    - Проверять только 2, потом нечетные числа
    - Проверять только простые делители (решето Эратосфена)
    - Вероятностные тесты (Миллер-Рабин) для больших чисел
    
    Примеры:
    n = 17: проверяем 2,3,4 (до √17 ≈ 4.1) - делителей нет → простое
    n = 15: проверяем 2,3 - находим 3 → составное (15 = 3×5)
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def generate_prime(min_val, max_val):
    """Генерация простого числа в диапазоне"""
    while True:
        num = random.randint(min_val, max_val)
        if is_prime(num):
            return num

def generate_rsa_keys():
    """Генерация пары RSA ключей"""
    # Выбираем два простых числа (маленькие для демонстрации)
    p = generate_prime(50, 100)
    q = generate_prime(50, 100)
    
    # Вычисляем n и функцию Эйлера
    n = p * q
    phi = (p - 1) * (q - 1)
    
    # Выбираем e (начинаем с небольшого числа для простоты)
    e = 3
    while gcd(e, phi) != 1:
        e += 2  # Увеличиваем на 2, чтобы оставаться нечетным
    
    # Вычисляем d
    d = mod_inverse(e, phi)
    
    public_key = (e, n)
    private_key = (d, n)
    
    return public_key, private_key, p, q

def rsa_encrypt(message, public_key):
    """Шифрование сообщения с помощью публичного ключа"""
    e, n = public_key
    # Преобразуем сообщение в число
    if isinstance(message, str):
        message_num = sum(ord(c) for c in message) % (n - 1) + 1
    else:
        message_num = message % (n - 1) + 1
    
    encrypted = pow(message_num, e, n)
    return encrypted, message_num

def rsa_decrypt(encrypted_message, private_key):
    """Расшифровка сообщения с помощью приватного ключа"""
    d, n = private_key
    decrypted = pow(encrypted_message, d, n)
    return decrypted

def create_digital_signature(message, private_key):
    """Создание цифровой подписи"""
    # Сначала хэшируем сообщение
    message_hash = better_hash(message)
    d, n = private_key
    
    # Приводим хэш к диапазону [1, n-1]
    message_hash = message_hash % (n - 1) + 1
    
    # Подписываем хэш приватным ключом
    signature = pow(message_hash, d, n)
    return signature, message_hash

def verify_signature(message, signature, public_key):
    """Проверка цифровой подписи"""
    # Хэшируем исходное сообщение
    message_hash = better_hash(message)
    e, n = public_key
    
    # Приводим хэш к диапазону [1, n-1]
    message_hash = message_hash % (n - 1) + 1
    
    # Расшифровываем подпись публичным ключом
    decrypted_hash = pow(signature, e, n)
    
    # Сравниваем хэши
    return message_hash == decrypted_hash, message_hash, decrypted_hash

def demonstrate_crypto():
    """Демонстрация работы криптографических функций"""
    print("=== Демонстрация хэш-функций ===")
    text = "Привет, блокчейн!"
    print(f"Текст: {text}")
    print(f"Простой хэш: {simple_hash(text)}")
    print(f"SHA-256 хэш: {better_hash(text)}")
    
    print("\n=== Демонстрация RSA шифрования ===")
    
    # Генерируем ключи
    public_key, private_key, p, q = generate_rsa_keys()
    print(f"Простые числа p={p}, q={q}")
    print(f"n = p * q = {p * q}")
    print(f"Публичный ключ (e, n): {public_key}")
    print(f"Приватный ключ (d, n): {private_key}")
    
    # Шифруем сообщение
    message = "Секрет"
    print(f"\nИсходное сообщение: {message}")
    
    encrypted, original_num = rsa_encrypt(message, public_key)
    print(f"Сообщение как число: {original_num}")
    print(f"Зашифрованное сообщение: {encrypted}")
    
    decrypted = rsa_decrypt(encrypted, private_key)
    print(f"Расшифрованное сообщение (число): {decrypted}")
    print(f"Проверка: {original_num == decrypted}")
    
    print("\n=== Демонстрация цифровой подписи ===")
    
    # Создаем подпись
    signature, hash_value = create_digital_signature(message, private_key)
    print(f"Хэш сообщения: {hash_value}")
    print(f"Цифровая подпись: {signature}")
    
    # Проверяем подпись
    is_valid, original_hash, decrypted_hash = verify_signature(message, signature, public_key)
    print(f"Исходный хэш: {original_hash}")
    print(f"Расшифрованный хэш: {decrypted_hash}")
    print(f"Подпись валидна: {is_valid}")
    
    # Проверяем с измененным сообщением
    fake_message = "Поддельное"
    is_fake_valid, fake_hash, _ = verify_signature(fake_message, signature, public_key)
    print(f"\nПроверка поддельного сообщения '{fake_message}':")
    print(f"Хэш поддельного сообщения: {fake_hash}")
    print(f"Подпись для поддельного сообщения валидна: {is_fake_valid}")

def interactive_hash_task():
    """Интерактивное задание на хэширование"""
    print("\n" + "="*50)
    print("ЗАДАНИЕ 1: Хэш-функции")
    print("="*50)
    
    print("Попробуйте хэшировать разные строки и посмотрите на результат!")
    
    while True:
        user_input = input("\nВведите строку для хэширования (или 'exit' для выхода): ")
        if user_input.lower() == 'exit':
            break
            
        simple = simple_hash(user_input)
        better = better_hash(user_input)
        
        print(f"Простой хэш: {simple}")
        print(f"SHA-256 хэш: {better}")
        
        # Показываем, что хэш детерминирован
        print(f"Повторный SHA-256 хэш: {better_hash(user_input)}")
        print("Заметили? Хэш всегда одинаковый для одной строки!")

def interactive_rsa_task():
    """Интерактивное задание на RSA"""
    print("\n" + "="*50)
    print("ЗАДАНИЕ 2: RSA Шифрование")
    print("="*50)
    
    # Генерируем ключи
    public_key, private_key, p, q = generate_rsa_keys()
    
    print("Сгенерированы RSA ключи:")
    print(f"p = {p}, q = {q}")
    print(f"n = {p * q}")
    print(f"Публичный ключ: {public_key}")
    print(f"Приватный ключ: {private_key}")
    
    while True:
        message = input("\nВведите сообщение для шифрования (или 'exit' для выхода): ")
        if message.lower() == 'exit':
            break
            
        try:
            encrypted, original_num = rsa_encrypt(message, public_key)
            decrypted = rsa_decrypt(encrypted, private_key)
            
            print(f"Исходное сообщение: '{message}'")
            print(f"Как число: {original_num}")
            print(f"Зашифровано: {encrypted}")
            print(f"Расшифровано: {decrypted}")
            print(f"Проверка: {original_num == decrypted}")
            
        except Exception as e:
            print(f"Ошибка: {e}")

def quiz():
    """Квиз по криптографии"""
    print("\n" + "="*50)
    print("КВИЗ: Проверьте свои знания!")
    print("="*50)
    
    questions = [
        {
            "question": "Что будет, если изменить хотя бы один символ в сообщении перед хэшированием?",
            "options": ["1. Хэш не изменится", "2. Хэш кардинально изменится", "3. Хэш изменится незначительно"],
            "correct": 2,
            "explanation": "Это называется 'эффект лавины' - малое изменение входа приводит к кардинальному изменению выхода"
        },
        {
            "question": "Что нужно для создания цифровой подписи?",
            "options": ["1. Публичный ключ", "2. Приватный ключ", "3. Оба ключа"],
            "correct": 2,
            "explanation": "Для создания подписи используется приватный ключ, для проверки - публичный"
        },
        {
            "question": "Можно ли восстановить исходное сообщение из его хэша?",
            "options": ["1. Да, всегда", "2. Нет, это невозможно", "3. Иногда"],
            "correct": 2,
            "explanation": "Хэш-функция работает только в одну сторону - это необратимая функция"
        }
    ]
    
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"\nВопрос {i}: {q['question']}")
        for option in q['options']:
            print(option)
        
        while True:
            try:
                answer = int(input("Ваш ответ (1, 2 или 3): "))
                if answer in [1, 2, 3]:
                    break
                else:
                    print("Введите 1, 2 или 3")
            except ValueError:
                print("Введите число!")
        
        if answer == q['correct']:
            print("✅ Правильно!")
            score += 1
        else:
            print("❌ Неправильно.")
        
        print(f"Объяснение: {q['explanation']}")
    
    print(f"\nВаш результат: {score}/{len(questions)}")
    if score == len(questions):
        print("🎉 Отлично! Вы разбираетесь в криптографии!")
    elif score >= len(questions) // 2:
        print("👍 Неплохо! Продолжайте изучать!")
    else:
        print("📚 Стоит повторить материал!")

if __name__ == "__main__":
    print("🔐 ДОБРО ПОЖАЛОВАТЬ В МИР КРИПТОГРАФИИ! 🔐")
    print("\nСначала посмотрим демонстрацию:")
    
    demonstrate_crypto()
    
    print("\nТеперь попробуйте сами!")
    
    # Интерактивные задания
    interactive_hash_task()
    interactive_rsa_task()
    quiz()
    
    print("\n🎓 Поздравляем! Вы изучили основы криптографии!")
    print("Помните: в реальном мире используются гораздо более сложные алгоритмы и большие числа!")
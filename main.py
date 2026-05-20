import time
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

DATABASE_URL = "mysql+mysqlconnector://titanic_user:titanic_password_123@localhost:3307/my_database"

MAX_RETRIES = 10
RETRY_DELAY = 10 

def main():
    print("Ініціалізація підключення до бази даних...")
    
    engine = create_engine(DATABASE_URL)
    
    connection_successful = False
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Спроба {attempt}/{MAX_RETRIES}: Підключення до MySQL...")
            
            with engine.connect() as connection:
                print("Успішно підключено до бази даних!")
                connection_successful = True
                break 
                
        except OperationalError as e:
            print(f"База даних ще не готова або недоступна. Помилка: {e}")
            if attempt < MAX_RETRIES:
                print(f"Очікування {RETRY_DELAY} секунд перед наступною спробою...\n")
                time.sleep(RETRY_DELAY)
            else:
                print("\n[ПОМИЛКА] Досягнуто максимальну кількість спроб підключення.")
                
    if not connection_successful:
        print("Не вдалося підключитися до бази даних. Завершення роботи скрипту.")
        return

    try:
        print("Зчитування даних з таблиці 'titanic' у pandas.DataFrame...")
        query = "SELECT * FROM titanic;"
        
        df = pd.read_sql_query(query, con=engine)
        
        print("\n" + "="*50)
        print("ОЧІКУВАНИЙ РЕЗУЛЬТАТ (Перші 5 рядків):")
        print("="*50)
        print(df.head())
        
        print("\n" + "="*50)
        print("ІНФОРМАЦІЯ ПРО ДАТАСЕТ:")
        print("="*50)
        print(f"Кількість рядків: {df.shape[0]}")
        print(f"Кількість колонок: {df.shape[1]}")
        
        print("\nКількість порожніх значень (NULL) у проблемних колонках:")
        print(df[['Age', 'Cabin', 'Embarked']].isna().sum())
        
    except Exception as e:
        print(f"Виникла помилка під час зчитування даних: {e}")

if __name__ == "__main__":
    main()
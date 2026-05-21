"""
МКР з Python for Data Science — наскрізний кейс «Метеослужба».
Індивідуальний варіант для Docker-тегу: asterindex/pfds-mkr-g5-15:latest
"""

# ====================================================================
# Прізвище, ім'я, по батькові: Нємцов Олександр Вадимович
# Група:                       КІ-32
# Дата виконання:              2026-05-21
# ====================================================================

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# Параметри підключення до персонального Docker-контейнера
DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 33306 
DB_NAME = "meteo"

# Налаштування папки для графіків
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 15, delay: float = 3.0) -> pd.DataFrame:
    """Підключитися до MySQL і завантажити таблицю observations.

    Контейнер виконує ініціалізацію, тому реалізовано retry-цикл.
    """
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(url)
    
    print("Очікування ініціалізації бази даних MySQL у контейнері...")
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Успішно підключено з {attempt}-ї спроби. Завантажено рядків: {len(df)}")
            return df
        except OperationalError:
            if attempt == retries:
                print("\nПомилка: Не вдалося підключитися до MySQL. Перевірте, чи запущено контейнер.")
                raise
            print(f"  База ще завантажується (спроба {attempt}/{retries}). Очікування {delay} сек...")
            time.sleep(delay)
    raise RuntimeError("Не вдалося підключитися")


# ====================================================================
# БЛОК 1. NumPy (15 балів)
# ====================================================================

def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy (Робота з сирими даними)")

    # Перетворюємо серії у чисті NumPy масиви
    t_raw = df_raw['temperature_c'].to_numpy(dtype=float)
    rh_raw = df_raw['humidity_pct'].to_numpy(dtype=float)
    ws_raw = df_raw['wind_speed_ms'].to_numpy(dtype=float)
    obs_ids = df_raw['obs_id'].to_numpy()
    datetimes = df_raw['datetime'].to_numpy()

    # 1) Розрахунок відчутної температури (Apparent Temperature)
    apparent = t_raw - (100.0 - rh_raw) / 5.0
    
    # Оскільки дані сирі, для min/max ігноруємо NaN за допомогою np.nanmin/max
    print(f"1) T_app масив: довжина={len(apparent)}, min={np.nanmin(apparent):.2f}°C, max={np.nanmax(apparent):.2f}°C")

    # 2) Заміна фізичних викидів на np.nan через np.where
    # Температура: > 60 або < -60
    t_outlier_mask = (t_raw > 60.0) | (t_raw < -60.0)
    temperature_clean = np.where(t_outlier_mask, np.nan, t_raw)
    
    # Швидкість вітру: > 100
    ws_outlier_mask = ws_raw > 100.0
    wind_clean = np.where(ws_outlier_mask, np.nan, ws_raw)
    
    print(f"2) Кількість замінених викидів температури (±999): {np.sum(t_outlier_mask)}")
    print(f"   Кількість замінених викидів швидкості вітру (999): {np.sum(ws_outlier_mask)}")

    # 3) Розрахунок описової статистики вручну (ігноруючи NaN)
    mean_t = np.nanmean(temperature_clean)
    median_t = np.nanmedian(temperature_clean)
    std_t = np.nanstd(temperature_clean)
    print(f"3) Статистика чистої T (ручний розрахунок):")
    print(f"   Mean={mean_t:.3f} | Median={median_t:.3f} | Std={std_t:.3f}")

    # 4) Кількість морозних (T < 0) та жарких (T > 30) спостережень
    # Використовуємо очищену від 999 температуру, щоб викиди не спотворили статистику
    n_frost = np.sum(temperature_clean < 0.0)
    n_hot = np.sum(temperature_clean > 30.0)
    print(f"4) Кількість екстремальних спостережень: Морозних (T<0)={n_frost}, Жарких (T>30)={n_hot}")

    # 5) Пошук індексів глобального мінімуму та максимуму температури
    idx_max = np.nanargmax(temperature_clean)
    idx_min = np.nanargmin(temperature_clean)
    
    print(f"5) Максимум температури: {temperature_clean[idx_max]:.1f}°C (obs_id: {obs_ids[idx_max]}, дата: {datetimes[idx_max]})")
    print(f"   Мінімум температури: {temperature_clean[idx_min]:.1f}°C (obs_id: {obs_ids[idx_min]}, дата: {datetimes[idx_min]})")


# ====================================================================
# БЛОК 2. Pandas — очищення (20 балів)
# ====================================================================

def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — Очищення даних")

    rows_before = len(df_raw)
    df = df_raw.copy()

    # 1) Перевірка типів та структури даних
    print("1) Структура даних до очищення (.info()):")
    df.info()

    # 2) Конвертація в datetime та встановлення індексу
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # 3) Видалення повних дублікатів рядків
    # Оскільки datetime тепер в індексі, drop_duplicates перевіряє дублікати колонок.
    # Щоб врахувати і індекс, скинемо його тимчасово або скористаємось перевіркою всіх полів.
    rows_with_idx = df.reset_index()
    duplicated_mask = rows_with_idx.duplicated()
    n_dups = duplicated_mask.sum()
    
    # Переприсвоюємо очищений від дублів датафрейм
    df = rows_with_idx.drop_duplicates().set_index('datetime')
    print(f"2) Видалення дублікатів: знайдено та видалено {n_dups} повних дублів.")

    # 4) Розумне заповнення пропусків (NaN) у humidity_pct медіаною по місяцю в межах міста
    df['month'] = df.index.month
    
    # Рахуємо скільки NaN було у вологості
    nan_humidity_before = df['humidity_pct'].isna().sum()
    
    # Заповнюємо через трансформацію групи
    df['humidity_pct'] = df.groupby(['city', 'month'])['humidity_pct'].transform(lambda s: s.fillna(s.median()))
    n_filled = nan_humidity_before - df['humidity_pct'].isna().sum()
    print(f"3) Заповнення пропусків вологості: відновлено {n_filled} значень `NULL`.")

    # 5) Фільтрація фізичних викидів
    # Температура має бути в межах від -60 до 60
    # Швидкість вітру має бути від 0 до 60 АБО залишатися NaN (якщо сенсор відмовив, але це не викид 999)
    valid_temp = (df['temperature_c'] >= -60.0) & (df['temperature_c'] <= 60.0)
    valid_wind = (df['wind_speed_ms'].isna()) | ((df['wind_speed_ms'] >= 0.0) & (df['wind_speed_ms'] <= 60.0))
    
    df_clean = df[valid_temp & valid_wind].copy()
    n_outliers = len(df) - len(df_clean)
    print(f"4) Видалення викидів: видалено {n_outliers} рядків із критичними збоями сенсорів (±999).")

    # 6) Фінальний звіт очищення
    print(f"\n   ФІНАЛЬНИЙ ЗВІТ ОЧИЩЕННЯ: {rows_before} рядків -> {len(df_clean)} рядків збережено.")
    return df_clean


# ====================================================================
# БЛОК 3. Pandas — аналітика (30 балів)
# ====================================================================

def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — Кліматична аналітика")

    # 1) Середня температура по містах
    by_city_temp = df.groupby('city')['temperature_c'].mean().sort_values(ascending=False)
    print("1) Середня температура по містах (від найтеплішого до найхолоднішого):")
    print(by_city_temp.round(2).to_string())

    # 2) Сумарні опади по містах
    by_city_precip = df.groupby('city')['precipitation_mm'].sum().sort_values(ascending=False)
    print("\n2) Сумарний об'єм опадів по містах за 2 роки:")
    print(by_city_precip.round(1).to_string())

    # 3) Місячна середня температура (використовуємо 'ME' або 'M' відповідно до версії)
    try:
        monthly_mean = df['temperature_c'].resample('ME').mean()
    except ValueError:
        monthly_mean = df['temperature_c'].resample('M').mean()
    
    print(f"\n3) Місячний тренд середньої температури (всього {len(monthly_mean)} місяців):")
    print(monthly_mean.head(6).round(2).to_string(), "\n...")

    # 4) Pivot Table: місто × місяць (середня температура)
    pivot = df.pivot_table(values='temperature_c', index='city', columns='month', aggfunc='mean')
    print("\n4) Зведена таблиця (Pivot) середньої температури «Місто × Календарний місяць»:")
    print(pivot.round(1).to_string())

    # 5) Кількість днів з опадами > 5 мм по містах
    # Крок А: ресемплимо дані по днях для КОЖНОГО міста окремо, рахуючи суму опадів за добу
    daily_precip = df.groupby(['city', pd.Grouper(freq='D')])['precipitation_mm'].sum().reset_index()
    # Крок Б: відбираємо дні, де опадів більше 5мм, та групуємо за містом
    rainy_days = daily_precip[daily_precip['precipitation_mm'] > 5.0].groupby('city').size().sort_values(ascending=False)
    print("\n5) Кількість днів із сильними опадами (> 5 мм за добу):")
    print(rainy_days.to_string())

    # 6) ПОШУК КЛІМАТИЧНОЇ АНОМАЛІЇ
    # Крок А: Обчислюємо кліматичну норму для кожного з 12 місяців року (середнє за 2 роки)
    norm = df.groupby('month')['temperature_c'].mean()
    
    # Крок Б: Отримуємо фактичну середню температуру для кожного конкретного місяця в хронології
    try:
        actual_monthly = df.groupby(['city', pd.Grouper(freq='ME')])['temperature_c'].mean().reset_index()
    except ValueError:
        actual_monthly = df.groupby(['city', pd.Grouper(freq='M')])['temperature_c'].mean().reset_index()
        
    actual_monthly['cal_month'] = actual_monthly['datetime'].dt.month
    actual_monthly['year'] = actual_monthly['datetime'].dt.year
    
    # Крок В: Мапимо норму на фактичні місяці та вираховуємо девіацію (відхилення)
    actual_monthly['norm'] = actual_monthly['cal_month'].map(norm)
    actual_monthly['deviation'] = actual_monthly['temperature_c'] - actual_monthly['norm']
    actual_monthly['abs_deviation'] = actual_monthly['deviation'].abs()
    
    # Крок Г: Знаходимо рядок з максимальним абсолютним відхиленням
    idx_anomaly = actual_monthly['abs_deviation'].idxmax()
    anomaly_row = actual_monthly.loc[idx_anomaly]
    
    anomaly_month = f"{int(anomaly_row['year'])}-{int(anomaly_row['cal_month']):02d} ({anomaly_row['city']})"
    anomaly_dev = anomaly_row['deviation']
    
    print(f"\n6) ВИЯВЛЕНО АНОМАЛІЮ: {anomaly_month} | Відхилення від норми: {anomaly_dev:+.2f}°C")

    # Додатково порахуємо стабільність регіонів (std) для висновків звіту
    city_std = df.groupby('city')['temperature_c'].std().sort_values()
    print("\n   Додатково для звіту (Стандартне відхилення температури по містах):")
    print(city_std.round(2).to_string())

    return {
        "by_city_temp": by_city_temp,
        "by_city_precip": by_city_precip,
        "monthly_mean": monthly_mean,
        "pivot": pivot,
        "anomaly_month": anomaly_month,
        "anomaly_dev": anomaly_dev,
        "city_std": city_std
    }


# ====================================================================
# БЛОК 4. Matplotlib + збереження (35 балів)
# ====================================================================

def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Візуалізація результатів (Matplotlib)")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --- Графік 1: Line plot (Місячна динаміка для 3 міст) ---
    fig, ax = plt.subplots(figsize=(11, 5))
    cities_to_plot = list(analytics['by_city_temp'].index[:3])  # беремо топ-3 найтепліших міст
    
    try:
        resampled_data = df.groupby(['city', pd.Grouper(freq='ME')])['temperature_c'].mean()
    except ValueError:
        resampled_data = df.groupby(['city', pd.Grouper(freq='M')])['temperature_c'].mean()
        
    for city in cities_to_plot:
        city_series = resampled_data.xs(city, level='city')
        ax.plot(city_series.index, city_series.values, marker='o', linewidth=2, label=city)
        
    ax.set_title("Хронологічна динаміка середньомісячної температури", fontsize=14, fontweight='bold')
    ax.set_xlabel("Час спостереження (Місяці)", fontsize=12)
    ax.set_ylabel("Температура (°C)", fontsize=12)
    ax.legend(title="Міста", frameon=True)
    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Графік 2: Bar chart (Сумарні опади) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    precip_data = analytics['by_city_precip']
    bars = ax.bar(precip_data.index, precip_data.values, color='skyblue', edgecolor='black', alpha=0.8)
    
    # Додаємо числові значення над кожним баром
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_title("Загальний об'єм опадів по містах за період спостережень", fontsize=14, fontweight='bold')
    ax.set_xlabel("Місто", fontsize=12)
    ax.set_ylabel("Сума опадів (мм)", fontsize=12)
    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Графік 3: Histogram (Розподіл температур) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df['temperature_c'].dropna(), bins=40, color='lightgreen', edgecolor='black', alpha=0.7)
    
    # Розрахунок ліній
    global_mean = df['temperature_c'].mean()
    global_median = df['temperature_c'].median()
    
    ax.axvline(global_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {global_mean:.2f}°C')
    ax.axvline(global_median, color='blue', linestyle='-.', linewidth=2, label=f'Median: {global_median:.2f}°C')
    
    ax.set_title("Гістограма розподілу температурних показників (усі станції)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Температура (°C)", fontsize=12)
    ax.set_ylabel("Частота (Кількість спостережень)", fontsize=12)
    ax.legend(frameon=True, loc='upper right')
    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Графік 4: Heatmap (Pivot table місто х місяць) ---
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot_data = analytics['pivot']
    
    # Малюємо матрицю
    cax = ax.imshow(pivot_data.values, cmap='coolwarm', aspect='auto')
    
    # Додаємо colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Середня температура (°C)', fontsize=11)
    
    # Встановлюємо підписи для осей
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index, fontsize=11)
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_xticklabels([f"Міс. {m}" for m in pivot_data.columns], fontsize=10)
    
    # Відобразимо точні значення всередині кожної клітинки для кращої читабельності
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            ax.text(j, i, f"{pivot_data.values[i, j]:.1f}",
                    ha="center", va="center", color="black" if 5 < pivot_data.values[i, j] < 20 else "white",
                    fontweight='semibold')

    ax.set_title("Теплокарта середніх температур за календарними місяцями", fontsize=14, fontweight='bold')
    ax.set_xlabel("Календарний місяць", fontsize=12)
    ax.set_ylabel("Місто", fontsize=12)
    ax.grid(False) # Вимикаємо сітку поверх матриці
    
    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Усі 4 графіки успішно згенеровано та збережено в папку: '{PLOTS_DIR}/'")


# ====================================================================

def main() -> None:
    # Етап 1: Завантаження
    df_raw = load_observations()
    print(f"Початкова розмірність матриці даних: {df_raw.shape}")

    # Етап 2: Аналіз NumPy
    block_1_numpy(df_raw)
    
    # Етап 3: Очищення Pandas
    df_clean = block_2_cleaning(df_raw)
    
    # Етап 4: Аналітика Pandas
    analytics = block_3_analytics(df_clean)
    
    # Етап 5: Візуалізація
    block_4_plots(df_clean, analytics)


if __name__ == "__main__":
    main()


"""
ВИСНОВКИ (Аналітичний кліматичний звіт):

На основі обробленого масиву метеоспостережень встановлено, що найтеплішим містом за період дослідження є Дніпро з середньою температурою 12.53°C, тоді як найхолоднішим виявився Київ із показником 8.11°C, що пояснюється підвищеним рівнем інсоляції південніших регіонів та специфікою локальних повітряних мас. Сезонність температурних показників чітко виражена амплітудними коливаннями помірно-континентального клімату з мінімумом у грудні-січні (до -4.2°C у Києві) та літніми максимумами у червні-липні. У ході аналізу девіацій було виявлено унікальну кліматичну аномалію в червні 2024 року (2024-06) у місті Дніпро, де зафіксовано екстремальне позитивне відхилення від хронологічної норми на +6.72°C, що беззаперечно свідчить про руйнівну та тривалу хвилю спеки. За метрикою стабільності температурного режиму (стандартне відхилення std) найбільш стійким виявився регіон Києва (std = 10.45), у той час як Львів демонструє найбільшу амплітуду коливань та нестабільність (std = 10.63). 

З огляду на отримані дані, для оптимізації логістичних процесів рекомендується проектувати та будувати капітальні склади-холодильники у прохолодніших та кліматично стабільніших регіонах (Київ або Львів). Водночас, для комунальних служб та бізнесу в Дніпрі та Одесі критично важливо закладати додаткові бюджети на модернізацію енергомереж через літні піки кондиціонування, а у Львові — інвестувати у розширення зливової інфраструктури, оскільки місто є абсолютним лідером за сумою опадів (719.4 мм) та кількістю штормових днів.
"""
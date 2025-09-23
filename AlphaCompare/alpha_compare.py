import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import argparse


def read_and_process_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=';')
        if 'time' not in df.columns or 'alpha' not in df.columns:
            raise ValueError(f"Файл {file_path} не содержит необходимых колонок 'time' и 'alpha'")

        # Преобразование времени в datetime
        df['datetime'] = pd.to_datetime(df['time'] / 1000, unit='ms')

        return df

    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return None


def plot_alpha_vs_time(file1_path, file2_path, output_path=None):
    """
    Построение графика alpha от времени для двух файлов
    """
    # Чтение данных
    df1 = read_and_process_csv(file1_path)
    df2 = read_and_process_csv(file2_path)

    if df1 is None or df2 is None:
        return

    # Создание графика
    plt.figure(figsize=(14, 7))

    # Построение графиков для обоих файлов
    plt.plot(df1['datetime'], df1['alpha'],
             label=f'Файл 1: {file1_path}', linewidth=1, marker='o', markersize=2)
    plt.plot(df2['datetime'], df2['alpha'],
             label=f'Файл 2: {file2_path}', linewidth=1, marker='s', markersize=2)

    # Настройка графика
    plt.title('Сравнение alpha от времени', fontsize=14)
    plt.xlabel('Время', fontsize=12)
    plt.ylabel('Alpha', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Форматирование оси времени - метки каждые 15 минут
    ax = plt.gca()

    # Установка формата времени
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))

    # Установка локатора - каждые 15 минут
    ax.xaxis.set_major_locator(dates.MinuteLocator(interval=15))

    # Автоматическое вращение меток
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Автоматическое масштабирование
    plt.tight_layout()

    # Сохранение или отображение графика
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в файл: {output_path}")

    plt.show()


def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Сравнение alpha от времени из двух CSV-файлов')
    parser.add_argument('file1', help='Путь к первому CSV-файлу')
    parser.add_argument('file2', help='Путь ко второму CSV-файлу')
    parser.add_argument('-o', '--output', help='Путь для сохранения графика (опционально)')

    args = parser.parse_args()

    # Построение графика
    plot_alpha_vs_time(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()
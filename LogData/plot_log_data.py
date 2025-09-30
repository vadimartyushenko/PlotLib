import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import re
from datetime import datetime
from matplotlib.lines import Line2D
import numpy as np
import argparse
import os


def parse_log_file(filename):
    """Парсинг файла лога и извлечение данных об ордерах и позициях"""

    # Регулярные выражения для извлечения данных
    order_pattern = r'(\w+)\s*:\s*size\s*([-\d.]+),\s*price\s*(\d+),\s*position\s*([-\d.]+)(?:,\s*alpha\s*([-\d.]+))?,\s*(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+)'

    orders_data = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Пропускаем строки с таблицей и другими служебными данными
            if any(keyword in line for keyword in ['Position close', 'Simulation results', '| position', '| --------']):
                continue

            match = re.search(order_pattern, line)
            if match:
                event_type, size, price, position, alpha, timestamp = match.groups()

                # Преобразуем timestamp в datetime объект
                dt = datetime.strptime(timestamp, '%d-%m-%Y %H:%M:%S.%f')

                # Преобразуем числовые значения
                size = float(size)
                position = float(position)
                price = float(price)
                alpha = float(alpha) if alpha else None

                orders_data.append({
                    'timestamp': dt,
                    'event_type': event_type,
                    'size': size,
                    'price': price,
                    'position': position,
                    'alpha': alpha,
                    'filename': os.path.basename(filename)
                })

    return pd.DataFrame(orders_data)


def plot_positions_separate_figure(df_list, filenames):
    """Построение графика позиций на отдельной фигуре"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(df_list)))

    for i, (df, filename) in enumerate(zip(df_list, filenames)):
        linestyle = '--' if i == 0 else '-'

        # Используем ступенчатый график для позиции
        times = df['timestamp'].values
        positions = df['position'].values

        # Строим ступенчатый график позиции
        ax.step(times, positions, where='post', linewidth=1.0,
                color=colors[i], label=filename, alpha=0.8, linestyle=linestyle)

        # Отмечаем точки исполнения ордеров
        fill_events = df[df['event_type'] == 'REPORT_FILL']
        if not fill_events.empty:
            # Определяем маркеры и цвета в зависимости от файла и направления сделки
            point_colors = []
            point_markers = []
            point_sizes = []

            for _, event in fill_events.iterrows():
                if i == 0:  # Первый файл
                    if event['size'] > 0:  # Покупка - синий треугольник вверх
                        point_colors.append('blue')
                        point_markers.append('^')
                    else:  # Продажа - красный треугольник вниз
                        point_colors.append('blue')
                        point_markers.append('v')
                else:  # Второй файл
                    if event['size'] > 0:  # Покупка - синий круг
                        point_colors.append('red')
                        point_markers.append('^')
                    else:  # Продажа - красный квадрат
                        point_colors.append('red')
                        point_markers.append('v')

                # Размер точки пропорционален объему сделки
                point_sizes.append(abs(event['size']) * 1)

            # Рисуем точки для каждого маркера отдельно
            for marker in set(point_markers):
                mask = [m == marker for m in point_markers]
                masked_events = fill_events.iloc[mask]
                masked_colors = [point_colors[j] for j in range(len(point_colors)) if mask[j]]
                masked_sizes = [point_sizes[j] for j in range(len(point_sizes)) if mask[j]]

                if len(masked_events) > 0:
                    ax.scatter(masked_events['timestamp'], masked_events['position'],
                               c=masked_colors, s=masked_sizes, marker=marker,
                               zorder=5, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_ylabel('Позиция', fontsize=12, fontweight='bold')
    ax.set_xlabel('Время', fontsize=12, fontweight='bold')
    ax.set_title('Изменение позиции', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Форматирование оси времени
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Добавляем горизонтальную линию на нулевой позиции
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    legend_elements = []
    # Добавляем элементы для линий файлов
    for i, filename in enumerate(filenames):
        line_style = '--' if i == 0 else '-'
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=2,
                                      linestyle=line_style, label=filename))

    legend_elements.extend([
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
               markersize=8, label=f'Покупки ({filenames[0]})'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue',
               markersize=8, label=f'Продажи ({filenames[0]})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markersize=8, label=f'Покупки ({filenames[1]})'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=8, label=f'Продажи ({filenames[1]})'),
    ])

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig


def plot_orders_activity_separate_figure(df_list, filenames):
    """Построение графика активности ордеров на отдельной фигуре"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(df_list)))

    # График 1: Размеры ордеров во времени
    for i, (df, filename) in enumerate(zip(df_list, filenames)):
        # Берем только новые ордера
        order_df = df[df['event_type'] == 'ORDER_NEW'].copy()

        if not order_df.empty:
            # Разделяем покупки и продажи
            buy_orders = order_df[order_df['size'] > 0]
            sell_orders = order_df[order_df['size'] < 0]

            # Покупки - синие столбцы
            if not buy_orders.empty:
                ax1.bar(buy_orders['timestamp'], buy_orders['size'],
                        width=0.0005, color='blue', alpha=0.7,
                        label=f'{filename} - Покупки' if i == 0 else "")

            # Продажи - красные столбцы
            if not sell_orders.empty:
                ax1.bar(sell_orders['timestamp'], sell_orders['size'],
                        width=0.0005, color='red', alpha=0.7,
                        label=f'{filename} - Продажи' if i == 0 else "")

    ax1.set_ylabel('Размер ордера', fontsize=12, fontweight='bold')
    ax1.set_title('Размеры выставленных ордеров по времени', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Форматирование оси времени
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # График 2: Активность ордеров (точечный график)
    for i, (df, filename) in enumerate(zip(df_list, filenames)):
        # Берем только новые ордера
        order_df = df[df['event_type'] == 'ORDER_NEW'].copy()

        if not order_df.empty:
            # Разделяем покупки и продажи
            buy_orders = order_df[order_df['size'] > 0]
            sell_orders = order_df[order_df['size'] < 0]

            # Покупки - синие точки
            if not buy_orders.empty:
                ax2.scatter(buy_orders['timestamp'], [i] * len(buy_orders),
                            s=np.abs(buy_orders['size']) * 30, color='blue',
                            alpha=0.7, label='Покупки' if i == 0 else "")

            # Продажи - красные точки
            if not sell_orders.empty:
                ax2.scatter(sell_orders['timestamp'], [i] * len(sell_orders),
                            s=np.abs(sell_orders['size']) * 30, color='red',
                            alpha=0.7, label='Продажи' if i == 0 else "")

    ax2.set_ylabel('Файл', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Время', fontsize=12, fontweight='bold')
    ax2.set_title('Активность ордеров по файлам', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(filenames)))
    ax2.set_yticklabels(filenames)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Форматирование оси времени
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    return fig


def plot_orders_by_file_separate_figures(df_list, filenames):
    """Построение отдельных графиков ордеров для каждого файла на отдельной фигуре"""

    n_files = len(df_list)
    fig, axes = plt.subplots(n_files, 1, figsize=(14, 4 * n_files))

    if n_files == 1:
        axes = [axes]

    for i, (df, filename) in enumerate(zip(df_list, filenames)):
        ax = axes[i]

        # Берем только новые ордера
        order_df = df[df['event_type'] == 'ORDER_NEW'].copy()

        if not order_df.empty:
            # Разделяем покупки и продажи
            buy_orders = order_df[order_df['size'] > 0]
            sell_orders = order_df[order_df['size'] < 0]

            # Покупки - синие столбцы
            if not buy_orders.empty:
                ax.bar(buy_orders['timestamp'], buy_orders['size'],
                       width=0.0005, color='blue', alpha=0.7, label='Покупки')

            # Продажи - красные столбцы
            if not sell_orders.empty:
                ax.bar(sell_orders['timestamp'], sell_orders['size'],
                       width=0.0005, color='red', alpha=0.7, label='Продажи')

        ax.set_ylabel('Размер ордера', fontsize=10, fontweight='bold')
        ax.set_title(f'Активность ордеров - {filename}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Форматирование оси времени
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Добавляем линию на нуле
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    return fig


def create_detailed_summary(df_list, filenames):
    """Создание детальной сводной таблицы"""

    print("=" * 100)
    print("ДЕТАЛЬНАЯ СТАТИСТИКА ПО ВСЕМ ФАЙЛАМ")
    print("=" * 100)

    summary_data = []

    for df, filename in zip(df_list, filenames):
        # Статистика по ордерам
        total_orders = len(df[df['event_type'] == 'ORDER_NEW'])
        filled_orders = len(df[df['event_type'] == 'REPORT_FILL'])
        canceled_orders = len(df[df['event_type'] == 'REPORT_CANCEL'])

        # Статистика по позиции
        max_position = df['position'].max()
        min_position = df['position'].min()
        final_position = df['position'].iloc[-1]

        # Статистика по объемам
        total_volume = df[df['event_type'] == 'REPORT_FILL']['size'].abs().sum()
        buy_volume = df[(df['event_type'] == 'REPORT_FILL') & (df['size'] > 0)]['size'].sum()
        sell_volume = abs(df[(df['event_type'] == 'REPORT_FILL') & (df['size'] < 0)]['size'].sum())

        # Количество сделок по направлениям
        buy_trades = len(df[(df['event_type'] == 'REPORT_FILL') & (df['size'] > 0)])
        sell_trades = len(df[(df['event_type'] == 'REPORT_FILL') & (df['size'] < 0)])

        summary_data.append({
            'Файл': filename,
            'Ордеров': total_orders,
            'Исполнено': filled_orders,
            'Отменено': canceled_orders,
            'Макс позиция': f"{max_position:+.1f}",
            'Мин позиция': f"{min_position:+.1f}",
            'Фин позиция': f"{final_position:+.1f}",
            'Объем': f"{total_volume:.1f}",
            'Покупки': f"{buy_volume:.1f}",
            'Продажи': f"{sell_volume:.1f}",
            'Сделки покупки': buy_trades,
            'Сделки продажи': sell_trades
        })

    # Выводим таблицу
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 100)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Анализ торговых логов')
    parser.add_argument('files', nargs='+', help='Файлы логов для анализа')
    parser.add_argument('--separate-file-orders', action='store_true',
                        help='Построить отдельные графики ордеров для каждого файла')
    parser.add_argument('--output', default='separate_analysis',
                        help='Базовое имя для выходных файлов')

    args = parser.parse_args()

    # Проверяем существование файлов
    valid_files = []
    for file in args.files:
        if os.path.exists(file):
            valid_files.append(file)
        else:
            print(f"Предупреждение: файл {file} не найден и будет пропущен")

    if not valid_files:
        print("Не найдено ни одного валидного файла для анализа")
        return

    try:
        # Парсим все файлы
        df_list = []
        filenames = []

        for file in valid_files:
            df = parse_log_file(file)
            if not df.empty:
                df_list.append(df)
                filenames.append(os.path.basename(file))
                print(f"Обработан файл: {file} ({len(df)} записей)")
            else:
                print(f"Предупреждение: файл {file} не содержит данных")

        if not df_list:
            print("Не удалось извлечь данные из файлов логов.")
            return

        # Строим график позиций на отдельной фигуре
        print("Строим график позиций...")
        fig_positions = plot_positions_separate_figure(df_list, filenames)
        output_file_positions = f"{args.output}_positions.png"
        plt.savefig(output_file_positions, dpi=300, bbox_inches='tight')
        print(f"График позиций сохранен как '{output_file_positions}'")
        plt.show(block=False)  # Показываем без блокировки

        # Строим график активности ордеров на отдельной фигуре
        print("Строим график активности ордеров...")
        fig_orders = plot_orders_activity_separate_figure(df_list, filenames)
        output_file_orders = f"{args.output}_orders.png"
        plt.savefig(output_file_orders, dpi=300, bbox_inches='tight')
        print(f"График ордеров сохранен как '{output_file_orders}'")
        plt.show(block=False)

        # Дополнительно: отдельные графики ордеров для каждого файла
        if args.separate_file_orders:
            print("Строим отдельные графики ордеров по файлам...")
            fig_individual_orders = plot_orders_by_file_separate_figures(df_list, filenames)
            output_file_individual = f"{args.output}_individual_orders.png"
            plt.savefig(output_file_individual, dpi=300, bbox_inches='tight')
            print(f"Индивидуальные графики ордеров сохранены как '{output_file_individual}'")
            plt.show(block=False)

        # Выводим статистику
        summary_df = create_detailed_summary(df_list, filenames)

        # Сохраняем статистику в CSV
        csv_file = f"{args.output}_summary.csv"
        summary_df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Статистика сохранена как '{csv_file}'")

        # Ждем закрытия всех окон
        print("\nВсе графики построены. Закройте окна графиков для завершения.")
        plt.show()  # Блокируем до закрытия всех окон

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
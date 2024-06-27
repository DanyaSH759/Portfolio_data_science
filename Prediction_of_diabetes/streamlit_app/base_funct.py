
def transform_df_base():
    """Функция делит базовый датасет на выборки и заполняет пропуский"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import numpy as np

    # грузим данные
    data = pd.read_excel('./diabetes_ds.xlsx', index_col=0)

    # удалим ошибочно отображенные строчки при загрузке датасте
    data = data.iloc[:-2]

    # Удалим сразу строчку с пустым таргетом
    data.dropna(subset=['Сахарный диабет (ЕСТЬ\НЕТ)'], inplace=True)

    # Отделим наши тренировочные данные от теста. Тест отложим до финального тестирования
    data, test_not_transform = train_test_split(data, test_size = 0.30, shuffle = True, random_state = 654321)

    # колонки с черезмерными дизбалансом классов.
    # в основном это лекарственные препараты
    data = data.drop(
        ['Цветовой показатель', 'Молнупиравир', 'Гидрохлортиазид',
        'Валсартан', 'Телмисартан', 'Кандесартан',
        'Изосорбида динитрат', 'Периндоприл', 'Рамиприл',
        'Лизиноприл', 'Карведилол', 'Атенолол',
        'Верапамил', 'Дилтиазем', 'Леркамен',
        'Ивабрадин', 'Триметазидин', 'ВПС' , 'НЕАЖБП',
        'Удлинение интервала QT', 'Аблация', 'Метформин', 'Дапаглифлозин', 'Триметазидин', 'Дигоксин', 'Амлодипин', 'Бисопролол', 'Эналаприл', 'Рамиприл',
        'Периндоприл', 'Изосорбида динитрат', 'Лозартан', 'Спиронолактон', 'Торасемид', 'Фуросемид', 'Фавипиравир', 'Гликированный гемоглобин', ], axis = 1)


    # удаление излишних информационных колонок
    data = data.drop(['Блокада НП Гисса (комментарий)', 
                                'Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ (комментарий)',], axis = 1)

    # удаление колонок, которые дублируют информацию
    data = data.drop(['БА (ЕСТЬ/НЕТ)', 'ХОБЛ (ЕСТЬ\НЕТ)', 'ИБС. Стенокардия (ЕСТЬ\НЕТ)', 'ИБС. ПИКС (ЕСТЬ\НЕТ)',
                                'АГ (ЕСТЬ\НЕТ)', 'ХБП (ЕСТЬ\НЕТ)', 'Мерцательная аритмия (ЕСТЬ\НЕТ)',
                                'ОЖИРЕНИЕ (ЕСТЬ\НЕТ)', 'BPB голеней (ЕСТЬ/НЕТ)', 'Антикоагулянты, препарат 2', 'Антиагреганты, препарат 2'], axis = 1)

    # удаление колонок, пропуски в которых трудно восстановить
    data = data.drop(['СРБ', 'Ферритин', 'Прокальцитонин', 'Альбумин',
                                'Лактат', 'АЧТВ' ,'ЛДГ',
                                'МНО', 'Фибриноген', 'D-димер'], axis = 1)

    # удаление неинформативных колонок
    data = data.drop(['АГ, стадия', 'АГ, риск', 'ХСН, ФК', 'ХБП (расчет по СКФ)',
                                'Рост, см', 'Вес, кг'], axis = 1)
    
    # заполним пропуски значением 0 т.к. это терапия, и если нет препаратов значит они не назначались
    data['Антикоагулянты, препарат'] = data['Антикоагулянты, препарат'].fillna(0)
    data['Антиагреганты, препарат 1'] = data['Антиагреганты, препарат 1'].fillna(0)

    # заменим значения, которые были записаны не верно
    data['BPB голеней'] = data['BPB голеней'].replace({'2-3': 2})
    data['Блокада НП Гисса'] = data['Блокада НП Гисса'].replace({'1-2': 2})
    data['AV- блокада'] = data['AV- блокада'].replace({'полная': 3, '1-2': 1, '2-3': 3})
    data['ХБП (исходн)'] = data['ХБП (исходн)'].replace({'с3а-с3б': 'c3a', 'с2-с3': 'c2', '2са1': 'c2',
                                                                    'с1 а3': 'c1'})
    data['Перенес/ не перенес КВИ'] = data['Перенес/ не перенес КВИ'].replace({1: 0, 2:1})

    # произведем сокращение классов и кодирование в данной колонке
    data['ХБП (исходн)'] = data['ХБП (исходн)'].replace({'с3а-с3б': 'c3a', 'с2-с3': 'c2', '2са1': 'c2',
                                                                    'с1 а3': 'c1'})
    data['ХБП (исходн)'] = data['ХБП (исходн)'].replace({'с1': 1, 'c1': 1, 'c2': 2, 'с2':2, '2а': 3,  'с3': 4, 'с3а': 5, 'c3a': 5, 'с3a': 5,'с3б': 6, 'с4': 7})

    data['Сахарный диабет (ЕСТЬ\НЕТ)'] = data['Сахарный диабет (ЕСТЬ\НЕТ)'].astype('int')
    data['Блокада НП Гисса'] = data['Блокада НП Гисса'].astype('int')
    data['AV- блокада'] = data['AV- блокада'].astype('int')
    data['ХБП (исходн)'] = data['ХБП (исходн)'].astype('int')
    data['Перенес/ не перенес КВИ'] = data['Перенес/ не перенес КВИ'].astype('int')

    def clean_value(val):
        '''Функция для замены значений из колонки из числового 
        (по типу "35-45") на 40'''
        try:
            return float(val)
        except ValueError:
            pass
        try:
            parts = val.split('-')
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            pass
        return 0

    # Применение функции clean_value к каждому элементу
    data['ХСН по фракции выброса, %'] = np.array([clean_value(val) for val in data['ХСН по фракции выброса, %']])
    data['ХСН по фракции выброса, %'] = data['ХСН по фракции выброса, %'].astype('float')

    data.dropna(subset=['Возраст', 'Эритроциты', 'Лейкоциты',
                         'Лимфоциты% ', 'ЛИМФОЦИТЫ, АБС (РАСЧЕТ ПО ФОРМУЛЕ)',
                         'Тромбоциты', 'Креатинин, мкмоль/л',
                         'СКФ CKD-EPI (расчет по общей формуле)',
                         'СТЕПЕНИ ОЖИРЕНИЯ ПО ИМТ', 'BPB голеней',
                         'Тромбоэмболический синдром',
                         'Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ',
                         'Мерцательная аритмия', 'ХСН по фракции выброса, %'], inplace=True)

    # Пропуски до 20% заполним медианными значениями
    to_median = ['СОЭ', 'Холестерин', 'ЛПНП', 'ЛПВП',
                 'Индекс атерогенности 2 (формула excel)', 'Глюкоза',
                 'АЛТ', 'АСТ', 'Натрийуретический пептид', ]

    for i in to_median:
        data.loc[data[i].isna(), i] = data.loc[data[i].notna(), i].median()

    # остальные пропуски закроем заглушками
    to_except = ['Натрий', 'Калий', 'Мочевина', 'Общий билирубин']
    for i in to_except:
        data.loc[data[i].isna(), i] = 0

    data = data.drop(
        ['Возраст', 'Пол', 'Лимфоциты% ', 'АСТ', 'АЛТ',
         'Натрийуретический пептид', 'СКФ CKD-EPI (расчет по общей формуле)',
         'Антиагреганты, препарат 1', 'BPB голеней',
         'ИБС. ПИКС', 'ХСН по фракции выброса, %',
         'Антикоагулянты, препарат', 'Гиполипидемические препараты (статины)' ], axis = 1)
    
    return data, test_not_transform


def base_graph(data, column):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    """Функция для отрисовки describe, boxplot, heatmap и hist"""

    # создание 1 линии графиков
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        specs=[[{"type": "table"}, {"type": "Box"}]],
        subplot_titles=(f"Стат. информация по колонке {column}", f"Boxplot {column}")
    )

    # переворачиваем и упаковываем данные с describe
    header_values = list(data[column].describe().index)
    cell_values = [[header_values[i], data[column].describe().values[i]] for i in range(len(data[column].describe()))]
    cell_values_transposed = list(map(list, zip(*cell_values)))

    # добавляем таблицу 
    fig.add_trace(
        go.Table(
            header=dict(
                values = ['index', 'values'],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=cell_values_transposed,
                align="left")
        ),
        row=1, col=1
    )

    # добавляем boxplot
    fig.add_trace(
        go.Box(
            y=data[column]
        ),
        row=1, col=2
    )

    # обновляет текст над графиками
    fig.update_layout(
        height=400, width=1200,
        showlegend=False,
        title_text=f"Информация по колонке {column}",
    )

    # выводим 1 ряд
    st.plotly_chart(fig)

    # Создаем тепловую карту с Plotly Express
    fig1 = px.density_heatmap(data, x=column, y="Сахарный диабет (ЕСТЬ\НЕТ)", text_auto=True)

    # Создаем гистограмму с Plotly Graph Objects
    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x = data[column],
            name="Перенес/ не перенес КВИ", nbinsx= 10
        )
    )
    
    fig2.update_layout(title="Пример столбчатой диаграммы",
                    xaxis_title="Перенес/ не перенес КВИ",
                    yaxis_title="Количество")

    # Создаем сетку с двумя графиками в одной строке
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Тепловая карта {column}", f"Гистограмма {column}"))

    # Добавляем тепловую карту на первую позицию в сетке
    fig.add_trace(fig1['data'][0], row=1, col=1)

    # Добавляем гистограмму на вторую позицию в сетке
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)

    # Обновляем общий заголовок и макет сетки
    fig.update_layout(title_text="",
                    xaxis_title=column,
                    yaxis_title="Сахарный диабет (ЕСТЬ\НЕТ)",
                    height=400, width=1200, showlegend=False)

    # выводим 2 ряд
    st.plotly_chart(fig)


class DataTransformer:
    """Класс предназначен для фиксации изменений из тренировочного
    датаста и применение их на тестевом"""

    def __init__(self):

        '''Инициализация параметров
        self.median_values - сохранение медиан каждой колонки
        self.columns_to_exclude - список колонок где используются загулушки в пропусках
        self.columns_in_dataset - колонки, которые должны остаться в датасете
        '''

        self.median_values = None
        self.columns_to_exclude = []
        self.columns_in_dataset = []

    def fit(self, df, columns_to_exclude=[]):
        '''Сохранение параметров из тренировочного датасета'''
        # колонкли для заполнения заглушками пустых значений
        self.columns_to_exclude = columns_to_exclude
        # Сохраняем медианные значения только для колонок, не включённых в исключения
        self.median_values = df.drop(columns=columns_to_exclude).median()
        self.columns_in_dataset = df.columns

    def transform(self, df):
        '''Трансформация тестового датасета по аналогии с тренировочным'''

        if self.median_values is None:
            raise ValueError("The fit method must be called before transform.")
        
        # Трансформируем только те колонки, которые не в списке исключений
        for col in df.columns:
            if col not in self.columns_to_exclude and col in self.median_values.index:
                df[col].fillna(self.median_values[col], inplace=True)
            else:
                df[col].fillna(0, inplace=True)
        
        # сохранение только нужных колонок
        df = df[self.columns_in_dataset]
        
        # трансформация некорректных значений
        df['Блокада НП Гисса'] = df['Блокада НП Гисса'].replace({'1-2': 2})
        df['AV- блокада'] = df['AV- блокада'].replace({'полная': 3, '1-2': 1, '2-3': 3})
        df['ХБП (исходн)'] = df['ХБП (исходн)'].replace({'с3а-с3б': 'c3a', 'с2-с3': 'c2', '2са1': 'c2',
                                                                        'с1 а3': 'c1'})
        df['Перенес/ не перенес КВИ'] = df['Перенес/ не перенес КВИ'].replace({1: 0, 2:1})

        # произведем сокращение классов и кодирование в данной колонке
        df['ХБП (исходн)'] = df['ХБП (исходн)'].replace({'с3а-с3б': 'c3a', 'с2-с3': 'c2', '2са1': 'c2',
                                                                        'с1 а3': 'c1'})
        df['ХБП (исходн)'] = df['ХБП (исходн)'].replace({'с1': 1, 'c1': 1, 'c2': 2, 'с2':2, '2а': 3,  'с3': 4, 'с3а': 5, 'c3a': 5, 'с3a': 5,'с3б': 6, 'с4': 7})

        df['Блокада НП Гисса'] = df['Блокада НП Гисса'].astype('int')
        df['AV- блокада'] = df['AV- блокада'].astype('int')
        df['ХБП (исходн)'] = df['ХБП (исходн)'].astype('int')
        df['Перенес/ не перенес КВИ'] = df['Перенес/ не перенес КВИ'].astype('int')
        df['Сахарный диабет (ЕСТЬ\НЕТ)'] = df['Сахарный диабет (ЕСТЬ\НЕТ)'].astype('int')
        df['АКШ'] = df['АКШ'].astype('int')
        df['Тромбоэмболический синдром'] = df['Тромбоэмболический синдром'].astype('int')
        df['Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ'] = df['Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ'].astype('int')

        return df
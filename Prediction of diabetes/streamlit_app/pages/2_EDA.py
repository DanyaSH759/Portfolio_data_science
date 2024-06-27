import streamlit as st
from set_params import load_page_setting
from io import StringIO
from base_funct import transform_df_base, base_graph
from PIL import Image

st.set_page_config(
    page_title="EDA",
    page_icon="📊"
)

# загрузка параметров фона стилей и т.д.
load_page_setting()

st.sidebar.header("EDA")
st.sidebar.success("Заполнение пропусков")
st.sidebar.success("Исследование данных")

st.markdown("""
    # Исследовательский анализ данных
    ## Решение проблемы пропусков

    Решено было удалить различные колонки, в которых есть утечка таргета или они не информативны. В основном это лекартсвенные препараты."""
)
with st.expander("Удаление столбцов"):
    st.code("""
    data_eda_1 = data_eda_1.drop(['Возраст', 'Пол', 'Лимфоциты% ', 'АСТ', 'АЛТ', 'Натрийуретический пептид', 'СКФ CKD-EPI (расчет по общей формуле)',
                                'Антиагреганты, препарат 1', 'BPB голеней',
                                'ИБС. ПИКС', 'ХСН по фракции выброса, %', 'Антикоагулянты, препарат', 'Гиполипидемические препараты (статины)' ], axis = 1)
        
    # удаление излишних информационных колонок
    data_eda_1 = data_eda_1.drop(['Блокада НП Гисса (комментарий)', 
                                'Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ (комментарий)',], axis = 1)

    # удаление колонок, которые дублируют информацию
    data_eda_1 = data_eda_1.drop(['БА (ЕСТЬ/НЕТ)', 'ХОБЛ (ЕСТЬ\НЕТ)', 'ИБС. Стенокардия (ЕСТЬ\НЕТ)', 'ИБС. ПИКС (ЕСТЬ\НЕТ)',
                                'АГ (ЕСТЬ\НЕТ)', 'ХБП (ЕСТЬ\НЕТ)', 'Мерцательная аритмия (ЕСТЬ\НЕТ)',
                                'ОЖИРЕНИЕ (ЕСТЬ\НЕТ)', 'BPB голеней (ЕСТЬ/НЕТ)'], axis = 1)

    # удаление колонок, пропуски в которых трудно восстановить
    data_eda_1 = data_eda_1.drop(['СРБ', 'Ферритин', 'Прокальцитонин', 'Альбумин',
                                'Лактат', 'АЧТВ' ,'ЛДГ',
                                'МНО', 'Фибриноген', 'D-димер'], axis = 1)

    # удаление неинформативных колонок
    data_eda_1 = data_eda_1.drop(['АГ, стадия', 'АГ, риск', 'ХСН, ФК', 'ХБП (расчет по СКФ)',
                                'Рост, см', 'Вес, кг'], axis = 1)
    # 
    data_eda_1 = data_eda_1.drop(['Цветовой показатель', 'Молнупиравир', 'Гидрохлортиазид',
                                'Валсартан', 'Телмисартан', 'Кандесартан',
                                'Изосорбида динитрат', 'Периндоприл', 'Рамиприл',
                                'Лизиноприл', 'Карведилол', 'Атенолол',
                                'Верапамил', 'Дилтиазем', 'Леркамен',
                                'Ивабрадин', 'Триметазидин', 'ВПС' , 'НЕАЖБП',
                                'Удлинение интервала QT', 'Аблация', 'Метформин', 'Дапаглифлозин', 'Триметазидин', 'Дигоксин', 'Амлодипин', 'Бисопролол', 'Эналаприл', 'Рамиприл',
                                'Периндоприл', 'Изосорбида динитрат', 'Лозартан', 'Спиронолактон', 'Торасемид', 'Фуросемид', 'Фавипиравир', 'Гликированный гемоглобин', 
                                ], axis = 1) """,
                                line_numbers = True)

st.markdown(
    """
    Исправили ошибочные записи в колонках, в некоторых провели трансформацию на числовые категории.
    """
)
with st.expander("Исправление некорректных данных"):
    st.code("""
        # заменим значения, которые были записаны не верно
        data_eda_1['BPB голеней'] = data_eda_1['BPB голеней'].replace({'2-3': 2})
        data_eda_1['Блокада НП Гисса'] = data_eda_1['Блокада НП Гисса'].replace({'1-2': 2})
        data_eda_1['AV- блокада'] = data_eda_1['AV- блокада'].replace({'полная': 3, '1-2': 1, '2-3': 3})
        data_eda_1['ХБП (исходн)'] = data_eda_1['ХБП (исходн)'].replace({'с3а-с3б': 'c3a', 'с2-с3': 'c2', '2са1': 'c2',
                                                                        'с1 а3': 'c1'})
        data_eda_1['Перенес/ не перенес КВИ'] = data_eda_1['Перенес/ не перенес КВИ'].replace({1: 0, 2:1})


        # произведем сокращение классов и кодирование в данной колонке
        data_eda_1['ХБП (исходн)'] = data_eda_1['ХБП (исходн)'].replace({'с3а-с3б': 'c3a', 'с2-с3': 'c2', '2са1': 'c2',
                                                                        'с1 а3': 'c1'})
        data_eda_1['ХБП (исходн)'] = data_eda_1['ХБП (исходн)'].replace({'с1': 1, 'c1': 1, 'c2': 2, 'с2':2, '2а': 3,  'с3': 4, 'с3а': 5, 'c3a': 5, 'с3a': 5,'с3б': 6, 'с4': 7})

            
        data_eda_1['Сахарный диабет (ЕСТЬ\НЕТ)'] = data_eda_1['Сахарный диабет (ЕСТЬ\НЕТ)'].astype('int')
        data_eda_1['Блокада НП Гисса'] = data_eda_1['Блокада НП Гисса'].astype('int')
        data_eda_1['AV- блокада'] = data_eda_1['AV- блокада'].astype('int')
        data_eda_1['ХБП (исходн)'] = data_eda_1['ХБП (исходн)'].astype('int')
        data_eda_1['Перенес/ не перенес КВИ'] = data_eda_1['Перенес/ не перенес КВИ'].astype('int'))

            
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
        data_eda_1['ХСН по фракции выброса, %'] = np.array([clean_value(val) for val in data_eda_1['ХСН по фракции выброса, %']])
        data_eda_1['ХСН по фракции выброса, %'] = data_eda_1['ХСН по фракции выброса, %'].astype('float')
            

        # скорректируем после заполнения пропусков тип данных
        data_eda_1v1['Тромбоэмболический синдром'] = data_eda_1v1['Тромбоэмболический синдром'].astype('int')
        data_eda_1v1['Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ'] = data_eda_1v1['Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ'].astype('int')
        """,

        line_numbers = True) 

st.markdown(
    """Далее заполнили пропуски. Удалили все пропуски <= 5%. Колонки с пропусками больше 20% заполнили значениями-заглушками, если пропусков меньше - заполнили медианным значением."""
)

with st.expander("Заполнение прпоусков"):
    st.code("""
    # удаляем пропуски меньше 5%
    data_eda_1v1.dropna(subset=['Возраст', 'Эритроциты', 'Лейкоциты', 'Лимфоциты% ', 'ЛИМФОЦИТЫ, АБС (РАСЧЕТ ПО ФОРМУЛЕ)',
    'Тромбоциты', 'Креатинин, мкмоль/л', 'СКФ CKD-EPI (расчет по общей формуле)',
    'СТЕПЕНИ ОЖИРЕНИЯ ПО ИМТ', 'BPB голеней', 'Тромбоэмболический синдром',
    'Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ', 'Мерцательная аритмия', 'ХСН по фракции выброса, %'], inplace=True)
                

    # Пропуски до 20% заполним медианными значениями
    to_median = ['СОЭ', 'Холестерин', 'ЛПНП', 'ЛПВП', 'Индекс атерогенности 2 (формула excel)', 'Глюкоза', 'АЛТ', 'АСТ', 'Натрийуретический пептид', ]
    for i in to_median:
        data_eda_1v1.loc[data_eda_1v1[i].isna(), i] = data_eda_1v1.loc[data_eda_1v1[i].notna(), i].median()

    # остальные пропуски закроем заглушками
    to_except = ['Натрий', 'Калий', 'Мочевина', 'Общий билирубин', 'Гликированный гемоглобин']
    for i in to_except:
        data_eda_1v1.loc[data_eda_1v1[i].isna(), i] = 0
    """, line_numbers = True)


st.markdown(
    """
    После всех манипуляций мы получили в итоге следующий датасет:
    """ 
)
df, _ = transform_df_base()

# Создаем буфер для записи вывода df.info()
buffer = StringIO()
df.info(buf=buffer, verbose = True, show_counts = True)

# Преобразуем буфер в строку и выводим с помощью st.text()
info_str = buffer.getvalue()

with st.expander("Просмотр первичного датасета"):
    st.text(info_str)
    st.dataframe(df.head())
    st.dataframe(df.tail())

st.markdown("""
    ## Переход к EDA

    Для начала мы оценим корреляцию Пирсона по каждой колонке к нашему таргету.
    """
)

img = Image.open('./streamlit_app/data/output.png')
st.image(img)

st.markdown(
    """
    Датасет был значительно сокращен. От изначальных 107 колонок осталось 32, что значительно упрощает определение самых важных признаков.
    По матрице корреляции видно, что в датасете ещё остались колонки, слабо коррелирующие с таргетом но на данный момент решено их оставить, т.к. после трансформации данных они могу дать ужее другой результат корреляции.

    Построим подробные графики для наших колонок
    """ 
)

column = st.selectbox('Выберите колонку для анализа', df.columns)
base_graph(df, column)


st.markdown(
"""
    ### Выводы

    Обнаружено большое кол-во аномальных значений, которые сильно выделяются на фоне остальных.
    На текущий момент, т.к. мы работаем с медицинскими данными, подобные сильные отклонения от
    нормы и медианных значений говорит о том что у пациента есть подозрения на те или иные заболевания.
    Поэтому если удалить подобные значения есть вероятность сокртатить качество модели из-за удаления
    подобных маркеров, которые могут указывать на возможные проблемы со здоровьем у пациента.
    """
)

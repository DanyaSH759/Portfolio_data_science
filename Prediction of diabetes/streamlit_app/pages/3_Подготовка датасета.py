import streamlit as st
from set_params import load_page_setting

st.set_page_config(
    page_title="Подготовка датасета",
    page_icon="🔧"
)

# загрузка параметров фона стилей и т.д.
load_page_setting()


st.sidebar.header("Подготовка датасета")
st.sidebar.success("Формирования класса для трансформации данных")
st.sidebar.success("Pipeline для кодирования/скалирования данных")

st.markdown("""
# Подготовка датасета

Для удобнйо работы с тестовыйм датасетом (и подобными датасетами в дальнейшем) решено было сформировать класс для подгона нового датасета под определенный формат.
Класс получал данные из тренировочного датастета, и применял все изменения проведенные на тренировочном датастете к полученному.

""")

with st.expander("Код класса для трансформации данных"):
    st.code("""
class DataTransformer:
    '''Класс предназначен для фиксации изменений из тренировочного
    датаста и применение их на тестевом'''

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

        return df """, line_numbers = True)


st.markdown(
"""
Был подготовлен pipeline для трансформации данных, для более удобной работы с датасетом. Так же от тренеровочной выборки отделена валидационная, для перебора для поиска лучшей модели.
"""
)

with st.expander("Подготовка выборок и организация pipeline"):
    st.code("""

# разделим датасет на фичи и таргет
x_train = data_base.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_train = data_base['Сахарный диабет (ЕСТЬ\НЕТ)']
            
# Трансформер для категориальных признаков
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Трансформер для числовых признаков
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Объединение трансформеров в единый ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, column_to_scaler),
        ('cat', categorical_transformer, column_to_OHE)
    ])
            
# Создание полного pipeline
pipeline_transform = Pipeline(steps=[('preprocessor', preprocessor)])

# Обучение pipeline на тренировочном наборе
pipeline_transform.fit(x_train) """, line_numbers = True)

st.markdown(
"""
По итогу трансформации на выходе у нас получается следующей датасет, котоырй мы планируем использовать для обучения моделей.
"""
)


with st.expander("Итоговый датасет"):
    st.code("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 315 entries, 0 to 314
Data columns (total 40 columns):
 #   Column                                                     Non-Null Count  Dtype  
---  ------                                                     --------------  -----  
 0   ИМТ 2 (ФОРМУЛА EXCEL)                                      315 non-null    float64
 1   Эритроциты                                                 315 non-null    float64
 2   Гемоглобин                                                 315 non-null    float64
 3   Лейкоциты                                                  315 non-null    float64
 4   ЛИМФОЦИТЫ, АБС (РАСЧЕТ ПО ФОРМУЛЕ)                         315 non-null    float64
 5   Тромбоциты                                                 315 non-null    float64
 6   СОЭ                                                        315 non-null    float64
 7   Холестерин                                                 315 non-null    float64
 8   ЛПНП                                                       315 non-null    float64
 9   ЛПВП                                                       315 non-null    float64
 10  Индекс атерогенности 2 (формула excel)                     315 non-null    float64
 11  Глюкоза                                                    315 non-null    float64
 12  Общий билирубин                                            315 non-null    float64
 13  Мочевина                                                   315 non-null    float64
 14  Креатинин, мкмоль/л                                        315 non-null    float64
 15  Натрий                                                     315 non-null    float64
 16  Калий                                                      315 non-null    float64
 17  СТЕПЕНИ ОЖИРЕНИЯ ПО ИМТ                                    315 non-null    float64
 18  Мерцательная аритмия                                       315 non-null    float64
 19  БА                                                         315 non-null    float64
 20  ХОБЛ                                                       315 non-null    float64
 21  ИБС. Стенокардия                                           315 non-null    float64
 22  АГ, степень                                                315 non-null    float64
 23  ХСН, стадии                                                315 non-null    float64
 24  Перенес/ не перенес КВИ_1                                  315 non-null    float64
 25  Тромбоэмболический синдром_1                               315 non-null    float64
 26  Гипертрофия миокарда левого желудочка по ЭКГ или ЭХО-КГ_1  315 non-null    float64
 27  Блокада НП Гисса_1                                         315 non-null    float64
 28  Блокада НП Гисса_2                                         315 non-null    float64
 29  AV- блокада_1                                              315 non-null    float64
 30  AV- блокада_2                                              315 non-null    float64
 31  AV- блокада_3                                              315 non-null    float64
 32  ХБП (исходн)_1                                             315 non-null    float64
 33  ХБП (исходн)_2                                             315 non-null    float64
 34  ХБП (исходн)_3                                             315 non-null    float64
 35  ХБП (исходн)_4                                             315 non-null    float64
 36  ХБП (исходн)_5                                             315 non-null    float64
 37  ХБП (исходн)_6                                             315 non-null    float64
 38  ХБП (исходн)_7                                             315 non-null    float64
 39  АКШ_1                                                      315 non-null    float64
dtypes: float64(40)
memory usage: 98.6 KB
""", line_numbers = True)
    
st.markdown(
"""
### Выводы
Был написан и применен класс для трансформации данных тестовой выборки, по аналогии с тренировочными данными (на их основании). Подготовлен пайплайн для кодирования данных, к определенным колонкам применяется OHE кодировани и скалирование.
"""
)
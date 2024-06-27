import streamlit as st
from set_params import load_page_setting

st.set_page_config(
    page_title="Baseline модель",
    page_icon="🚅"
)

# загрузка параметров фона стилей и т.д.
load_page_setting()

st.sidebar.header("Baseline модель")
st.sidebar.success("Обучение двух базовых моделей для дальнейшего сравнения")

st.markdown("""
# Создание Baseline модели

За baseline возьмем RandomForestClassifier. Оценивать качаество будем на валидационной выборке

""")

st.markdown(
"""
Первый baseline. Данные не подвергались трансформации. Результат метрики f1_macro= 0.6825
""" 
)

st.code("""
# разделим датасет на 2 выборки (train val)

data_base = data_eda_1v1_drop.copy()

x_train = data_base.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_train = data_base['Сахарный диабет (ЕСТЬ\НЕТ)']

x_train_baseline, x_val_baseline, y_train_baseline, y_val_baseline = train_test_split(
    x_train, y_train,
    test_size = 0.10, shuffle = True,
    random_state = RANDOM_STATE)

model = RandomForestClassifier(random_state = RANDOM_STATE)
model.fit(x_train_baseline, y_train_baseline)
        
predict = model.predict(x_val_baseline)
print(f'f1 score у baseline модели = {f1_score(y_val_baseline, predict, average = "macro")}')

f1 score у baseline модели = 0.6825396825396826 """, line_numbers = True)

st.markdown(
"""
Второй baseline. Данные пропущены через pipeline. Результат метрики f1_macro= 0.712
""" 
)

st.code("""# разделим датасет на 2 выборки (train val)

# сделаем копию датасета
data_base = data_eda_1v1_drop.copy()

x_train = data_base.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_train = data_base['Сахарный диабет (ЕСТЬ\НЕТ)']

x_train_baseline, x_val_baseline, y_train_baseline, y_val_baseline = train_test_split(
    x_train, y_train,
    test_size = 0.10, shuffle = True,
    random_state = RANDOM_STATE)

x_train_baseline = pd.DataFrame(pipeline_transform.transform(x_train_baseline))
x_train_baseline.columns = x_train_save_column

x_val_baseline = pd.DataFrame(pipeline_transform.transform(x_val_baseline))
x_val_baseline.columns = x_train_save_column

model = RandomForestClassifier(random_state = RANDOM_STATE)
model.fit(x_train_baseline, y_train_baseline)
        
predict = model.predict(x_val_baseline)
print(f'f1 score у baseline модели (pipeline encoder)= {f1_score(y_val_baseline, predict, average = "macro")}')

f1 score у baseline модели (pipeline encoder) = 0.7117117117117118 """, line_numbers = True)


st.markdown(
"""
### Выводы
В ходе эксперемента было вявлено, что baseline даёт лучшие метрики f1 макро = 0.71 на валидации используя кодированный датасет
"""
)
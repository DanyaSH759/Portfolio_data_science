import streamlit as st
from set_params import load_page_setting

# параметры описания страницы в браузере
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

# загрузка параметров фона стилей и т.д.
load_page_setting()

st.write(" # Итог работы нашей команды над задачей диабета 😎")

# боковое меню
st.sidebar.header("Информация о проекте")
st.sidebar.success("В меню выше отдельыне страницы с \
                    каждым блоком нашей работы.")

st.markdown(
    """
    ### Состав команды ʕ ᵔᴥᵔ ʔ
    """
)
col1, col2, col3 = st.columns([1, 1, 1])

col1.markdown(
    """
    Данила
    - githib [клик](https://github.com/DanyaSH759)
    - telegram [тук](https://t.me/shulyakds)
    """
)
col2.markdown(
    """
    Дмитрий
    - githib [клик](https://github.com/dimons8218)
    - telegram [тук](https://t.me/SukhorukovDmitry)
    """
)
col3.markdown(
    """
    Эдуард
    - githib [клик](https://github.com/eduardmakichyan)
    - telegram [тук](https://t.me/EduardMakichyan)
    """
)

st.markdown(
    """
    ### Задача нашей команды 🚨
    """
)
st.markdown(
    """
    Разработать систему определяющую сахарный диабет у пациента
    """
)

st.markdown(
    """
    ### Условия выполнения, данные, план работ 🕐🖋️📄:
    """
)

col1, col2, col3 = st.columns([1, 1, 1])

col1.markdown(
    """
    - Деление на трейн и тест 70/30
    - Random state - 654321
    - Метрика - F1 Macro
    - Метрики на train и test не должны сильно отличаться
    """
)
col2.markdown(
    """
    - В датасет входит 107 колонок с различной информацией
      о состоянии здоровья,
    принимаемых лекарствах, различных анализах и поставленных
      диагнозах реальных пациентов
    """
)
col3.markdown(
    """
    - Загрузка данных
    - Исследовательский анализ данных
    - Подготовка датасета
    - Создание Baseline модели
    - Подбор оптимальной модели
    - Проведение финального тестирования на тестовой выборке
    - Разработка сервиса для модели (по возможности)
    """
)

st.markdown(
    """
    ### Чек-лист выполненных работ ✔
    """
)

_1_ = st.checkbox("Загружены необходимые библиотеки и датасет", True)
_2_ = st.checkbox("Заполнены пропуски в датасете", True)
_3_ = st.checkbox("Проведено первичное исследования данных", True)
_4_ = st.checkbox("Подготовлен датасет для ML", True)
_5_ = st.checkbox("Сформирована baseline модель", True)
_6_ = st.checkbox("Подобрана оптимальная модель", True)
_7_ = st.checkbox("Проведено финальное тестирвоание лучшей модели", True)
_8_ = st.checkbox("Проведено исследование лучшей модели", True)
_9_ = st.checkbox("Реализован сервис streamlit для презентации работы")

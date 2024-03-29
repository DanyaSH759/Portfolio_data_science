# Описание задачи: 
Как только водитель забронировал автомобиль, сел за руль и выбрал маршрут, система должна оценить уровень риска. Если уровень риска высок, водитель увидит предупреждение и рекомендации по маршруту. Для этого нужно создать модель предсказания ДТП.

# Описание полей данных

- collisions — общая информация о ДТП. 
- parties — информация об участниках ДТП. 
- vehicles — информация о пострадавших машинах.

# План работы:

- Изучений файлов с данными
- EDA основных данных
- Разработка моделей
- Тестирование лучшей модели
- Общий вывод по модели


# Общий вывод:

В ходе исследования была выявлена самая лучшая модель для предсказания возможного ДТП - CatBoost. Результат тестовой выборки составляет 0.65. Модель не сильно превосходит случайное угадывание, всего на 15%, поэтому она лишь немного переходит порог адекватности, так же присутствует переобучение. Сказывается отсутствие факторов, которые могли бы корелировать с целевой переменной. Рекомендую добавить информацию о водительском стаже, износе тормозов и шин.

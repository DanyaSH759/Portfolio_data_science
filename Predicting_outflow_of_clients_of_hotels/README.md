# Описание задачи:
Цель проекта  - Прогнозирование оттока клиентов в сети отелей «Как в гостях»

Нужно разработать систему, которая предсказывает отказ от брони. Если модель покажет, что бронь будет отменена, то клиенту предлагается внести депозит. Размер депозита — 80% от стоимости номера за одни сутки и затрат на разовую уборку. Деньги будут списаны со счёта клиента, если он всё же отменит бронь.

# Предоставленные данные:
В таблицах hotel_train и hotel_test содержатся одинаковые столбцы:
-	id — номер записи;
-	adults — количество взрослых постояльцев;
-	arrival_date_year — год заезда;
-	arrival_date_month — месяц заезда;
-	arrival_date_week_number — неделя заезда;
-	arrival_date_day_of_month — день заезда;
-	babies — количество младенцев;
-	booking_changes — количество изменений параметров заказа;
-	children — количество детей от 3 до 14 лет;
-	country — гражданство постояльца;
-	customer_type — тип заказчика: 
-	Contract — договор с юридическим лицом;
-	Group — групповой заезд;
-	Transient — не связано с договором или групповым заездом;
-	Transient-party — не связано с договором или групповым заездом, но связано с бронированием типа Transient.
-	days_in_waiting_list — сколько дней заказ ожидал подтверждения;
-	distribution_channel — канал дистрибуции заказа;
-	is_canceled — отмена заказа;
-	is_repeated_guest — признак того, что гость бронирует номер второй раз;
-	lead_time — количество дней между датой бронирования и датой прибытия;
-	meal — опции заказа:
>	SC — нет дополнительных опций;
>	BB — включён завтрак;
>	HB — включён завтрак и обед;
>	FB — включён завтрак, обед и ужин.
-	previous_bookings_not_canceled — количество подтверждённых заказов у клиента;
-	previous_cancellations — количество отменённых заказов у клиента;
-	required_car_parking_spaces — необходимость места для автомобиля;
-	reserved_room_type — тип забронированной комнаты;
-	stays_in_weekend_nights — количество ночей в выходные дни;
-	stays_in_week_nights — количество ночей в будние дни;
-	total_nights — общее количество ночей;
-	total_of_special_requests — количество специальных отметок.

# План работ:
- Изучений файлов с данными
- Предобработка и исследовательский анализ данных
- Вычисление бизнес-метрики
- Разработка модели ML
- Выявить признаки «ненадёжного» клиента

# Общий вывод:
В ходе исследования была выявлена самая лучшая модель для решения задачи классификации (отменит ли клиент бронь в отеле или нет) - "Случайный лес". Параметры auc-roc данной модели составляет 0.79. Данная метрика была выбрана по причине того, что она вычисляется из показателей True Positive и False Positive.
Суммарная разница между системами (депозит и без депозита) с учетом затрат на разработку систему составила 568 тысяч, то есть в течении года затраты на разработку системы окупятся за счет введения депозитов.

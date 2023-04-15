# yolov8
Реализовано раздельное детектирование и треккинг объектов по уже готовым массивам из детекции, что позволяет подбирать архитектуру трекера и постобработки изолировано, без необходимости запуска предикта YOLO (ускорение подбора треккера и постобработки примерно в 100 раз и без траты лимита GPU в колаб) - также потенциально возможна генетика архитектуры постобработки и треккинга. Кроме того, реализован проброс касок и жилетов мимо трекера, что позволяет лучше трековать людей, а также не приводит к фильтрации касок и жилетов трекером, это позволяет лучше детектировать нарушения, избегать ложных фиксаций нарушений. Также реализовано восстановление в массиве информации нетрекованных боксов людей (которые трекер отфильтровал, отбросил).
В постобработке реализовано два подхода по подсчету людей а также вероятностный подход фиксации нарушений. Под вероятностным понимается отношение количества кадров в которых конкретный айди находится в каске и жилете к общему количеству кадров с этим айди (отношение больше минимального порога - нарушения нет).

![hippo](https://imgur.com/a/UTX1ofV)

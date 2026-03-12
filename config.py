"""
Конфигурация алгоритма Re-identification.

Содержит параметры порогов, весов и правил для объединения треков в сущности.
"""

from dataclasses import dataclass


@dataclass
class ReIDConfig:
    """Конфигурация алгоритма ReID."""
    
    # Пороги для жёстких фильтров
    # Если различие превышает порог, треки не могут быть объединены
    
    # Цвет: если цвета заданы и различаются, штраф
    color_mismatch_penalty: float = 0.5  # Штраф за несовпадение цвета (0..1)
    color_match_boost: float = 0.1  # Бонус за совпадение цвета
    
    # Марка: если марки заданы и различаются, штраф
    make_mismatch_penalty: float = 0.7  # Штраф за несовпадение марки
    make_match_boost: float = 0.15  # Бонус за совпадение марки

    # Тип ТС
    vehicle_type_mismatch_penalty: float = 0.5  # Штраф за несовпадение типа ТС
    vehicle_type_match_boost: float = 0.1  # Бонус за совпадение типа ТС

    # Тип кузова
    body_type_mismatch_penalty: float = 0.2  # Штраф за несовпадение типа кузова
    body_type_match_boost: float = 0.05  # Бонус за совпадение типа кузова

    # Класс размера
    size_class_mismatch_penalty: float = 0.2  # Штраф за несовпадение класса размера
    size_class_match_boost: float = 0.05  # Бонус за совпадение класса размера

    # Номерной знак: очень сильный признак
    plate_full_match_boost: float = 0.7   # Большой бонус за полный матч номера
    plate_mismatch_penalty: float = 0.9   # Сильный штраф за явное несовпадение
    plate_partial_match_boost: float = 0.3  # Бонус за частичное совпадение (например, по шаблону)
    
    # Источник: если источники одинаковые, небольшой бонус (одна камера)
    same_source_boost: float = 0.05  # Бонус за один источник
    
    # Веса для признаков траектории (сумма должна быть ~1.0)
    weight_avg_speed: float = 0.15  # Вес средней скорости
    weight_median_speed: float = 0.15  # Вес медианной скорости
    weight_heading: float = 0.20  # Вес направления
    weight_path_length: float = 0.10  # Вес длины пути
    weight_displacement_distance: float = 0.15  # Вес расстояния перемещения
    weight_displacement_heading: float = 0.10  # Вес направления перемещения
    weight_time_gap: float = 0.10  # Вес временного разрыва
    weight_geographic_proximity: float = 0.05  # Вес географической близости
    
    # Пороги для нормализации различий
    max_speed_diff_mps: float = 20.0  # Максимальная разница скоростей для нормализации
    max_path_length_diff_m: float = 1000.0  # Максимальная разница длины пути
    max_displacement_diff_m: float = 500.0  # Максимальная разница перемещения
    max_time_gap_seconds: float = 300.0  # Максимальный допустимый разрыв (5 минут)
    max_geographic_distance_m: float = 200.0  # Максимальное расстояние для близости
    
    # Порог confidence для объединения треков в одну сущность
    confidence_threshold: float = 0.75  # Минимальный confidence для объединения
    
    # Порог для географической близости (расстояние между конечной точкой одного
    # трека и начальной точкой другого)
    geographic_proximity_threshold_m: float = 100.0  # Метры
    
    def normalize_speed_diff(self, diff: float) -> float:
        """Нормализация разницы скоростей в диапазон [0, 1]."""
        if diff < 0:
            diff = -diff
        return min(1.0, diff / self.max_speed_diff_mps)
    
    def normalize_path_length_diff(self, diff: float) -> float:
        """Нормализация разницы длины пути в диапазон [0, 1]."""
        if diff < 0:
            diff = -diff
        return min(1.0, diff / self.max_path_length_diff_m)
    
    def normalize_displacement_diff(self, diff: float) -> float:
        """Нормализация разницы перемещения в диапазон [0, 1]."""
        if diff < 0:
            diff = -diff
        return min(1.0, diff / self.max_displacement_diff_m)
    
    def normalize_time_gap(self, gap: float) -> float:
        """Нормализация временного разрыва в диапазон [0, 1]."""
        if gap < 0:  # Перекрытие
            return 0.0
        return min(1.0, gap / self.max_time_gap_seconds)
    
    def normalize_geographic_distance(self, distance: float) -> float:
        """Нормализация географического расстояния в диапазон [0, 1]."""
        return min(1.0, distance / self.max_geographic_distance_m)
    
    def normalize_heading_diff(self, diff: float) -> float:
        """Нормализация разницы направлений в диапазон [0, 1]."""
        # Учитываем цикличность (0 и 360 близки)
        diff = abs(diff)
        if diff > 180:
            diff = 360 - diff
        return diff / 180.0  # Нормализация к [0, 1]


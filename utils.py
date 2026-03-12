"""
Утилиты для нормализации данных и вычисления признаков треков.

Все функции детерминированы и не используют внешние зависимости.
"""

from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from schema import Point, Track


def normalize_time(ts: str) -> datetime:
    """
    Нормализация времени из ISO 8601 в datetime.
    
    Args:
        ts: Строка времени в формате ISO 8601
        
    Returns:
        datetime: Объект datetime
    """
    # Замена 'Z' на '+00:00' для совместимости с fromisoformat
    normalized = ts.replace('Z', '+00:00')
    return datetime.fromisoformat(normalized)


def sort_points_by_time(points: List[Point]) -> List[Point]:
    """
    Сортировка точек по времени (ts).
    
    Args:
        points: Список точек
        
    Returns:
        List[Point]: Отсортированный список точек
    """
    return sorted(points, key=lambda p: normalize_time(p.ts))


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Вычисление расстояния между двумя точками на сфере (формула гаверсинуса).
    
    Args:
        lat1, lon1: Координаты первой точки
        lat2, lon2: Координаты второй точки
        
    Returns:
        float: Расстояние в метрах
    """
    import math
    
    # Радиус Земли в метрах
    R = 6371000.0
    
    # Перевод в радианы
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Формула гаверсинуса
    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def compute_track_features(track: Track) -> Dict[str, Any]:
    """
    Вычисление признаков трека для ReID.
    
    Вычисляет:
    - Средняя скорость (м/с)
    - Медиана скорости (м/с)
    - Доминирующее направление (градусы)
    - Длина пути (метры)
    - Bounding box (min_lat, max_lat, min_lon, max_lon)
    - Типичная скорость (м/с) - медиана или средняя
    - Вектор перемещения start->end (lat_diff, lon_diff, distance_m, heading_deg)
    
    Args:
        track: Трек для анализа
        
    Returns:
        Dict: Словарь с признаками
    """
    # Сортировка точек по времени
    sorted_points = sort_points_by_time(track.points)
    
    # Извлечение скоростей (исключая None)
    speeds = [p.speed_mps for p in sorted_points if p.speed_mps is not None]
    
    # Извлечение направлений (исключая None)
    headings = [p.heading_deg for p in sorted_points if p.heading_deg is not None]
    
    # Вычисление средней скорости
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    
    # Вычисление медианы скорости
    median_speed = None
    if speeds:
        sorted_speeds = sorted(speeds)
        n = len(sorted_speeds)
        if n % 2 == 0:
            median_speed = (sorted_speeds[n // 2 - 1] + sorted_speeds[n // 2]) / 2
        else:
            median_speed = sorted_speeds[n // 2]
    
    # Доминирующее направление (медиана направлений)
    dominant_heading = None
    if headings:
        # Нормализация направлений для вычисления медианы
        # Используем подход с векторами для учёта цикличности 0-360
        import math
        sin_sum = sum(math.sin(math.radians(h)) for h in headings)
        cos_sum = sum(math.cos(math.radians(h)) for h in headings)
        avg_angle_rad = math.atan2(sin_sum, cos_sum)
        dominant_heading = math.degrees(avg_angle_rad) % 360
    
    # Длина пути (сумма расстояний между последовательными точками)
    path_length = 0.0
    for i in range(len(sorted_points) - 1):
        p1 = sorted_points[i]
        p2 = sorted_points[i + 1]
        path_length += haversine_distance(p1.lat, p1.lon, p2.lat, p2.lon)
    
    # Bounding box
    lats = [p.lat for p in sorted_points]
    lons = [p.lon for p in sorted_points]
    bbox = {
        'min_lat': min(lats),
        'max_lat': max(lats),
        'min_lon': min(lons),
        'max_lon': max(lons),
    }
    
    # Типичная скорость (используем медиану, если есть, иначе среднюю)
    typical_speed = median_speed if median_speed is not None else avg_speed
    
    # Вектор перемещения start->end
    start_point = sorted_points[0]
    end_point = sorted_points[-1]
    
    lat_diff = end_point.lat - start_point.lat
    lon_diff = end_point.lon - start_point.lon
    distance_m = haversine_distance(
        start_point.lat, start_point.lon,
        end_point.lat, end_point.lon
    )
    
    # Вычисление направления перемещения
    import math
    heading_deg = None
    if distance_m > 0:
        lat1_rad = math.radians(start_point.lat)
        lat2_rad = math.radians(end_point.lat)
        lon_diff_rad = math.radians(lon_diff)
        
        y = math.sin(lon_diff_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff_rad))
        
        heading_rad = math.atan2(y, x)
        heading_deg = math.degrees(heading_rad) % 360
    
    displacement = {
        'lat_diff': lat_diff,
        'lon_diff': lon_diff,
        'distance_m': distance_m,
        'heading_deg': heading_deg,
    }
    
    return {
        'avg_speed': avg_speed,
        'median_speed': median_speed,
        'dominant_heading': dominant_heading,
        'path_length': path_length,
        'bbox': bbox,
        'typical_speed': typical_speed,
        'displacement': displacement,
        'num_points': len(sorted_points),
        'duration_seconds': (
            (normalize_time(track.end_ts) - normalize_time(track.start_ts)).total_seconds()
        ),
    }


def time_overlap(track1: Track, track2: Track) -> float:
    """
    Вычисление перекрытия по времени между двумя треками (в секундах).
    
    Args:
        track1, track2: Треки для сравнения
        
    Returns:
        float: Перекрытие в секундах (0 если нет перекрытия)
    """
    start1 = normalize_time(track1.start_ts)
    end1 = normalize_time(track1.end_ts)
    start2 = normalize_time(track2.start_ts)
    end2 = normalize_time(track2.end_ts)
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    return (overlap_end - overlap_start).total_seconds()


def time_gap(track1: Track, track2: Track) -> float:
    """
    Вычисление временного разрыва между двумя треками (в секундах).
    
    Положительное значение означает, что track2 начинается после track1.
    Отрицательное значение означает перекрытие.
    
    Args:
        track1, track2: Треки для сравнения
        
    Returns:
        float: Разрыв в секундах
    """
    end1 = normalize_time(track1.end_ts)
    start2 = normalize_time(track2.start_ts)
    
    return (start2 - end1).total_seconds()


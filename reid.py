"""
Модуль Re-identification: идентификация одного и того же автомобиля
по косвенным признакам при частично или полностью скрытом номерном знаке.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from schema import Track, Batch
from utils import (
    compute_track_features, time_gap, haversine_distance,
    sort_points_by_time, normalize_time
)
from config import ReIDConfig


@dataclass
class Entity:
    """Сущность (гипотеза об одном автомобиле)."""
    entity_id: str
    track_ids: Set[str] = field(default_factory=set)


@dataclass
class Link:
    """Связь трека с сущностью."""
    entity_id: str
    track_id: str
    confidence: float  # 0..1
    reasons: List[str] = field(default_factory=list)


def _normalize_string(s: Optional[str]) -> Optional[str]:
    """Нормализация строки для сравнения (приведение к нижнему регистру, удаление пробелов)."""
    if s is None:
        return None
    return s.strip().lower()


def _compare_color(color1: Optional[str], color2: Optional[str]) -> Tuple[float, List[str]]:
    """
    Сравнение цветов двух треков.
    
    Returns:
        Tuple[float, List[str]]: (score, reasons)
        score: -1.0 (несовпадение), 0.0 (неизвестно), 1.0 (совпадение)
    """
    c1 = _normalize_string(color1)
    c2 = _normalize_string(color2)
    
    if c1 is None or c2 is None:
        return 0.0, []
    
    if c1 == c2:
        return 1.0, ["совпадение цвета"]
    else:
        return -1.0, [f"различие цвета: {color1} vs {color2}"]


def _compare_make(make1: Optional[str], make2: Optional[str]) -> Tuple[float, List[str]]:
    """
    Сравнение марок двух треков.
    
    Returns:
        Tuple[float, List[str]]: (score, reasons)
        score: -1.0 (несовпадение), 0.0 (неизвестно), 1.0 (совпадение)
    """
    m1 = _normalize_string(make1)
    m2 = _normalize_string(make2)
    
    if m1 is None or m2 is None:
        return 0.0, []
    
    if m1 == m2:
        return 1.0, ["совпадение марки"]
    else:
        return -1.0, [f"различие марки: {make1} vs {make2}"]


def _compare_simple_categorical(
    v1: Optional[str],
    v2: Optional[str],
    equal_reason: str,
    diff_reason_template: str
) -> Tuple[float, List[str]]:
    """
    Универсальное сравнение категориальных признаков (vehicle_type, body_type, size_class).

    Returns:
        Tuple[float, List[str]]: (score, reasons)
        score: -1.0 (несовпадение), 0.0 (неизвестно), 1.0 (совпадение)
    """
    n1 = _normalize_string(v1)
    n2 = _normalize_string(v2)

    if n1 is None or n2 is None:
        return 0.0, []

    if n1 == n2:
        return 1.0, [equal_reason]
    else:
        return -1.0, [diff_reason_template.format(v1=v1, v2=v2)]


def _compare_plate(
    plate1: Optional[str],
    plate2: Optional[str],
    conf1: Optional[float],
    conf2: Optional[float],
) -> Tuple[float, List[str]]:
    """
    Сравнение номерных знаков.

    Идея:
    - Полное совпадение при достаточной уверенности => сильный плюс.
    - Явное различие при высокой уверенности обоих => сильный минус.
    - Если один или оба номера с низкой уверенностью / отсутствуют — влияние слабое или нулевое.
    """
    reasons: List[str] = []

    n1 = _normalize_string(plate1)
    n2 = _normalize_string(plate2)

    if n1 is None or n2 is None:
        return 0.0, []

    # Нормализуем confidence (если None, считаем ~0.5 — средняя уверенность)
    c1 = conf1 if conf1 is not None else 0.5
    c2 = conf2 if conf2 is not None else 0.5
    avg_conf = (c1 + c2) / 2.0

    if n1 == n2:
        reasons.append(f"совпадение номерного знака (conf≈{avg_conf:.2f})")
        return 1.0 * avg_conf, reasons

    # При сильно разных номерах и высокой уверенности — отрицательный сигнал
    if avg_conf >= 0.7:
        reasons.append(f"различие номерного знака при высокой уверенности (conf≈{avg_conf:.2f})")
        return -1.0 * avg_conf, reasons

    # При низкой уверенности не наказываем сильно
    reasons.append(f"различие номерного знака при низкой уверенности (conf≈{avg_conf:.2f})")
    return -0.2 * avg_conf, reasons


def _plates_strong_conflict(t1: Track, t2: Track, conf_threshold: float = 0.8) -> bool:
    """
    Жёсткое правило для permissioned/аудит-сценариев:
    если у обоих треков распознан номер и он различается при высокой уверенности,
    то такие треки НЕ объединяем независимо от остальных признаков.
    """
    p1 = _normalize_string(t1.plate_text)
    p2 = _normalize_string(t2.plate_text)
    if p1 is None or p2 is None:
        return False
    if p1 == p2:
        return False
    c1 = t1.plate_confidence if getattr(t1, "plate_confidence", None) is not None else 0.5
    c2 = t2.plate_confidence if getattr(t2, "plate_confidence", None) is not None else 0.5
    return c1 >= conf_threshold and c2 >= conf_threshold


def _compute_trajectory_similarity(
    track1: Track,
    track2: Track,
    features1: Dict,
    features2: Dict,
    config: ReIDConfig
) -> Tuple[float, List[str]]:
    """
    Вычисление сходства траекторий двух треков.
    
    Returns:
        Tuple[float, List[str]]: (similarity_score, reasons)
        similarity_score: значение от 0 (непохожи) до 1 (очень похожи)
    """
    reasons = []
    score = 0.0
    total_weight = 0.0
    
    # Сравнение средней скорости
    if features1['avg_speed'] is not None and features2['avg_speed'] is not None:
        diff = abs(features1['avg_speed'] - features2['avg_speed'])
        normalized_diff = config.normalize_speed_diff(diff)
        similarity = 1.0 - normalized_diff
        score += similarity * config.weight_avg_speed
        total_weight += config.weight_avg_speed
        reasons.append(f"средняя скорость: diff={diff:.2f} м/с, similarity={similarity:.2f}")
    
    # Сравнение медианной скорости
    if features1['median_speed'] is not None and features2['median_speed'] is not None:
        diff = abs(features1['median_speed'] - features2['median_speed'])
        normalized_diff = config.normalize_speed_diff(diff)
        similarity = 1.0 - normalized_diff
        score += similarity * config.weight_median_speed
        total_weight += config.weight_median_speed
        reasons.append(f"медианная скорость: diff={diff:.2f} м/с, similarity={similarity:.2f}")
    
    # Сравнение направления
    if features1['dominant_heading'] is not None and features2['dominant_heading'] is not None:
        diff = abs(features1['dominant_heading'] - features2['dominant_heading'])
        normalized_diff = config.normalize_heading_diff(diff)
        similarity = 1.0 - normalized_diff
        score += similarity * config.weight_heading
        total_weight += config.weight_heading
        reasons.append(f"направление: diff={diff:.1f}°, similarity={similarity:.2f}")
    
    # Сравнение длины пути
    diff = abs(features1['path_length'] - features2['path_length'])
    normalized_diff = config.normalize_path_length_diff(diff)
    similarity = 1.0 - normalized_diff
    score += similarity * config.weight_path_length
    total_weight += config.weight_path_length
    reasons.append(f"длина пути: diff={diff:.1f} м, similarity={similarity:.2f}")
    
    # Сравнение перемещения (расстояние)
    if (features1['displacement']['distance_m'] is not None and
            features2['displacement']['distance_m'] is not None):
        diff = abs(features1['displacement']['distance_m'] - features2['displacement']['distance_m'])
        normalized_diff = config.normalize_displacement_diff(diff)
        similarity = 1.0 - normalized_diff
        score += similarity * config.weight_displacement_distance
        total_weight += config.weight_displacement_distance
        reasons.append(f"перемещение: diff={diff:.1f} м, similarity={similarity:.2f}")
    
    # Сравнение направления перемещения
    if (features1['displacement']['heading_deg'] is not None and
            features2['displacement']['heading_deg'] is not None):
        diff = abs(features1['displacement']['heading_deg'] - features2['displacement']['heading_deg'])
        normalized_diff = config.normalize_heading_diff(diff)
        similarity = 1.0 - normalized_diff
        score += similarity * config.weight_displacement_heading
        total_weight += config.weight_displacement_heading
        reasons.append(f"направление перемещения: diff={diff:.1f}°, similarity={similarity:.2f}")
    
    # Временной разрыв
    gap = time_gap(track1, track2)
    normalized_gap = config.normalize_time_gap(gap)
    # Малый разрыв или перекрытие увеличивают сходство
    similarity = 1.0 - normalized_gap
    score += similarity * config.weight_time_gap
    total_weight += config.weight_time_gap
    if gap < 0:
        reasons.append(f"временное перекрытие: {abs(gap):.1f} с")
    else:
        reasons.append(f"временной разрыв: {gap:.1f} с, similarity={similarity:.2f}")
    
    # Географическая близость (расстояние между конечной точкой одного трека
    # и начальной точкой другого)
    sorted_points1 = sort_points_by_time(track1.points)
    sorted_points2 = sort_points_by_time(track2.points)
    
    end_point1 = sorted_points1[-1]
    start_point2 = sorted_points2[0]
    
    distance = haversine_distance(
        end_point1.lat, end_point1.lon,
        start_point2.lat, start_point2.lon
    )
    
    normalized_distance = config.normalize_geographic_distance(distance)
    similarity = 1.0 - normalized_distance
    score += similarity * config.weight_geographic_proximity
    total_weight += config.weight_geographic_proximity
    reasons.append(f"географическая близость: {distance:.1f} м, similarity={similarity:.2f}")
    
    # Нормализация по суммарному весу
    if total_weight > 0:
        score = score / total_weight
    
    return score, reasons


def _compute_similarity(
    track1: Track,
    track2: Track,
    features1: Dict,
    features2: Dict,
    config: ReIDConfig
) -> Tuple[float, List[str]]:
    """
    Вычисление общего сходства между двумя треками.
    
    Returns:
        Tuple[float, List[str]]: (confidence, reasons)
        confidence: значение от 0 до 1
    """
    reasons = []
    base_score = 0.0
    
    # Жёсткие / полужёсткие фильтры
    # Цвет
    color_score, color_reasons = _compare_color(track1.color, track2.color)
    if color_score < 0:
        base_score -= config.color_mismatch_penalty
        reasons.extend(color_reasons)
    elif color_score > 0:
        base_score += config.color_match_boost
        reasons.extend(color_reasons)
    
    # Марка
    make_score, make_reasons = _compare_make(track1.make, track2.make)
    if make_score < 0:
        base_score -= config.make_mismatch_penalty
        reasons.extend(make_reasons)
    elif make_score > 0:
        base_score += config.make_match_boost
        reasons.extend(make_reasons)
    
    # Тип ТС
    vehicle_type_score, vehicle_type_reasons = _compare_simple_categorical(
        track1.vehicle_type,
        track2.vehicle_type,
        equal_reason="совпадение типа ТС",
        diff_reason_template="различие типа ТС: {v1} vs {v2}",
    )
    if vehicle_type_score < 0:
        base_score -= config.vehicle_type_mismatch_penalty
        reasons.extend(vehicle_type_reasons)
    elif vehicle_type_score > 0:
        base_score += config.vehicle_type_match_boost
        reasons.extend(vehicle_type_reasons)

    # Тип кузова
    body_type_score, body_type_reasons = _compare_simple_categorical(
        track1.body_type,
        track2.body_type,
        equal_reason="совпадение типа кузова",
        diff_reason_template="различие типа кузова: {v1} vs {v2}",
    )
    if body_type_score < 0:
        base_score -= config.body_type_mismatch_penalty
        reasons.extend(body_type_reasons)
    elif body_type_score > 0:
        base_score += config.body_type_match_boost
        reasons.extend(body_type_reasons)

    # Класс размера
    size_class_score, size_class_reasons = _compare_simple_categorical(
        track1.size_class,
        track2.size_class,
        equal_reason="совпадение класса размера",
        diff_reason_template="различие класса размера: {v1} vs {v2}",
    )
    if size_class_score < 0:
        base_score -= config.size_class_mismatch_penalty
        reasons.extend(size_class_reasons)
    elif size_class_score > 0:
        base_score += config.size_class_match_boost
        reasons.extend(size_class_reasons)

    # Номерной знак
    plate_score, plate_reasons = _compare_plate(
        track1.plate_text,
        track2.plate_text,
        getattr(track1, "plate_confidence", None),
        getattr(track2, "plate_confidence", None),
    )
    if plate_score < 0:
        # Штраф пропорционален plate_score (отрицательный) и plate_mismatch_penalty
        base_score += plate_score * config.plate_mismatch_penalty
        reasons.extend(plate_reasons)
    elif plate_score > 0:
        # Бонус за совпадение / частичное совпадение
        base_score += plate_score * config.plate_full_match_boost
        reasons.extend(plate_reasons)

    # Источник
    if track1.source_id == track2.source_id:
        base_score += config.same_source_boost
        reasons.append("один источник наблюдения")
    
    # Сходство траектории
    trajectory_score, trajectory_reasons = _compute_trajectory_similarity(
        track1, track2, features1, features2, config
    )
    
    # Итоговый confidence: базовая оценка + сходство траектории
    # Базовая оценка может быть отрицательной, поэтому используем сигмоиду
    confidence = base_score + trajectory_score
    
    # Нормализация через сигмоиду для получения значения в [0, 1]
    # Используем простую сигмоиду: 1 / (1 + exp(-k * (x - 0.5)))
    import math
    k = 10.0  # Коэффициент крутизны
    confidence = 1.0 / (1.0 + math.exp(-k * (confidence - 0.5)))
    
    reasons.extend(trajectory_reasons)
    reasons.append(f"итоговый confidence: {confidence:.3f}")
    
    return confidence, reasons


def reidentify(batch: Batch, config: Optional[ReIDConfig] = None) -> Tuple[List[Entity], List[Link]]:
    """
    Основной алгоритм Re-identification.
    
    Алгоритм:
    1. Вычисление признаков для всех треков
    2. Попарное сравнение всех треков (confidence матрица)
    3. Детерминированная greedy-кластеризация с complete-linkage:
       трек добавляется в сущность только если confidence со ВСЕМИ треками сущности >= threshold
       (это предотвращает «склеивание цепочкой», когда внутри сущности есть пары с низким confidence).
    
    Args:
        batch: Пакет треков для обработки
        config: Конфигурация ReID (по умолчанию используется ReIDConfig())
        
    Returns:
        Tuple[List[Entity], List[Link]]: (список сущностей, список связей)
    """
    if config is None:
        config = ReIDConfig()

    # Детерминированный порядок треков (важно для воспроизводимости)
    tracks_sorted = sorted(batch.tracks, key=lambda t: (normalize_time(t.start_ts), t.track_id))
    track_by_id: Dict[str, Track] = {t.track_id: t for t in tracks_sorted}

    # Вычисление признаков
    track_features: Dict[str, Dict[str, Any]] = {
        t.track_id: compute_track_features(t) for t in tracks_sorted
    }

    # Матрица confidence (без хранения reasons для всех пар — экономия памяти)
    pair_conf: Dict[Tuple[str, str], float] = {}

    def _pair_key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    for i in range(len(tracks_sorted)):
        for j in range(i + 1, len(tracks_sorted)):
            t1 = tracks_sorted[i]
            t2 = tracks_sorted[j]
            c, _ = _compute_similarity(
                t1,
                t2,
                track_features[t1.track_id],
                track_features[t2.track_id],
                config,
            )
            pair_conf[_pair_key(t1.track_id, t2.track_id)] = c

    def get_conf(a: str, b: str) -> float:
        if a == b:
            return 1.0
        return pair_conf[_pair_key(a, b)]

    # Greedy clustering with complete-linkage constraint
    entities: List[Entity] = []
    track_to_entity: Dict[str, str] = {}

    for t in tracks_sorted:
        best_entity_idx: Optional[int] = None
        best_entity_score: float = -1.0

        for idx, ent in enumerate(entities):
            member_ids = sorted(ent.track_ids)

            # Жёсткое отсечение по конфликтующему номеру
            if any(_plates_strong_conflict(t, track_by_id[mid]) for mid in member_ids):
                continue

            confs = [get_conf(t.track_id, mid) for mid in member_ids]
            min_conf = min(confs) if confs else 1.0
            if min_conf < config.confidence_threshold:
                continue

            avg_conf = sum(confs) / len(confs) if confs else 1.0

            # Выбираем сущность с максимальным avg_conf (детерминированно)
            if avg_conf > best_entity_score:
                best_entity_score = avg_conf
                best_entity_idx = idx

        if best_entity_idx is None:
            entity_id = f"entity_{len(entities)}"
            ent = Entity(entity_id=entity_id, track_ids={t.track_id})
            entities.append(ent)
            track_to_entity[t.track_id] = entity_id
        else:
            ent = entities[best_entity_idx]
            ent.track_ids.add(t.track_id)
            track_to_entity[t.track_id] = ent.entity_id

    # Формируем Links: 1 link на трек (привязка к сущности)
    links: List[Link] = []
    for ent in entities:
        member_ids = sorted(ent.track_ids)
        for tid in member_ids:
            if len(member_ids) == 1:
                links.append(
                    Link(
                        entity_id=ent.entity_id,
                        track_id=tid,
                        confidence=1.0,
                        reasons=["singleton"],
                    )
                )
                continue

            # Лучший матч трека внутри сущности
            best_mid = None
            best_conf = -1.0
            for mid in member_ids:
                if mid == tid:
                    continue
                c = get_conf(tid, mid)
                if c > best_conf:
                    best_conf = c
                    best_mid = mid

            assert best_mid is not None
            # reasons пересчитываем только для одной пары (tid, best_mid)
            t1 = track_by_id[tid]
            t2 = track_by_id[best_mid]
            c, reasons = _compute_similarity(
                t1,
                t2,
                track_features[tid],
                track_features[best_mid],
                config,
            )
            reasons = [f"best_match_with={best_mid}"] + reasons
            links.append(
                Link(
                    entity_id=ent.entity_id,
                    track_id=tid,
                    confidence=float(c),
                    reasons=reasons,
                )
            )

    return entities, links


"""
Отдельный модуль для расчёта и накопления статистики качества ReID.
"""

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


def extract_reference_groups(reference_payload):
    """
    Извлекает эталонные группы из поддерживаемых форматов файла.

    Поддерживаемые структуры:
    - {"entities": [...]} c полем track_ids
    - {"vehicles": [...]} c полем track_ids
    - [...] где элементы содержат track_ids
    """
    if isinstance(reference_payload, list):
        return reference_payload

    if not isinstance(reference_payload, dict):
        raise ValueError("Эталонный файл должен содержать объект или список групп")

    if isinstance(reference_payload.get("entities"), list):
        return reference_payload["entities"]

    if isinstance(reference_payload.get("vehicles"), list):
        return reference_payload["vehicles"]

    raise ValueError(
        "Не удалось найти группы в эталонном файле: ожидается ключ 'entities' или 'vehicles'"
    )


def _normalize_entity_groups(entities_payload):
    """Преобразование списка сущностей в список множеств track_id."""
    groups = []
    for entity in entities_payload:
        track_ids = {
            str(track_id).strip()
            for track_id in entity.get("track_ids", [])
            if str(track_id).strip()
        }
        if track_ids:
            groups.append(track_ids)
    return groups


def _safe_divide(numerator, denominator, default=0.0):
    """Безопасное деление с дефолтным значением."""
    return numerator / denominator if denominator else default


def _build_track_map(groups, common_tracks_set):
    """Строит отображение track_id -> индекс группы."""
    track_map = {}
    for index, group in enumerate(groups):
        for track_id in group & common_tracks_set:
            track_map[track_id] = index
    return track_map


def _compute_cluster_purity(predicted_groups, reference_map, common_tracks_set):
    """
    Взвешенная purity по предсказанным кластерам.

    Показывает, насколько каждый найденный кластер состоит из одного эталонного объекта.
    """
    total_tracks = 0
    dominant_tracks = 0

    for group in predicted_groups:
        group_tracks = sorted(group & common_tracks_set)
        if not group_tracks:
            continue
        counts = {}
        for track_id in group_tracks:
            ref_idx = reference_map[track_id]
            counts[ref_idx] = counts.get(ref_idx, 0) + 1
        dominant_tracks += max(counts.values())
        total_tracks += len(group_tracks)

    return _safe_divide(dominant_tracks, total_tracks, default=1.0)


def _compute_entity_continuity(reference_groups, predicted_map, common_tracks_set):
    """
    Доля треков, которые для каждой эталонной машины попали в её крупнейший предсказанный фрагмент.

    Значение 1.0 означает отсутствие фрагментации эталонных сущностей.
    """
    total_tracks = 0
    largest_fragments = 0

    for group in reference_groups:
        group_tracks = sorted(group & common_tracks_set)
        if not group_tracks:
            continue
        counts = {}
        for track_id in group_tracks:
            pred_idx = predicted_map[track_id]
            counts[pred_idx] = counts.get(pred_idx, 0) + 1
        largest_fragments += max(counts.values())
        total_tracks += len(group_tracks)

    return _safe_divide(largest_fragments, total_tracks, default=1.0)


def _compute_temporal_coherence(predicted_groups, reference_map, track_time_lookup, common_tracks_set):
    """
    Доля соседних по времени переходов внутри найденных кластеров, которые согласованы с эталоном.

    Переход считается согласованным, если соседние по времени треки относятся к одной эталонной машине
    и не имеют отрицательного временного разрыва.
    """
    evaluated_transitions = 0
    coherent_transitions = 0

    for group in predicted_groups:
        timed_tracks = []
        for track_id in group & common_tracks_set:
            time_info = track_time_lookup.get(track_id)
            if time_info is None:
                continue
            timed_tracks.append((time_info["start"], time_info["end"], track_id))

        timed_tracks.sort(key=lambda item: (item[0], item[2]))
        if len(timed_tracks) < 2:
            continue

        for left, right in zip(timed_tracks, timed_tracks[1:]):
            evaluated_transitions += 1
            same_reference = reference_map[left[2]] == reference_map[right[2]]
            non_negative_gap = right[0] >= left[1]
            if same_reference and non_negative_gap:
                coherent_transitions += 1

    return _safe_divide(coherent_transitions, evaluated_transitions, default=1.0)


def evaluate_entities(predicted_entities, reference_entities, track_time_lookup=None):
    """
    Оценка качества кластеризации по попарочным связям треков.

    Сравнение выполняется только по пересечению track_id между прогнозом и эталоном,
    чтобы метрика оставалась устойчивой при частично несовпадающих наборах.
    """
    predicted_groups = _normalize_entity_groups(predicted_entities)
    reference_groups = _normalize_entity_groups(reference_entities)

    predicted_tracks = set().union(*predicted_groups) if predicted_groups else set()
    reference_tracks = set().union(*reference_groups) if reference_groups else set()
    common_tracks = sorted(predicted_tracks & reference_tracks)
    common_tracks_set = set(common_tracks)

    predicted_map = _build_track_map(predicted_groups, common_tracks_set)
    reference_map = _build_track_map(reference_groups, common_tracks_set)

    tp = fp = fn = tn = 0
    for left, right in combinations(common_tracks, 2):
        predicted_same = predicted_map.get(left) == predicted_map.get(right)
        reference_same = reference_map.get(left) == reference_map.get(right)

        if predicted_same and reference_same:
            tp += 1
        elif predicted_same and not reference_same:
            fp += 1
        elif not predicted_same and reference_same:
            fn += 1
        else:
            tn += 1

    total_pairs = tp + fp + fn + tn
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    rand_accuracy = _safe_divide(tp + tn, total_pairs, default=1.0)

    predicted_common_groups = {
        frozenset(group & common_tracks_set)
        for group in predicted_groups
        if group & common_tracks_set
    }
    reference_common_groups = {
        frozenset(group & common_tracks_set)
        for group in reference_groups
        if group & common_tracks_set
    }
    exact_group_match = (
        len(predicted_common_groups & reference_common_groups) /
        len(reference_common_groups)
        if reference_common_groups
        else 1.0
    )
    cluster_purity_score = _compute_cluster_purity(
        predicted_groups,
        reference_map,
        common_tracks_set,
    )
    entity_continuity_score = _compute_entity_continuity(
        reference_groups,
        predicted_map,
        common_tracks_set,
    )
    temporal_coherence_score = _compute_temporal_coherence(
        predicted_groups,
        reference_map,
        track_time_lookup or {},
        common_tracks_set,
    )

    return {
        "predicted_tracks": len(predicted_tracks),
        "reference_tracks": len(reference_tracks),
        "common_tracks": len(common_tracks),
        "missing_in_prediction": sorted(reference_tracks - predicted_tracks),
        "extra_in_prediction": sorted(predicted_tracks - reference_tracks),
        "predicted_entities": len(predicted_groups),
        "reference_entities": len(reference_groups),
        "pairwise_true_positive": tp,
        "pairwise_false_positive": fp,
        "pairwise_false_negative": fn,
        "pairwise_true_negative": tn,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "rand_accuracy": rand_accuracy,
        "exact_group_match": exact_group_match,
        "temporal_coherence_score": temporal_coherence_score,
        "cluster_purity_score": cluster_purity_score,
        "entity_continuity_score": entity_continuity_score,
    }


def update_statistics_file(stats_path, input_path, reference_path, batch_id, metrics):
    """Обновляет отдельный файл статистики без вывода пользователю."""
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    run_record = {
        "timestamp": now,
        "batch_id": batch_id,
        "input_file": str(input_path),
        "reference_file": str(reference_path),
        "metrics": metrics,
    }

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = {
            "updated_at": now,
            "runs_count": 0,
            "last_run": None,
            "history": [],
        }

    stats["updated_at"] = now
    stats["runs_count"] = int(stats.get("runs_count", 0)) + 1
    stats["last_run"] = run_record
    history = stats.get("history", [])
    history.append(run_record)
    stats["history"] = history

    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# predicted_tracks
# Сколько треков алгоритм вообще выдал в результате.
# reference_tracks
# Сколько треков есть в эталонной разметке.
# common_tracks
# Сколько треков совпадает между прогнозом и эталоном. Именно по ним считается качество.
# missing_in_prediction
# Какие треки есть в эталоне, но алгоритм их не учёл.
# extra_in_prediction
# Какие треки есть у алгоритма, но их нет в эталоне.
# predicted_entities
# Сколько машин в итоге нашёл алгоритм.
# reference_entities
# Сколько машин должно быть по эталону.
# pairwise_true_positive
# Сколько пар треков алгоритм правильно объединил в одну машину.
# pairwise_false_positive
# Сколько пар треков алгоритм ошибочно склеил, хотя это разные машины.
# pairwise_false_negative
# Сколько пар треков алгоритм не объединил, хотя это одна и та же машина.
# pairwise_true_negative
# Сколько пар треков алгоритм правильно объединил в одну машину.
# pairwise_false_positive
# Сколько пар треков алгоритм ошибочно склеил, хотя это разные машины.
# pairwise_false_negative
# Сколько пар треков алгоритм не объединил, хотя это одна и та же машина.
# pairwise_true_negative
# Сколько пар треков алгоритм правильно оставил раздельно.
# pairwise_precision
# Насколько алгоритм аккуратен, когда объединяет треки.
# Если значение высокое, значит он редко склеивает разные машины.
# pairwise_recall
# Насколько алгоритм умеет находить все правильные объединения.
# Если значение высокое, значит он редко пропускает треки одной и той же машины.
# pairwise_f1
# Главная итоговая метрика качества.
# Она учитывает и аккуратность объединения, и полноту объединения. Чем выше, тем лучше.
# rand_accuracy
# Общая доля правильных решений по всем парам треков: где надо объединить и где надо разделить.
# exact_group_match
# Сколько машин алгоритм восстановил идеально, без единой ошибки в составе треков.
# temporal_coherence_score
# Насколько логично треки внутри одной найденной группы идут по времени.
# Если метрика высокая, значит внутри группы мало временных противоречий.
# cluster_purity_score
# Насколько “чистые” получаются группы.
# Если метрика высокая, значит в одной группе обычно лежат треки одной машины, а не смесь из разных.
# entity_continuity_score
# Насколько целостно алгоритм собирает одну реальную машину.
# Если метрика высокая, значит треки одной машины не распадаются на много отдельных групп.
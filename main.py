"""
Точка входа для запуска модуля обработки данных о движении транспорта.

Загружает JSON-файл, валидирует данные, вычисляет признаки и выполняет ReID.
"""

import json
import sys
import time
from pathlib import Path
from blockchain import LocalBlockchain
from schema import load_batch
from utils import compute_track_features
from reid import reidentify
from config import ReIDConfig
from metrics import evaluate_entities, extract_reference_groups, update_statistics_file
from json_cache_storage import JsonAuditCache
from postgres_storage import PostgresAuditStorage
from utils import normalize_time
from violations import detect_speed_violations


def print_batch_info(batch):
    """Вывод информации о загруженном пакете."""

    print("=" * 80)
    print("ИНФОРМАЦИЯ О ПАКЕТЕ")
    print("=" * 80)
    print(f"Batch ID: {batch.batch_id or 'не указан'}")
    print(f"Timestamp: {batch.timestamp or 'не указан'}")
    print(f"Количество треков: {len(batch.tracks)}")
    print()

    for i, track in enumerate(batch.tracks, 1):
        print(f"Трек #{i}: {track.track_id}")
        print(f"  Источник: {track.source_id}")
        print(f"  Время: {track.start_ts} - {track.end_ts}")
        print(f"  Номерной знак: {track.plate_text or 'не указан'}")
        print(f"  Цвет: {track.color or 'не указан'}")
        print(f"  Марка: {track.make or 'не указан'}")
        print(f"  Модель: {track.model or 'не указан'}")
        print(f"  Количество точек: {len(track.points)}")
        print()


def print_features(batch, track_features):
    """Вывод вычисленных признаков для всех треков."""
    print("=" * 80)
    print("ВЫЧИСЛЕННЫЕ ПРИЗНАКИ")
    print("=" * 80)

    for track in batch.tracks:
        features = track_features[track.track_id]
        print(f"Трек: {track.track_id}")
        print(
            f"  Средняя скорость: {features['avg_speed']:.2f} м/с" if features[
                'avg_speed'] else "  Средняя скорость: не указана")
        print(f"  Медианная скорость: {features['median_speed']:.2f} м/с" if
              features['median_speed'] else "  Медианная скорость: не указана")
        print(
            f"  Доминирующее направление: {features['dominant_heading']:.1f}°" if
            features[
                'dominant_heading'] else "  Доминирующее направление: не указано")
        print(f"  Длина пути: {features['path_length']:.1f} м")
        print(f"  Длительность: {features['duration_seconds']:.1f} с")
        print(
            f"  Перемещение: {features['displacement']['distance_m']:.1f} м" if
            features['displacement']['distance_m'] else "  Перемещение: 0 м")
        print(
            f"  Направление перемещения: {features['displacement']['heading_deg']:.1f}°" if
            features['displacement'][
                'heading_deg'] else "  Направление перемещения: не указано")
        bbox = features['bbox']
        print(
            f"  Bounding box: ({bbox['min_lat']:.6f}, {bbox['min_lon']:.6f}) - ({bbox['max_lat']:.6f}, {bbox['max_lon']:.6f})")
        print()


def print_reid_results(entities, links):
    """Вывод результатов ReID."""
    print("=" * 80)
    print("РЕЗУЛЬТАТЫ RE-IDENTIFICATION")
    print("=" * 80)

    print(f"Найдено сущностей: {len(entities)}")
    print()

    # Группировка связей по сущностям
    entity_links = {}
    for link in links:
        if link.entity_id not in entity_links:
            entity_links[link.entity_id] = []
        entity_links[link.entity_id].append(link)

    # Вывод информации о каждой сущности
    for entity in sorted(entities, key=lambda e: e.entity_id):
        print(f"Сущность: {entity.entity_id}")
        print(f"  Треки: {', '.join(sorted(entity.track_ids))}")

        # Вывод связей с объяснениями
        if entity.entity_id in entity_links:
            print("  Связи:")
            for link in sorted(entity_links[entity.entity_id],
                               key=lambda l: l.confidence, reverse=True):
                print(
                    f"    Трек {link.track_id}: confidence={link.confidence:.3f}")
                if link.reasons:
                    print(f"      Причины:")
                    for reason in link.reasons[
                        :5]:  # Показываем первые 5 причин
                        print(f"        - {reason}")
        print()

    # Статистика
    print("Статистика:")
    print(f"  Всего треков: {len(set(link.track_id for link in links))}")
    print(f"  Всего связей: {len(links)}")
    high_confidence_links = [l for l in links if l.confidence >= 0.75]
    print(f"  Связей с confidence >= 0.75: {len(high_confidence_links)}")
    print()


def print_violations(violations):
    """Вывод информации о нарушениях скорости."""
    print("=" * 80)
    print("НАРУШЕНИЯ")
    print("=" * 80)
    print(f"Ограничение скорости: {violations.get('speed_limit_kph')} км/ч")
    print(f"Найдено нарушений: {violations.get('count', 0)}")
    for item in violations.get("speeding", []):
        print(
            f"  track_id={item['track_id']}, max_speed={item['max_speed_kph']} км/ч, "
            f"source={item['source_id']}"
        )
    print()


def build_track_features(batch):
    """Вычисление признаков по всем трекам пакета."""
    return {
        track.track_id: compute_track_features(track)
        for track in batch.tracks
    }


def serialize_entities(entities):
    """Сериализация сущностей в детерминированный JSON-совместимый вид."""
    return [
        {
            "entity_id": entity.entity_id,
            "track_ids": sorted(entity.track_ids),
        }
        for entity in sorted(entities, key=lambda item: item.entity_id)
    ]


def serialize_links(links):
    """Сериализация связей в детерминированный JSON-совместимый вид."""
    return [
        {
            "entity_id": link.entity_id,
            "track_id": link.track_id,
            "confidence": float(link.confidence),
            "reasons": list(link.reasons),
        }
        for link in sorted(
            links,
            key=lambda item: (item.entity_id, item.track_id, -item.confidence),
        )
    ]


def print_cached_result(cached_record):
    """Вывод ранее сохранённого результата без повторного расчёта."""
    print("=" * 80)
    print("НАЙДЕН ГОТОВЫЙ РЕЗУЛЬТАТ В КЭШЕ (CACHE HIT)")
    print("=" * 80)
    metadata = cached_record.get("metadata") or {}
    reid_result = cached_record.get("reid_result") or {}
    print(f"Batch ID: {metadata.get('batch_id') or cached_record.get('batch_id')}")
    print(f"Количество треков: {metadata.get('num_tracks') or cached_record.get('num_tracks')}")
    if cached_record.get("id") is not None:
        print(f"DB row id: {cached_record.get('id')}")
    print(f"Input hash: {cached_record.get('input_hash')}")
    print(f"Найдено сущностей: {len(reid_result.get('entities', []))}")
    print(f"Найдено связей: {len(reid_result.get('links', []))}")
    violations = reid_result.get("violations")
    if violations:
        print(f"Найдено нарушений: {violations.get('count', 0)}")
    print()
    print(json.dumps(reid_result, ensure_ascii=False, indent=2))


def main():
    """Основная функция."""
    request_started_at = time.perf_counter()

    if len(sys.argv) < 2:
        print("Использование: python main.py <путь_к_json_файлу>")
        print("Пример: python main.py examples/sample_batch_extended.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Ошибка: файл {json_path} не найден")
        sys.exit(1)

    try:
        # Загрузка JSON
        print(f"Загрузка данных из {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        json_cache = JsonAuditCache.from_env()
        try:
            cached_json = json_cache.find_cached_record(input_payload=json_data)
            if cached_json:
                print_cached_result(cached_json)
                return
        except Exception as cache_exc:
            print(f" Предупреждение: не удалось проверить JSON-кэш: {cache_exc}")

        postgres_storage = PostgresAuditStorage.from_env()
        if postgres_storage.enabled:
            try:
                cached = postgres_storage.find_cached_record(input_payload=json_data)
                if cached:
                    print_cached_result(cached)
                    return
            except Exception as db_exc:
                print(f" Предупреждение: не удалось проверить кэш PostgreSQL: {db_exc}")

        # Валидация и загрузка пакета
        print("Валидация данных...")
        batch = load_batch(json_data)
        print(" Данные успешно загружены и валидированы")
        print()

        violations = detect_speed_violations(batch)
        batch.violations = violations
        print_violations(violations)

        # Вывод информации о пакете
        print_batch_info(batch)

        # Вычисление признаков
        print("Вычисление признаков...")
        track_features = build_track_features(batch)
        print_features(batch, track_features)

        # ReID
        print("Выполнение Re-identification...")
        config = ReIDConfig()
        entities, links = reidentify(batch, config)
        print(" ReID завершён")
        print()

        # Вывод результатов
        print_reid_results(entities, links)

        entities_out = serialize_entities(entities)
        track_time_lookup = {
            track.track_id: {
                "start": normalize_time(track.start_ts),
                "end": normalize_time(track.end_ts),
            }
            for track in batch.tracks
        }

        # Тихое обновление статистики качества по эталону, если он доступен
        reference_path = Path("result/result.json")
        if reference_path.exists():
            with open(reference_path, "r", encoding="utf-8") as f:
                reference_payload = json.load(f)
            evaluation = evaluate_entities(
                predicted_entities=entities_out,
                reference_entities=extract_reference_groups(reference_payload),
                track_time_lookup=track_time_lookup,
            )
            request_processing_seconds = time.perf_counter() - request_started_at
            evaluation["request_processing_time_seconds"] = round(
                request_processing_seconds, 6
            )
            evaluation["request_processing_time_ms"] = round(
                request_processing_seconds * 1000.0, 3
            )
            update_statistics_file(
                stats_path=Path("result/statistics.json"),
                input_path=json_path,
                reference_path=reference_path,
                batch_id=batch.batch_id,
                metrics=evaluation,
            )

        # Блокчейн-аудит
        print("Запись в локальный blockchain-журнал...")
        blockchain = LocalBlockchain()
        links_out = serialize_links(links)
        block = blockchain.add_audit_record(
            input_payload=json_data,
            track_features=track_features,
            reid_result={
                "entities": entities_out,
                "links": links_out,
                "violations": violations,
            },
            metadata={
                "batch_id": batch.batch_id,
                "num_tracks": len(batch.tracks),
                "source_file": str(json_path),
            },
        )
        is_valid, validation_message = blockchain.verify_chain()
        print(f" Blockchain-блок сохранён: index={block.index}, hash={block.block_hash}")
        print(f" Проверка цепочки: {validation_message}")
        if not is_valid:
            raise RuntimeError("После записи цепочка блоков не прошла проверку")

        try:
            cache_input_hash = json_cache.save_record(
                input_payload=json_data,
                track_features=track_features,
                reid_result={
                    "entities": entities_out,
                    "links": links_out,
                    "violations": violations,
                },
                metadata={
                    "batch_id": batch.batch_id,
                    "num_tracks": len(batch.tracks),
                    "source_file": str(json_path),
                },
                blockchain={
                    "block_index": block.index,
                    "block_hash": block.block_hash,
                    "previous_hash": block.previous_hash,
                    "merkle_root": block.merkle_root,
                    "difficulty": block.difficulty,
                    "chain_valid": is_valid,
                    "validation_message": validation_message,
                },
            )
            print(f" JSON-кэш обновлён: hash={cache_input_hash}")
        except Exception as cache_exc:
            print(f" Предупреждение: не удалось сохранить JSON-кэш: {cache_exc}")

        if postgres_storage.enabled:
            try:
                db_row_id = postgres_storage.save_audit_record(
                    input_payload=json_data,
                    track_features=track_features,
                    reid_result={
                        "entities": entities_out,
                        "links": links_out,
                        "violations": violations,
                    },
                    metadata={
                        "batch_id": batch.batch_id,
                        "num_tracks": len(batch.tracks),
                        "source_file": str(json_path),
                    },
                    blockchain={
                        "block_index": block.index,
                        "block_hash": block.block_hash,
                        "previous_hash": block.previous_hash,
                        "merkle_root": block.merkle_root,
                        "difficulty": block.difficulty,
                        "chain_valid": is_valid,
                        "validation_message": validation_message,
                    },
                )
                print(f" PostgreSQL-запись сохранена: id={db_row_id}")
            except Exception as db_exc:
                print(f" Предупреждение: не удалось сохранить в PostgreSQL: {db_exc}")

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

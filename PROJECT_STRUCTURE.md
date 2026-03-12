# Структура проекта

```
HardAutoLife/
  schema.py              # Модели данных (dataclasses) и валидация
  config.py              # Конфигурация ReID (пороги, веса)
  utils.py               # Утилиты: нормализация, вычисление признаков
  reid.py                # Алгоритм Re-identification
  main.py                # Точка входа для запуска примера
  README.md              # Описание проекта
  SCHEMA.md              # Описание JSON-схемы
  PROJECT_STRUCTURE.md   # Ты тут
  tests/
     __init__.py
     test_schema.py     # Тесты валидации и загрузки
     test_utils.py      # Тесты утилит и признаков
     test_reid.py       # Тесты ReID алгоритма
  examples/
      sample_batch_minimal.json    # Минимальный пример
      sample_batch_extended.json   # Расширенный пример
```

## Мдули

### schema.py
- `Point`: точка трека (ts, lat, lon, speed_mps, heading_deg)
- `Track`: трек (track_id, source_id, start_ts, end_ts, plate_text, color, make, model, points)
- `Batch`: пакет треков (batch_id, timestamp, tracks)
- `load_batch()`: загрузка и валидация из JSON/dict

### config.py
- `ReIDConfig`: конфигурация алгоритма ReID
  - Пороги для фильтров (color, make, source)
  - Веса для признаков траектории
  - Порог confidence для объединения

### utils.py
- `normalize_time()`: нормализация ISO 8601
- `sort_points_by_time()`: сортировка точек
- `compute_track_features()`: вычисление признаков трека
  - Средняя скорость, медиана скорости
  - Доминирующее направление
  - Длина пути
  - Bounding box
  - Типичная скорость
  - Вектор перемещения start->end

### reid.py
- `Entity`: сущность (entity_id)
- `Link`: связь трека с сущностью (entity_id, track_id, confidence, reasons)
- `reidentify()`: основной алгоритм ReID
  - Жёсткие фильтры (color, make, source)
  - Вычисление сходства траекторий
  - Кластеризация (greedy или union-find)

### main.py
- Загрузка JSON
- Валидация
- Вычисление признаков
- Запуск ReID
- Вывод результатов


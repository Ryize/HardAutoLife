# JSON-схема входных данных

## Структура Batch

```json
{
  "batch_id": "string (optional)",
  "timestamp": "ISO 8601 (optional)",
  "tracks": [
    {
      "track_id": "string (required)",
      "source_id": "string (required)",
      "start_ts": "ISO 8601 (required)",
      "end_ts": "ISO 8601 (required)",
      "plate_text": "string|null (optional)",
      "plate_confidence": "float|null (optional, 0..1)",
      "color": "string|null (optional)",
      "make": "string|null (optional)",
      "model": "string|null (optional)",
      "vehicle_type": "string|null (optional, car/truck/bus/motorcycle/van/suv/...)",
      "body_type": "string|null (optional, sedan/hatchback/coupe/... )",
      "size_class": "string|null (optional, small/medium/large)",
      "lane_id": "string|int|null (optional)",
      "relative_position": "float|null (optional, 0..1, поперёк дороги)",
      "points": [
        {
          "ts": "ISO 8601 (required)",
          "lat": "float (required, -90..90)",
          "lon": "float (required, -180..180)",
          "speed_mps": "float|null (optional, >= 0)",
          "heading_deg": "float|null (optional, 0..360)"
        }
      ]
    }
  ]
}
```

## Правила валидации

### Batch
- `batch_id`: опциональный идентификатор пакета
- `timestamp`: опциональная метка времени пакета (ISO 8601)
- `tracks`: массив треков (минимум 0 элементов)

### Track
- `track_id`: обязательное, уникальное в рамках пакета
- `source_id`: обязательное, идентификатор источника/зоны наблюдения
- `start_ts`, `end_ts`: обязательные, ISO 8601, `start_ts <= end_ts`
- `plate_text`: опциональное, может быть `null`, пустой строкой или `"-"` (все трактуются как «нет данных»)
- `plate_confidence`: опциональное, float в диапазоне [0, 1], доверие к распознаванию номера
- `color`: опциональное, строка (например, "red", "blue", "white", "black", "gray", "silver")
- `make`: опциональное, марка автомобиля (например, "Toyota", "BMW")
- `model`: опциональное, модель автомобиля
- `vehicle_type`: опциональное, тип ТС (`car`, `truck`, `bus`, `motorcycle`, `van`, `suv`, ...)
- `body_type`: опциональное, тип кузова (`sedan`, `hatchback`, `coupe`, `wagon`, ...)
- `size_class`: опциональное, класс размера (`small`, `medium`, `large`)
- `lane_id`: опциональное, идентификатор полосы/дорожного сегмента
- `relative_position`: опциональное, относительное положение поперёк дороги в диапазоне [0, 1]
- `points`: обязательный массив точек, минимум 1 точка

### Point
- `ts`: обязательное, ISO 8601, должно быть в диапазоне `[start_ts, end_ts]`
- `lat`: обязательное, float в диапазоне [-90, 90]
- `lon`: обязательное, float в диапазоне [-180, 180]
- `speed_mps`: опциональное, float >= 0 (метры в секунду), может быть `null`
- `heading_deg`: опциональное, float в диапазоне [0, 360), может быть `null`

### Обработка пропусков
- Если `speed_mps` отсутствует или `null` — считается неизвестной
- Если `heading_deg` отсутствует или `null` — считается неизвестным
- Для строковых полей (`plate_text`, `color`, `make`, `model`, `vehicle_type`, `body_type`, `size_class`, `lane_id`) значения `null`, `""`, `"-"` трактуются как «нет данных» и внутри системы приводятся к `None`
- `plate_confidence`, `relative_position` могут быть `null` и тогда не учитываются при ReID
- Точки автоматически сортируются по `ts` после загрузки

### ReID атрибуты
- **Обязательные для ReID**: нет (все косвенные/прямые признаки опциональны)
- **Сильные прямые признаки**: `plate_text` (+ `plate_confidence`), `color`, `make`, `model`
- **Дополнительные визуальные признаки**: `vehicle_type`, `body_type`, `size_class`
- **Поведенческие/контекстные признаки**: производные от `points` (скорость, направление, траектория, остановки), `lane_id`, `relative_position`, совместимость `source_id`


## Общая идея проекта

Проект реализует локальный модуль обработки подготовленных данных о движении транспорта с поддержкой:

- **Восстановления и анализа траекторий** по последовательности точек (lat/lon/время/скорость/направление).
- **Re-identification (ReID)** транспортных средств: связывание треков одного и того же автомобиля по косвенным признакам, даже при частично или полностью скрытом номерном знаке.
- **Криптожурнала (локальный permissioned-блокчейн)** для аудита и неизменяемой фиксации ключевых событий обработки (получение входных данных, выполнение ReID и т.п.).

Всё работает **локально**, без внешних сетей и БД. Вся криптография построена на стандартной библиотеке Python.

---

## Основные сценарии использования

1. **Анализ движения и ReID**
   - Загрузить заранее подготовленный JSON с треками.
   - Провести валидацию данных и нормализацию.
   - Вычислить производные признаки по траектории (скорость, длина пути, направление и т.д.).
   - Выполнить ReID: объединить треки в сущности (гипотезы автомобилей) на основе признаков.
   - Получить объяснимый вывод: какие треки попали в одну сущность и почему (reasons).

2. **Аудит и блокчейн-журнал**
   - Зафиксировать факт получения батча (`batch_received`) в блокчейне.
   - Зафиксировать факт выполнения ReID (`reid_done`) с хэшами входных данных, результата и конфигурации.
   - Смайнить блок с транзакциями аудита (Proof-of-Work).
   - Проверить целостность цепочки и, при необходимости, выгрузить криптографическое доказательство (anchor) для конкретного события.

---

## Структура проекта (модули и ответственность)

### Модуль данных и признаков

- **`schema.py`**
  - Dataclasses:
    - `Point`: точка трека (`ts`, `lat`, `lon`, `speed_mps`, `heading_deg`).
    - `Track`: трек (`track_id`, `source_id`, `start_ts`, `end_ts`, `plate_text`, `plate_confidence`, `color`, `make`, `model`, `vehicle_type`, `body_type`, `size_class`, `lane_id`, `relative_position`, `points`).
    - `Batch`: пакет треков (`batch_id`, `timestamp`, `tracks`).
  - Функция `load_batch(json_str_or_dict) -> Batch`:
    - Принимает JSON-строку или словарь.
    - Валидирует типы и диапазоны:
      - координаты (`lat`, `lon`);
      - время (`start_ts <= end_ts`, точки внутри диапазона трека);
      - скорость (`speed_mps >= 0`), направление (`0 <= heading_deg < 360`);
      - `plate_confidence` и `relative_position` в диапазоне \[0, 1].
    - Нормализует строковые поля: `null`, `""`, `"-"` → `None`.
    - Проверяет уникальность `track_id` в пределах пакета.

- **`utils.py`**
  - Вспомогательные функции:
    - `normalize_time(ts) -> datetime` — ISO 8601 → `datetime`.
    - `sort_points_by_time(points)` — сортировка точек по времени.
    - `haversine_distance(lat1, lon1, lat2, lon2)` — геодезическое расстояние.
  - Вычисление признаков трека:
    - `compute_track_features(track) -> dict`:
      - средняя и медианная скорость;
      - доминирующее направление;
      - длина пути;
      - bounding box по координатам;
      - типичная скорость;
      - вектор перемещения (расстояние и направление start→end);
      - длительность трека.
  - Временные отношения:
    - `time_overlap(track1, track2)`;
    - `time_gap(track1, track2)`.

### Модуль Re-identification

- **`config.py`**
  - Dataclass `ReIDConfig`:
    - веса признаков (скорость, направление, длина пути, географическая близость, временной gap и др.);
    - штрафы/бонусы за:
      - цвет (`color_match_boost`, `color_mismatch_penalty`),
      - марку (`make_match_boost`, `make_mismatch_penalty`),
      - тип ТС, тип кузова, класс размера,
      - номерной знак (`plate_*`),
      - общий порог `confidence_threshold` для объединения треков в одну сущность;
    - функции нормализации различий (скорость, расстояние, угол, время).

- **`reid.py`**
  - Dataclasses:
    - `Entity`: сущность (гипотеза автомобиля) с `entity_id` и множеством `track_ids`.
    - `Link`: связь `entity_id` ↔ `track_id` с `confidence` и списком `reasons`.
  - Алгоритм:
    1. Для каждого трека считаются признаки (`compute_track_features`).
    2. Все пары треков сравниваются:
       - Жёсткие/полужёсткие фильтры:
         - цвет, марка, тип ТС, тип кузова, размер;
         - номерной знак с учётом `plate_confidence`;
         - источник (`source_id`).
       - Сходство траектории:
         - скорость, направление, длина пути;
         - вектор перемещения;
         - временной gap/overlap;
         - географическая близость.
       - Итоговый скор → прогон через сигмоиду → `confidence ∈ [0, 1]`.
       - В `reasons` записываются текстовые объяснения (diff, similarity).
    3. Кластеризация:
       - greedy / union-find: треки с `confidence >= threshold` объединяются в одну сущность.
  - Результат:
    - список `Entity`;
    - список `Link` с подробными объяснениями.

### Модуль криптожурнала и блокчейна

- **`ledger/models.py`**
  - `Transaction`: аудиторская транзакция:
    - `tx_id`, `ts`, `event`, `payload`, `tx_hash`.
    - `payload` — только хэши и метаданные, без «тяжёлых» данных.
  - `Block`: блок цепочки:
    - `index`, `ts`, `prev_hash`, `merkle_root`, `difficulty`, `nonce`, `validator_id`, `block_hash`, `signature`, `transactions`.

- **`ledger/hash.py`**
  - `canonical_json(obj)` — канонический JSON (sort_keys, компактные сепараторы).
  - `sha256_json(obj)` — SHA-256 от канонического JSON.
  - `hash_transaction(tx)` — хэш транзакции без `tx_hash`.
  - `hash_block_header(block)` — хэш заголовка блока.
  - `hmac_sha256(key, data)` — подпись блока (HMAC-SHA256).

- **`ledger/merkle.py`**
  - `build_merkle_root(hashes)` — Merkle-root для хэшей транзакций.
  - `build_merkle_proof(hashes, index)` — Merkle-proof для конкретной транзакции (для экспорта доказательства).

- **`ledger/chain.py`**
  - `BlockchainConfig`:
    - `validator_id`, `validator_key`, `difficulty` (количество ведущих нулей в `block_hash`).
  - `Blockchain`:
    - Хранилище:
      - `ledger/data/chain.jsonl` — цепочка блоков (каждая строка — JSON блока).
      - `ledger/data/mempool.jsonl` — неподтверждённые транзакции.
    - Методы:
      - `init_chain()` — создаёт genesis-блок при отсутствии цепочки.
      - `add_transaction(tx)` — кладёт транзакцию в mempool.
      - `mine_block()` — собирает транзакции из mempool, ищет `nonce` (PoW), подписывает и добавляет блок в цепочку.
      - `verify_chain()` — проверка:
        - связности `prev_hash`;
        - Merkle-root;
        - корректности `block_hash`;
        - выполнения PoW (`block_hash` начинается с N нулей);
        - подписи валидатора (HMAC-SHA256).
      - `find_transaction(tx_id)` — поиск транзакции и блока по `tx_id`.

- **`ledger/cli.py` и `ledger/__main__.py`**
  - CLI-интерфейс (запуск через `python -m ledger`):
    - `init` — создать genesis-блок и пустой mempool.
    - `add_tx` — добавить транзакцию аудита в mempool:
      - `--event` — тип события (`reid_done`, `batch_received`, ...);
      - `--input` — путь к входному JSON, хэш пишется в `payload.input_hash`;
      - `--output` — путь к выходному JSON, хэш пишется в `payload.output_hash`;
      - `--meta` — дополнительный JSON со служебными полями (например, `batch_id`).
    - `mine` — смайнить блок из транзакций mempool.
    - `verify` — проверить целостность цепочки.
    - `export --anchor <tx_id>` — вывести доказательство для транзакции:
      - `block_index`, `block_hash`, `tx_hash`, `merkle_root`, `merkle_proof`.

### Модуль аудита

- **`audit.py`**
  - Детерминированное хэширование:
    - `hash_batch(batch)` — хэш всего `Batch`.
    - `hash_reid_output(entities, links)` — хэш результата ReID (с сортировкой сущностей и связей).
    - `hash_reid_config(config)` — хэш конфигурации ReID.
  - Запись аудиторских транзакций:
    - `record_batch_received(batch) -> tx_id`
      - создаёт транзакцию `batch_received` с `input_hash` и `batch_id`;
      - добавляет её в mempool блокчейна;
      - возвращает `tx_id` (anchor).
    - `record_reid_done(batch, entities, links, config) -> tx_id`
      - создаёт транзакцию `reid_done` с `input_hash`, `output_hash`, `params_hash`, `batch_id`;
      - добавляет в mempool;
      - возвращает `tx_id`.

---

## Основной сценарий работы (end-to-end)

### 1. Запуск анализа и ReID

Команда:

```bash
python main.py examples/sample_batch_extended.json
```

Шаги внутри `main.py`:

1. **Загрузка JSON**:
   - файл читается из `examples/`;
   - данные передаются в `load_batch()`.

2. **Валидация и нормализация** (модуль `schema`):
   - проверяются все поля треков и точек;
   - строки `"", "-"` и `null` приводятся к `None`;
   - проверяется временная согласованность (`start_ts`, `end_ts`, точки).

3. **Запись аудита `batch_received`** (модуль `audit` + `ledger`):
   - считается хэш всего `Batch`;
   - формируется транзакция `batch_received` с `batch_id` и `input_hash`;
   - транзакция добавляется в mempool блокчейна;
   - на экран выводится `tx_id`.

4. **Вычисление признаков траекторий** (модуль `utils`):
   - для каждого трека считается набор статистик движения.

5. **ReID** (модуль `reid`):
   - попарное сравнение треков с учётом визуальных и поведенческих признаков;
   - расчёт `confidence` для каждой пары;
   - greedy кластеризация в сущности.
   - Результат: список `Entity` и `Link`, печать итоговых сущностей и причин объединения.

6. **Запись аудита `reid_done`**:
   - считается `input_hash` (как для `batch_received`);
   - считается `output_hash` (по `entities` и `links`);
   - считается `params_hash` (по `ReIDConfig`);
   - создаётся транзакция `reid_done` и добавляется в mempool;
   - выводится её `tx_id`.

7. **Майнинг блока**:
   - из mempool берутся все транзакции (как минимум `batch_received` и `reid_done`);
   - формируется новый блок:
     - считается Merkle-root транзакций;
     - перебором `nonce` подбирается `block_hash` с заданным количеством ведущих нулей (difficulty);
     - считаются HMAC-подпись `signature`.
   - блок дописывается в `ledger/data/chain.jsonl`;
   - mempool очищается.

8. **Проверка блокчейна**:
   - последовательно читаются все блоки;
   - проверяется связность, Merkle-root, PoW и подписи;
   - на экран выводится результат верификации.

---

## Примеры CLI для блокчейна

### Инициализация

```bash
python -m ledger init
```

Создаёт genesis-блок и файлы `chain.jsonl` и `mempool.jsonl` в `ledger/data/`.

### Добавить произвольную транзакцию аудита

```bash
python -m ledger add_tx \
  --event reid_done \
  --input examples/sample_batch_extended.json \
  --output examples/sample_batch_extended.json \
  --meta '{"batch_id":"batch_002"}'
```

### Майнинг блока

```bash
python -m ledger mine
```

### Проверка целостности цепочки

```bash
python -m ledger verify
```

### Экспорт доказательства для конкретного события

```bash
python -m ledger export --anchor <tx_id>
```

Вернёт JSON с:

- `block_index`, `block_hash`;
- `tx_hash`;
- `merkle_root`, `merkle_proof`.

---

## Тесты и воспроизводимость

- Все основные компоненты покрыты unit-тестами (`tests/`):
  - валидация схемы и загрузка данных;
  - вычисление признаков;
  - алгоритм ReID;
  - корректность блокчейн-цепочки (цепочка ломается при изменении старого блока, PoW, Merkle-root);
  - детерминизм хэшей аудита.
- Запуск тестов:

```bash
python -m unittest discover tests -v
```

Благодаря детерминированной сериализации (canonical JSON) и фиксированным правилам нормализации один и тот же вход всегда даёт один и тот же результат ReID и один и тот же набор хэшей в криптожурнале.


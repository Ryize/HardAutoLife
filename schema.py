"""
Модуль определения схемы данных и валидации входных данных.

Использует dataclasses из стандартной библиотеки Python для минимизации зависимостей.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
import json


@dataclass
class Point:
    """Точка трека с координатами, временем и опциональными параметрами движения."""
    ts: str  # ISO 8601
    lat: float
    lon: float
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None

    def __post_init__(self):
        """Валидация точки после создания."""
        # Валидация координат
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"lat должен быть в диапазоне [-90, 90], получено: {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"lon должен быть в диапазоне [-180, 180], получено: {self.lon}")
        
        # Валидация скорости
        if self.speed_mps is not None and self.speed_mps < 0:
            raise ValueError(f"speed_mps должен быть >= 0, получено: {self.speed_mps}")
        
        # Валидация направления
        if self.heading_deg is not None:
            if not (0 <= self.heading_deg < 360):
                raise ValueError(f"heading_deg должен быть в диапазоне [0, 360), получено: {self.heading_deg}")
        
        # Валидация времени (базовая проверка формата ISO 8601)
        try:
            datetime.fromisoformat(self.ts.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f"ts должен быть в формате ISO 8601, получено: {self.ts}")


@dataclass
class Track:
    """Трек движения транспортного средства."""
    track_id: str
    source_id: str
    start_ts: str  # ISO 8601
    end_ts: str  # ISO 8601
    points: List[Point]
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None  # 0..1
    color: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    vehicle_type: Optional[str] = None  # car/truck/bus/...
    body_type: Optional[str] = None  # sedan/hatchback/...
    size_class: Optional[str] = None  # small/medium/large
    lane_id: Optional[str] = None
    relative_position: Optional[float] = None  # 0..1

    def __post_init__(self):
        """Валидация трека после создания."""
        # Проверка наличия точек
        if not self.points:
            raise ValueError(f"Трек {self.track_id} должен содержать хотя бы одну точку")
        
        # Проверка временных меток
        try:
            start_dt = datetime.fromisoformat(self.start_ts.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(self.end_ts.replace('Z', '+00:00'))
        except ValueError as e:
            raise ValueError(f"Неверный формат времени в треке {self.track_id}: {e}")
        
        if start_dt > end_dt:
            raise ValueError(
                f"start_ts ({self.start_ts}) должен быть <= end_ts ({self.end_ts}) "
                f"для трека {self.track_id}"
            )
        
        # Проверка, что все точки находятся в временном диапазоне трека
        for point in self.points:
            try:
                point_dt = datetime.fromisoformat(point.ts.replace('Z', '+00:00'))
            except ValueError as e:
                raise ValueError(f"Неверный формат времени в точке трека {self.track_id}: {e}")
            
            if not (start_dt <= point_dt <= end_dt):
                raise ValueError(
                    f"Точка с ts={point.ts} находится вне диапазона "
                    f"[{self.start_ts}, {self.end_ts}] для трека {self.track_id}"
                )
        
        # Нормализация строковых полей:
        # пустая строка или "-" -> None
        def _clean_optional_str(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            s = str(value).strip()
            if s == "" or s == "-":
                return None
            return s

        self.plate_text = _clean_optional_str(self.plate_text)
        self.color = _clean_optional_str(self.color)
        self.make = _clean_optional_str(self.make)
        self.model = _clean_optional_str(self.model)
        self.vehicle_type = _clean_optional_str(self.vehicle_type)
        self.body_type = _clean_optional_str(self.body_type)
        self.size_class = _clean_optional_str(self.size_class)
        self.lane_id = _clean_optional_str(self.lane_id)

        # Валидация plate_confidence
        if self.plate_confidence is not None:
            if not (0.0 <= self.plate_confidence <= 1.0):
                raise ValueError(
                    f"plate_confidence должен быть в диапазоне [0, 1], "
                    f"получено: {self.plate_confidence}"
                )

        # Валидация relative_position
        if self.relative_position is not None:
            if not (0.0 <= self.relative_position <= 1.0):
                raise ValueError(
                    f"relative_position должен быть в диапазоне [0, 1], "
                    f"получено: {self.relative_position}"
                )


@dataclass
class Batch:
    """Пакет треков для обработки."""
    tracks: List[Track]
    batch_id: Optional[str] = None
    timestamp: Optional[str] = None
    violations: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Валидация пакета после создания."""
        # Проверка уникальности track_id в пакете
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            duplicates = [tid for tid in track_ids if track_ids.count(tid) > 1]
            raise ValueError(f"Обнаружены дубликаты track_id: {set(duplicates)}")


def parse_point(data: Dict[str, Any]) -> Point:
    """Парсинг точки из словаря."""
    return Point(
        ts=data['ts'],
        lat=float(data['lat']),
        lon=float(data['lon']),
        speed_mps=float(data['speed_mps']) if data.get('speed_mps') is not None else None,
        heading_deg=float(data['heading_deg']) if data.get('heading_deg') is not None else None,
    )


def parse_track(data: Dict[str, Any]) -> Track:
    """Парсинг трека из словаря."""
    points = [parse_point(p) for p in data['points']]
    
    return Track(
        track_id=str(data['track_id']),
        source_id=str(data['source_id']),
        start_ts=str(data['start_ts']),
        end_ts=str(data['end_ts']),
        points=points,
        plate_text=data.get('plate_text'),
        plate_confidence=float(data['plate_confidence']) if 'plate_confidence' in data and data['plate_confidence'] is not None else None,
        color=data.get('color'),
        make=data.get('make'),
        model=data.get('model'),
        vehicle_type=data.get('vehicle_type'),
        body_type=data.get('body_type'),
        size_class=data.get('size_class'),
        lane_id=str(data['lane_id']) if 'lane_id' in data and data['lane_id'] is not None else None,
        relative_position=float(data['relative_position']) if 'relative_position' in data and data['relative_position'] is not None else None,
    )


def load_batch(data: Union[str, Dict[str, Any]]) -> Batch:
    """
    Загрузка и валидация пакета данных.
    
    Args:
        data: JSON-строка или словарь с данными пакета
        
    Returns:
        Batch: Валидированный объект пакета
        
    Raises:
        ValueError: При ошибках валидации
        json.JSONDecodeError: При ошибках парсинга JSON
    """
    # Парсинг JSON, если передан строкой
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}")
    
    # Проверка структуры
    if not isinstance(data, dict):
        raise ValueError("Данные должны быть словарём или JSON-строкой")
    
    if 'tracks' not in data:
        raise ValueError("Отсутствует обязательное поле 'tracks'")
    
    if not isinstance(data['tracks'], list):
        raise ValueError("Поле 'tracks' должно быть списком")
    
    # Парсинг треков
    tracks = []
    for i, track_data in enumerate(data['tracks']):
        try:
            track = parse_track(track_data)
            tracks.append(track)
        except Exception as e:
            raise ValueError(f"Ошибка парсинга трека #{i}: {e}")
    
    # Создание Batch
    batch = Batch(
        tracks=tracks,
        batch_id=data.get('batch_id'),
        timestamp=data.get('timestamp'),
        violations=data.get('violations'),
    )
    
    return batch

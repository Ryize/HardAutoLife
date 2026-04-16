from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from blockchain import LocalBlockchain
from config import ReIDConfig
from json_cache_storage import JsonAuditCache, compute_input_hash
from postgres_storage import PostgresAuditStorage
from reid import reidentify
from schema import load_batch
from utils import compute_track_features
from violations import detect_speed_violations, detect_speed_violations_from_payload


app = FastAPI(title="HardAutoLife ReID API", version="1.0.0")


@app.get("/health", summary="Проверка живости сервиса")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/audit/chain", summary="Аудит целостности всей blockchain-цепочки")
async def audit_chain() -> Dict[str, Any]:
    blockchain = LocalBlockchain()
    chain_valid, validation_message = blockchain.verify_chain()
    chain = blockchain.load_chain()
    last_block = chain[-1] if chain else None

    return {
        "chain_valid": chain_valid,
        "validation_message": validation_message,
        "chain_length": len(chain),
        "last_block": (
            {
                "index": last_block.index,
                "block_hash": last_block.block_hash,
                "previous_hash": last_block.previous_hash,
                "timestamp": last_block.timestamp,
                "difficulty": last_block.difficulty,
                "transactions_count": len(last_block.transactions),
            }
            if last_block is not None
            else None
        ),
    }


@app.post("/reid", summary="Выполнить ReID по Batch-данным")
async def reid_endpoint(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Вход: Batch в том же формате, что и для load_batch().

    Выход:
        {
          "batch_id": "...",
          "num_tracks": <int>,
          "entities": [
            {"entity_id": "...", "track_ids": ["...", "..."]}
          ],
          "links": [
            {
              "entity_id": "...",
              "track_id": "...",
              "confidence": <float>,
              "reasons": ["...", ...]
            }
          ],
          "track_features": {
            "<track_id>": { ... признаки ... }
          }
        }
    """
    json_cache = JsonAuditCache.from_env()
    try:
        cached_json = json_cache.find_cached_record(input_payload=batch)
    except Exception:
        cached_json = None
    if cached_json:
        cached_metadata = cached_json.get("metadata") or {}
        cached_reid = cached_json.get("reid_result") or {}
        violations = cached_reid.get("violations")
        if violations is None:
            try:
                violations = detect_speed_violations_from_payload(
                    cached_json.get("input_payload") or batch
                )
            except Exception:
                violations = {"speed_limit_kph": 60.0, "count": 0, "speeding": []}
        return {
            "batch_id": cached_metadata.get("batch_id") or cached_json.get("batch_id"),
            "num_tracks": cached_metadata.get("num_tracks") or cached_json.get("num_tracks"),
            "entities": cached_reid.get("entities", []),
            "links": cached_reid.get("links", []),
            "violations": violations,
            "track_features": cached_json.get("track_features") or {},
            "blockchain": cached_json.get("blockchain"),
            "cache": {
                "source": "json",
                "hit": True,
                "input_hash": cached_json.get("input_hash"),
            },
            "postgres": {
                "enabled": bool(PostgresAuditStorage.from_env().enabled),
                "saved": False,
                "cache_hit": False,
            },
        }

    postgres_storage = PostgresAuditStorage.from_env()
    if postgres_storage.enabled:
        try:
            cached = postgres_storage.find_cached_record(input_payload=batch)
        except Exception:
            cached = None
        if cached:
            cached_metadata = cached.get("metadata") or {}
            cached_reid = cached.get("reid_result") or {}
            violations = cached_reid.get("violations")
            if violations is None:
                try:
                    violations = detect_speed_violations_from_payload(
                        cached.get("input_payload") or batch
                    )
                except Exception:
                    violations = {"speed_limit_kph": 60.0, "count": 0, "speeding": []}
            return {
                "batch_id": cached_metadata.get("batch_id") or cached.get("batch_id"),
                "num_tracks": cached_metadata.get("num_tracks") or cached.get("num_tracks"),
                "entities": cached_reid.get("entities", []),
                "links": cached_reid.get("links", []),
                "violations": violations,
                "track_features": cached.get("track_features") or {},
                "blockchain": cached.get("blockchain"),
                "cache": {
                    "source": "postgres",
                    "hit": True,
                    "input_hash": cached.get("input_hash"),
                },
                "postgres": {
                    "enabled": True,
                    "saved": False,
                    "cache_hit": True,
                    "row_id": cached.get("id"),
                    "input_hash": cached.get("input_hash"),
                },
            }

    try:
        batch_obj = load_batch(batch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    violations = detect_speed_violations(batch_obj)
    batch_obj.violations = violations

    config = ReIDConfig()
    entities, links = reidentify(batch_obj, config)

    # Признаки по трекам
    features: Dict[str, Any] = {
        track.track_id: compute_track_features(track) for track in batch_obj.tracks
    }

    # Сериализация сущностей и связей в простой JSON-формат
    entities_out = [
        {
            "entity_id": e.entity_id,
            "track_ids": sorted(list(e.track_ids)),
        }
        for e in entities
    ]

    links_out = [
        {
            "entity_id": l.entity_id,
            "track_id": l.track_id,
            "confidence": float(l.confidence),
            "reasons": list(l.reasons),
        }
        for l in links
    ]

    blockchain = LocalBlockchain()
    block = blockchain.add_audit_record(
        input_payload=batch,
        track_features=features,
        reid_result={
            "entities": entities_out,
            "links": links_out,
            "violations": violations,
        },
        metadata={
            "batch_id": batch_obj.batch_id,
            "num_tracks": len(batch_obj.tracks),
            "source": "api:/reid",
        },
    )
    chain_valid, chain_message = blockchain.verify_chain()

    json_cache_info: Dict[str, Any] = {
        "source": "json",
        "hit": False,
    }
    try:
        cache_input_hash = json_cache.save_record(
            input_payload=batch,
            track_features=features,
            reid_result={
                "entities": entities_out,
                "links": links_out,
                "violations": violations,
            },
            metadata={
                "batch_id": batch_obj.batch_id,
                "num_tracks": len(batch_obj.tracks),
                "source": "api:/reid",
            },
            blockchain={
                "block_index": block.index,
                "block_hash": block.block_hash,
                "previous_hash": block.previous_hash,
                "merkle_root": block.merkle_root,
                "difficulty": block.difficulty,
                "chain_valid": chain_valid,
                "validation_message": chain_message,
            },
        )
        json_cache_info["input_hash"] = cache_input_hash
        json_cache_info["saved"] = True
    except Exception as cache_exc:
        json_cache_info["saved"] = False
        json_cache_info["error"] = str(cache_exc)

    db_result: Dict[str, Any] = {
        "enabled": False,
        "saved": False,
    }

    db_result["cache_hit"] = False
    if postgres_storage.enabled:
        db_result["enabled"] = True
        try:
            db_row_id = postgres_storage.save_audit_record(
                input_payload=batch,
                track_features=features,
                reid_result={
                    "entities": entities_out,
                    "links": links_out,
                    "violations": violations,
                },
                metadata={
                    "batch_id": batch_obj.batch_id,
                    "num_tracks": len(batch_obj.tracks),
                    "source": "api:/reid",
                },
                blockchain={
                    "block_index": block.index,
                    "block_hash": block.block_hash,
                    "previous_hash": block.previous_hash,
                    "merkle_root": block.merkle_root,
                    "difficulty": block.difficulty,
                    "chain_valid": chain_valid,
                    "validation_message": chain_message,
                },
            )
            db_result["saved"] = True
            db_result["row_id"] = db_row_id
            db_result["input_hash"] = compute_input_hash(batch)
        except Exception as db_exc:
            db_result["error"] = str(db_exc)

    return {
        "batch_id": batch_obj.batch_id,
        "num_tracks": len(batch_obj.tracks),
        "entities": entities_out,
        "links": links_out,
        "violations": violations,
        "track_features": features,
        "blockchain": {
            "block_index": block.index,
            "block_hash": block.block_hash,
            "previous_hash": block.previous_hash,
            "merkle_root": block.merkle_root,
            "difficulty": block.difficulty,
            "chain_valid": chain_valid,
            "validation_message": chain_message,
        },
        "cache": json_cache_info,
        "postgres": db_result,
    }

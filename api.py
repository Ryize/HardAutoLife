from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from blockchain import LocalBlockchain
from config import ReIDConfig
from reid import reidentify
from schema import load_batch
from utils import compute_track_features


app = FastAPI(title="HardAutoLife ReID API", version="1.0.0")


@app.get("/health", summary="Проверка живости сервиса")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


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
    try:
        batch_obj = load_batch(batch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        },
        metadata={
            "batch_id": batch_obj.batch_id,
            "num_tracks": len(batch_obj.tracks),
            "source": "api:/reid",
        },
    )
    chain_valid, chain_message = blockchain.verify_chain()

    return {
        "batch_id": batch_obj.batch_id,
        "num_tracks": len(batch_obj.tracks),
        "entities": entities_out,
        "links": links_out,
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
    }

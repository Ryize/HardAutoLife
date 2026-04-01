"""
Локальный permissioned blockchain-модуль для аудита обработки данных.

Особенности:
- хранение блоков в директории с append-only файлами;
- proof-of-work с настраиваемой сложностью;
- merkle root для набора транзакций в блоке;
- полная верификация целостности транзакций и цепочки блоков.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import hmac
import json
import time
import uuid


GENESIS_PREVIOUS_HASH = "0" * 64


def canonical_json(data: Any) -> str:
    """JSON-сериализация."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: str) -> str:
    """SHA-256 хэш строки."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_payload(data: Any) -> str:
    """Хэш произвольного JSON-совместимого объекта."""
    return sha256_hex(canonical_json(data))


def compute_merkle_root(transaction_hashes: List[str]) -> str:
    """Вычисление merkle root по списку хэшей транзакций."""
    if not transaction_hashes:
        return sha256_hex("[]")

    level = list(transaction_hashes)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])

        next_level: List[str] = []
        for index in range(0, len(level), 2):
            combined = level[index] + level[index + 1]
            next_level.append(sha256_hex(combined))
        level = next_level

    return level[0]


@dataclass
class Transaction:
    """Транзакция аудита или служебная транзакция."""

    tx_id: str
    tx_type: str
    timestamp: float
    payload: Dict[str, Any]
    payload_hash: str
    tx_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "tx_hash": self.tx_hash,
        }

    @classmethod
    def create(
        cls,
        tx_type: str,
        payload: Dict[str, Any],
        timestamp: Optional[float] = None,
        tx_id: Optional[str] = None,
    ) -> "Transaction":
        tx_timestamp = time.time() if timestamp is None else float(timestamp)
        transaction_id = tx_id or str(uuid.uuid4())
        payload_hash = hash_payload(payload)
        tx_hash = hash_payload(
            {
                "tx_id": transaction_id,
                "tx_type": tx_type,
                "timestamp": tx_timestamp,
                "payload_hash": payload_hash,
            }
        )
        return cls(
            tx_id=transaction_id,
            tx_type=tx_type,
            timestamp=tx_timestamp,
            payload=payload,
            payload_hash=payload_hash,
            tx_hash=tx_hash,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        return cls(
            tx_id=data["tx_id"],
            tx_type=data["tx_type"],
            timestamp=float(data["timestamp"]),
            payload=data["payload"],
            payload_hash=data["payload_hash"],
            tx_hash=data["tx_hash"],
        )


@dataclass
class Block:
    """Блок локального блокчейна."""

    index: int
    timestamp: float
    previous_hash: str
    difficulty: int
    nonce: int
    merkle_root: str
    transactions: List[Transaction]
    block_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "difficulty": self.difficulty,
            "nonce": self.nonce,
            "merkle_root": self.merkle_root,
            "transactions": [transaction.to_dict() for transaction in self.transactions],
            "block_hash": self.block_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        return cls(
            index=int(data["index"]),
            timestamp=float(data["timestamp"]),
            previous_hash=data["previous_hash"],
            difficulty=int(data["difficulty"]),
            nonce=int(data["nonce"]),
            merkle_root=data["merkle_root"],
            transactions=[Transaction.from_dict(item) for item in data["transactions"]],
            block_hash=data["block_hash"],
        )


class LocalBlockchain:
    """Файловая реализация локального blockchain-журнала."""

    def __init__(
        self,
        storage_dir: Path | str = "blockchain_data",
        difficulty: int = 3,
        chain_id: str = "hardautolife-audit-chain",
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.blocks_dir = self.storage_dir / "blocks"
        self.difficulty = int(difficulty)
        self.chain_id = chain_id
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_genesis_block()

    def _ensure_genesis_block(self) -> None:
        if any(self.blocks_dir.glob("*.json")):
            return

        genesis_transaction = Transaction.create(
            tx_type="genesis",
            timestamp=0.0,
            tx_id="genesis-tx",
            payload={
                "chain_id": self.chain_id,
                "description": "Локальный blockchain-журнал аудита обработки",
                "proof_algorithm": "pow",
            },
        )
        genesis_block = self._mine_block(
            index=0,
            previous_hash=GENESIS_PREVIOUS_HASH,
            transactions=[genesis_transaction],
            timestamp=0.0,
            difficulty=self.difficulty,
        )
        self._write_block(genesis_block)

    def _block_path(self, index: int) -> Path:
        return self.blocks_dir / f"{index:08d}.json"

    def _block_files(self) -> List[Path]:
        return sorted(self.blocks_dir.glob("*.json"))

    def _write_block(self, block: Block) -> None:
        block_path = self._block_path(block.index)
        if block_path.exists():
            raise ValueError(f"Блок {block.index} уже существует: {block_path}")
        block_path.write_text(canonical_json(block.to_dict()) + "\n", encoding="utf-8")

    def load_chain(self) -> List[Block]:
        chain: List[Block] = []
        for block_file in self._block_files():
            block_data = json.loads(block_file.read_text(encoding="utf-8"))
            chain.append(Block.from_dict(block_data))
        return chain

    def get_last_block(self) -> Block:
        chain = self.load_chain()
        if not chain:
            raise ValueError("Цепочка блоков пуста")
        return chain[-1]

    def _calculate_block_hash(
        self,
        index: int,
        timestamp: float,
        previous_hash: str,
        difficulty: int,
        nonce: int,
        merkle_root: str,
        transactions: List[Transaction],
    ) -> str:
        block_core = {
            "index": index,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
            "difficulty": difficulty,
            "nonce": nonce,
            "merkle_root": merkle_root,
            "transactions": [transaction.to_dict() for transaction in transactions],
        }
        return hash_payload(block_core)

    def _mine_block(
        self,
        index: int,
        previous_hash: str,
        transactions: List[Transaction],
        timestamp: Optional[float] = None,
        difficulty: Optional[int] = None,
    ) -> Block:
        block_timestamp = time.time() if timestamp is None else float(timestamp)
        block_difficulty = self.difficulty if difficulty is None else int(difficulty)
        merkle_root = compute_merkle_root([transaction.tx_hash for transaction in transactions])
        target_prefix = "0" * block_difficulty
        nonce = 0

        while True:
            block_hash = self._calculate_block_hash(
                index=index,
                timestamp=block_timestamp,
                previous_hash=previous_hash,
                difficulty=block_difficulty,
                nonce=nonce,
                merkle_root=merkle_root,
                transactions=transactions,
            )
            if block_hash.startswith(target_prefix):
                return Block(
                    index=index,
                    timestamp=block_timestamp,
                    previous_hash=previous_hash,
                    difficulty=block_difficulty,
                    nonce=nonce,
                    merkle_root=merkle_root,
                    transactions=transactions,
                    block_hash=block_hash,
                )
            nonce += 1

    def add_block(
        self,
        transactions: List[Transaction],
        timestamp: Optional[float] = None,
    ) -> Block:
        if not transactions:
            raise ValueError("Нельзя создать блок без транзакций")

        is_valid, message = self.verify_chain()
        if not is_valid:
            raise ValueError(f"Цепочка повреждена: {message}")

        last_block = self.get_last_block()
        new_block = self._mine_block(
            index=last_block.index + 1,
            previous_hash=last_block.block_hash,
            transactions=transactions,
            timestamp=timestamp,
        )
        self._write_block(new_block)
        return new_block

    def add_audit_record(
        self,
        input_payload: Dict[str, Any],
        track_features: Dict[str, Any],
        reid_result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Block:
        transaction = create_audit_transaction(
            input_payload=input_payload,
            track_features=track_features,
            reid_result=reid_result,
            metadata=metadata,
            timestamp=timestamp,
        )
        return self.add_block([transaction], timestamp=timestamp)

    def verify_chain(self) -> Tuple[bool, str]:
        chain = self.load_chain()
        if not chain:
            return False, "цепочка не содержит блоков"

        expected_previous_hash = GENESIS_PREVIOUS_HASH
        for index, block in enumerate(chain):
            if block.index != index:
                return False, f"некорректный индекс блока {block.index}"

            if block.previous_hash != expected_previous_hash:
                return False, f"нарушена ссылка previous_hash в блоке {block.index}"

            transaction_hashes: List[str] = []
            for transaction in block.transactions:
                expected_payload_hash = hash_payload(transaction.payload)
                if not hmac.compare_digest(transaction.payload_hash, expected_payload_hash):
                    return False, f"payload_hash транзакции {transaction.tx_id} не совпадает"

                expected_tx_hash = hash_payload(
                    {
                        "tx_id": transaction.tx_id,
                        "tx_type": transaction.tx_type,
                        "timestamp": transaction.timestamp,
                        "payload_hash": transaction.payload_hash,
                    }
                )
                if not hmac.compare_digest(transaction.tx_hash, expected_tx_hash):
                    return False, f"tx_hash транзакции {transaction.tx_id} не совпадает"
                transaction_hashes.append(transaction.tx_hash)

            expected_merkle_root = compute_merkle_root(transaction_hashes)
            if not hmac.compare_digest(block.merkle_root, expected_merkle_root):
                return False, f"merkle_root блока {block.index} не совпадает"

            expected_block_hash = self._calculate_block_hash(
                index=block.index,
                timestamp=block.timestamp,
                previous_hash=block.previous_hash,
                difficulty=block.difficulty,
                nonce=block.nonce,
                merkle_root=block.merkle_root,
                transactions=block.transactions,
            )
            if not hmac.compare_digest(block.block_hash, expected_block_hash):
                return False, f"block_hash блока {block.index} не совпадает"

            if not block.block_hash.startswith("0" * block.difficulty):
                return False, f"proof-of-work блока {block.index} не проходит по difficulty"

            expected_previous_hash = block.block_hash

        return True, "цепочка валидна"


def create_audit_transaction(
    input_payload: Dict[str, Any],
    track_features: Dict[str, Any],
    reid_result: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> Transaction:
    """Создание audit-транзакции, связывающей вход и результат через хэши."""
    metadata = metadata or {}
    payload = {
        "input_hash": hash_payload(input_payload),
        "track_features_hash": hash_payload(track_features),
        "reid_result_hash": hash_payload(reid_result),
        "metadata": metadata,
        "summary": {
            "batch_id": metadata.get("batch_id"),
            "num_tracks": metadata.get("num_tracks"),
            "num_entities": len(reid_result.get("entities", [])),
            "num_links": len(reid_result.get("links", [])),
        },
    }
    return Transaction.create(tx_type="audit", payload=payload, timestamp=timestamp)

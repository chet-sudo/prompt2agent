"""Short term memory buffer implementation."""
from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List

from prompt2agent.utils.logging import get_logger

logger = get_logger(__name__)


class ShortTermMemoryBuffer:
    """In-memory ring buffer per agent."""

    def __init__(self, *, max_items: int = 10) -> None:
        self.max_items = max_items
        self._buffer: Deque[str] = deque(maxlen=max_items)

    def add(self, item: str) -> None:
        logger.debug("Memory add: %s", item)
        self._buffer.append(item)

    def extend(self, items: Iterable[str]) -> None:
        for item in items:
            self.add(item)

    def dump(self) -> List[str]:
        return list(self._buffer)

    def clear(self) -> None:
        logger.debug("Clearing memory buffer")
        self._buffer.clear()

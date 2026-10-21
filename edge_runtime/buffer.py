"""Bounded queues; factory retains FIFO order, guidance profiles retain fresh frames."""
from collections import deque
from threading import Condition
from typing import Any


class FrameBuffer:
    def __init__(self, capacity: int = 4, policy: str = "latest"):
        if policy not in {"fifo", "latest"}:
            raise ValueError("Unknown buffer policy")
        self.capacity = max(1, capacity)
        self.policy = policy
        self._items: deque[Any] = deque()
        self._condition = Condition()
        self.closed = False

    def put(self, packet: Any) -> list[Any]:
        dropped = []
        with self._condition:
            if self.closed:
                return [packet]
            if self.policy == "latest":
                dropped.extend(self._items)
                self._items.clear()
            elif len(self._items) >= self.capacity:
                # Preserve queued triggered work. Explicitly report rejected new work.
                return [packet]
            self._items.append(packet)
            self._condition.notify()
        return dropped

    def get(self, timeout: float = .2) -> Any | None:
        with self._condition:
            self._condition.wait_for(lambda: self._items or self.closed, timeout)
            return self._items.popleft() if self._items else None

    def close(self) -> list[Any]:
        with self._condition:
            self.closed = True
            pending = list(self._items)
            self._items.clear()
            self._condition.notify_all()
            return pending

    def __len__(self):
        with self._condition:
            return len(self._items)

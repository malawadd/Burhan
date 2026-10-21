"""Bounded asynchronous evidence writer; submit performs no filesystem I/O."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sqlite3
import threading
import uuid


class EventStore:
    MAX_PAYLOAD_BYTES = 256 * 1024
    MAX_JPEG_BYTES = 5 * 1024 * 1024
    MAX_QUEUE_BYTES = 16 * 1024 * 1024
    CLOSE_TIMEOUT_SECONDS = 5.0

    def __init__(self, root: Path, max_events=1000, max_media_bytes=50*1024*1024):
        if not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if not isinstance(max_media_bytes, int) or max_media_bytes < 0:
            raise ValueError("max_media_bytes must be a nonnegative integer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.media = self.root / "evidence"
        self.media.mkdir(exist_ok=True)
        self.db = self.root / "events.sqlite3"
        self.max_events, self.max_media_bytes = max_events, max_media_bytes
        self.queue: queue.Queue = queue.Queue(maxsize=64)
        self.dropped = self.written = 0
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._closed = False
        self._pending_bytes = self._outstanding = self._in_flight = 0
        self._flush_timed_out = False
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, profile TEXT NOT NULL, session_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, media_name TEXT)")
            conn.execute("CREATE INDEX IF NOT EXISTS events_created ON events(created_at DESC)")
        self._maintain()

    @contextmanager
    def connect(self):
        """Commit/rollback AND close; sqlite's own context manager does not close."""
        conn = sqlite3.connect(self.db, timeout=2)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _start_locked(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="edge-events", daemon=True)
            self._thread.start()

    def start(self):
        with self._lock:
            if not self._closed:
                self._start_locked()

    def _reject(self, message):
        with self._lock:
            self.dropped += 1
            self.error = message
        return False

    def submit(self, profile, session_id, kind, payload, jpeg=None) -> bool:
        with self._lock:
            if self._closed:
                self.dropped += 1
                self.error = "Event store is closed"
                return False
        try:
            for value in (profile, session_id, kind):
                if not isinstance(value, str) or len(value.encode("utf-8")) > 512:
                    raise ValueError("Event identifiers must be strings of at most 512 UTF-8 bytes")
            encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
            payload_size = len(encoded.encode("utf-8"))
            if payload_size > self.MAX_PAYLOAD_BYTES:
                raise ValueError("Event payload exceeds byte limit")
            if jpeg is not None and not isinstance(jpeg, (bytes, bytearray, memoryview)):
                raise ValueError("JPEG evidence must be bytes")
            media_size = jpeg.nbytes if isinstance(jpeg, memoryview) else len(jpeg or b"")
            if media_size > min(self.MAX_JPEG_BYTES, self.max_media_bytes):
                raise ValueError("JPEG evidence exceeds per-item or media-retention byte limit")
            immutable_jpeg = bytes(jpeg) if jpeg is not None else None
            event = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                     "profile": profile, "session_id": session_id, "kind": kind, "payload": encoded}
            byte_size = payload_size + media_size + sum(len(value.encode("utf-8")) for value in (profile, session_id, kind)) + 128
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            return self._reject(f"Rejected event: {exc}")
        with self._lock:
            if self._closed:
                self.dropped += 1
                self.error = "Event store is closed"
                return False
            if self._pending_bytes + byte_size > self.MAX_QUEUE_BYTES:
                self.dropped += 1
                self.error = "Event queue byte limit reached"
                return False
            try:
                self.queue.put_nowait((event, immutable_jpeg, byte_size))
            except queue.Full:
                self.dropped += 1
                self.error = "Event queue item limit reached"
                return False
            self._pending_bytes += byte_size
            self._outstanding += 1
        return True

    def _run(self):
        while not self._stop.is_set() or not self.queue.empty():
            try:
                event, jpeg, byte_size = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            with self._lock:
                self._in_flight += 1
            try:
                self._persist(event, jpeg)
                with self._lock:
                    self.written += 1
                    self.error = None
                try:
                    self._maintain()
                except Exception as exc:
                    with self._lock:
                        self.error = f"Retention cleanup failed: {type(exc).__name__}: {exc}"
            except Exception as exc:
                with self._lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.dropped += 1
                try:
                    self._maintain()
                except Exception as cleanup_exc:
                    with self._lock:
                        self.error += f"; cleanup failed: {cleanup_exc}"
            finally:
                with self._lock:
                    self._pending_bytes -= byte_size
                    self._outstanding -= 1
                    self._in_flight -= 1
                self.queue.task_done()

    def _persist(self, event, jpeg):
        media_name = event["id"] + ".jpg" if jpeg else None
        media_path = self.media / media_name if media_name else None
        temporary = self.media / (event["id"] + ".tmp") if media_name else None
        committed = False
        try:
            if media_path is not None:
                self._prune_media(reserve_bytes=len(jpeg))
                temporary.write_bytes(jpeg)
                temporary.replace(media_path)
            with self.connect() as conn:
                conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (
                    event["id"], event["created_at"], event["profile"], event["session_id"],
                    event["kind"], event["payload"], media_name))
            committed = True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if media_path is not None and not committed:
                media_path.unlink(missing_ok=True)

    def _media_files(self):
        return sorted(self.media.glob("*.jpg"), key=lambda path: (path.stat().st_mtime_ns, path.name))

    def _prune_media(self, reserve_bytes=0):
        files = self._media_files()
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total + reserve_bytes <= self.max_media_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
            with self.connect() as conn:
                conn.execute("UPDATE events SET media_name=NULL WHERE media_name=?", (path.name,))

    def _maintain(self):
        """Startup/after-write reconciliation; metadata retention survives restart."""
        with self.connect() as conn:
            conn.execute("DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY created_at DESC,id DESC LIMIT -1 OFFSET ?)", (self.max_events,))
            referenced = {row[0] for row in conn.execute("SELECT media_name FROM events WHERE media_name IS NOT NULL")}
        for temporary in self.media.glob("*.tmp"):
            temporary.unlink(missing_ok=True)
        for path in self.media.glob("*.jpg"):
            if path.name not in referenced:
                path.unlink(missing_ok=True)
        with self.connect() as conn:
            for name in referenced:
                if Path(name).name != name or not (self.media / name).is_file():
                    conn.execute("UPDATE events SET media_name=NULL WHERE media_name=?", (name,))
        self._prune_media()

    def list(self, limit=50, profile=None):
        limit = max(0, min(int(limit), self.max_events))
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM events " + ("WHERE profile=? " if profile else "") + "ORDER BY created_at DESC,id DESC LIMIT ?", ((profile, limit) if profile else (limit,))).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            results.append(item)
        return results

    def status(self):
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {"pending": self._outstanding, "queued": self.queue.qsize(), "in_flight": self._in_flight,
                    "pending_bytes": self._pending_bytes, "written": self.written, "lost_events": self.dropped,
                    "error": self.error, "max_events": self.max_events,
                    "media_limit_mb": self.max_media_bytes//1024//1024,
                    "closed": self._closed, "writer_alive": alive,
                    "flushed": self._closed and self._outstanding == 0 and not alive,
                    "flush_timed_out": self._flush_timed_out and alive,
                    "payload_byte_limit": self.MAX_PAYLOAD_BYTES,
                    "queue_byte_limit": self.MAX_QUEUE_BYTES}

    def close(self):
        with self._lock:
            self._closed = True
            self._stop.set()
            if self._outstanding:
                self._start_locked()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self.CLOSE_TIMEOUT_SECONDS)
        with self._lock:
            self._flush_timed_out = thread is not None and thread.is_alive()
        return self.status()

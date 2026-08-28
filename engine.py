# engine.py
"""
Simplified Event-Driven Engine for the APT tool.
- Lightweight message passing
- Resource management with external manifest
- Error handling
"""

import collections
import logging
from traceback import format_exc

logger = logging.getLogger(__name__)


class EventDispatcher:
    """Simple pub/sub style dispatcher using message_type as primary key."""

    def __init__(self):
        self._handlers = {}  # message_type -> list of callbacks
        self._message_queue = collections.deque()

    def register(self, message_type: str, callback):
        """Register a handler for a specific message type."""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(callback)

    def notify(self, message: dict):
        """Queue a message and process the queue."""
        if not isinstance(message, dict) or "message_type" not in message:
            logger.error("Invalid message format: missing 'message_type'")
            return

        self._message_queue.append(message)
        self._process_queue()

    def _enqueue_result(self, result):
        """
        Insert handler results at the *front* of the queue so causal
        follow-ups (e.g. store_data after precip_analysis) run before
        later siblings already sitting in the queue (e.g. generate_pdf).
        Lists are reversed so relative order is preserved under appendleft.
        """
        if isinstance(result, dict):
            if "message_type" in result:
                self._message_queue.appendleft(result)
        elif isinstance(result, list):
            for item in reversed(result):
                if isinstance(item, dict) and "message_type" in item:
                    self._message_queue.appendleft(item)

    def _process_queue(self):
        """Process all messages in the queue."""
        while self._message_queue:
            current = self._message_queue.popleft()
            msg_type = current["message_type"]

            # Internal fan-out: expand packed follow-up lists without handlers.
            # Nested _followups inside the list are rejected to avoid loops.
            if msg_type == "_followups":
                messages = current.get("messages") or []
                for m in messages:
                    if not isinstance(m, dict) or "message_type" not in m:
                        continue
                    nested_type = m["message_type"]
                    if nested_type.startswith("_"):
                        logger.error(
                            f"Refusing nested internal message_type "
                            f"'{nested_type}' inside _followups"
                        )
                        continue
                    self._message_queue.append(m)
                continue

            handlers = self._handlers.get(msg_type, [])
            for handler in handlers:
                try:
                    result = handler(current)
                    self._enqueue_result(result)
                except Exception as e:
                    logger.error(f"Handler error for {msg_type}: {e}", exc_info=True)
                    self._queue_error(current, handler, e)

    def _queue_error(self, original_msg: dict, handler, exception):
        """Clear queue to prevent cascade failures after an error."""
        error_msg = {
            "original_message": original_msg,
            "error": str(exception),
            "traceback": format_exc(),
        }
        self._message_queue.clear()
        logger.error(error_msg)

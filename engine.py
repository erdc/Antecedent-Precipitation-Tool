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

    def _process_queue(self):
        """Process all messages in the queue."""
        while self._message_queue:
            current = self._message_queue.popleft()
            msg_type = current["message_type"]

            handlers = self._handlers.get(msg_type, [])
            for handler in handlers:
                try:
                    result = handler(current)
                    if isinstance(result, dict):
                        self._message_queue.append(result)
                except Exception as e:
                    logger.error(f"Handler error for {msg_type}: {e}", exc_info=True)
                    self._queue_error(current, handler, e)

    def _queue_error(self, original_msg: dict, handler, exception):
        """Queue error message and clear queue to prevent cascade failures."""
        error_msg = {
            "original_message": original_msg,
            "error": str(exception),
            "traceback": format_exc(),
        }
        self._message_queue.clear()  # Prevent further processing after error
        logger.error(error_msg)

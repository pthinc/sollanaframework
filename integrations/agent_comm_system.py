# agent_comm_system.py
"""Lightweight agent communication system (threaded event bus)."""

import threading
import time
from typing import Callable, Dict, List, Any

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Dict[str,Any]], None]]] = {}
        self.lock = threading.Lock()

    def subscribe(self, event_type: str, handler: Callable[[Dict[str,Any]], None]):
        with self.lock:
            self.subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, payload: Dict[str,Any]):
        handlers = self.subscribers.get(event_type, [])
        for handler in handlers:
            threading.Thread(target=handler, args=(payload,), daemon=True).start()

class ContextManager:
    def __init__(self):
        self.context: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def set(self, key: str, value: Any):
        with self.lock:
            self.context[key] = value

    def get(self, key: str) -> Any:
        with self.lock:
            return self.context.get(key)

    def update(self, updates: Dict[str,Any]):
        with self.lock:
            self.context.update(updates)

    def snapshot(self) -> Dict[str,Any]:
        with self.lock:
            return dict(self.context)

class AgentCommSystem:
    def __init__(self):
        self.event_bus = EventBus()
        self.context = ContextManager()
        self.agents: Dict[str, Callable[[Dict[str,Any]], None]] = {}

    def register_agent(self, name: str, handler: Callable[[Dict[str,Any]], None]):
        self.agents[name] = handler
        self.event_bus.subscribe(f"msg_to_{name}", handler)

    def send(self, to: str, message: Dict[str,Any]):
        self.event_bus.publish(f"msg_to_{to}", message)

    def broadcast(self, message: Dict[str,Any]):
        for name in self.agents:
            self.send(name, message)

    def update_context(self, updates: Dict[str,Any]):
        self.context.update(updates)

    def get_context(self) -> Dict[str,Any]:
        return self.context.snapshot()

# Örnek kullanım
if __name__ == "__main__":
    system = AgentCommSystem()

    def hipofiz_handler(msg):
        print(f"[Hipofiz] mesaj alındı: {msg}")
        system.update_context({"hipofiz_last": msg})

    def dengecik_handler(msg):
        print(f"[Dengecik] mesaj alındı: {msg}")
        system.update_context({"dengecik_last": msg})

    system.register_agent("hipofiz", hipofiz_handler)
    system.register_agent("dengecik", dengecik_handler)

    system.send("hipofiz", {"type":"telemetry", "H":0.86})
    system.send("dengecik", {"type":"status", "D":0.61, "C":0.47, "S":0.29})

    time.sleep(0.5)
    print("📦 Bağlam snapshot:", system.get_context())

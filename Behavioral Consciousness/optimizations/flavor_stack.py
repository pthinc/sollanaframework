# flavor_stack.py
class FlavorCollisionError(Exception):
    pass

class FlavorStack:
    def __init__(self, max_depth: int = 32):
        self.stack = []
        self.max_depth = max_depth
        self.counts = {}

    def push(self, pattern: str):
        if self.counts.get(pattern, 0) > 0:
            raise FlavorCollisionError(f"Flavor collision: {pattern}")
        if len(self.stack) >= self.max_depth:
            raise RuntimeError("FlavorStack overflow")
        self.stack.append(pattern)
        self.counts[pattern] = self.counts.get(pattern, 0) + 1

    def pop(self):
        if not self.stack:
            return None
        p = self.stack.pop()
        self.counts[p] -= 1
        if self.counts[p] == 0:
            del self.counts[p]
        return p

    def count(self, pattern: str) -> int:
        return self.counts.get(pattern, 0)

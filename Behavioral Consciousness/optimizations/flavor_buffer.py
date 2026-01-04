# flavor_buffer.py
class FlavorBuffer:
    def __init__(self, approval_threshold: float = 0.5):
        self.buffer = []
        self.approval_threshold = approval_threshold

    def inject(self, flavor_item: Dict[str, Any], approval: float):
        if approval < self.approval_threshold:
            return self.fallback("gentle_suggestion")
        self.buffer.append(flavor_item)
        return {"status":"accepted"}

    def fallback(self, tag: str):
        # produce gentle suggestion skeleton
        return {"status":"fallback", "tag": tag, "payload": {"text": "Nazik bir öneri: ..."}}

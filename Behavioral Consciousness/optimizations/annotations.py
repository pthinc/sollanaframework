# annotations.py
def BehaviorTag(tag: str):
    def deco(fn):
        setattr(fn, "_behavior_tag", tag)
        return fn
    return deco

def Flavor(name: str):
    def deco(fn):
        setattr(fn, "_flavor", name)
        return fn
    return deco

# usage
@BehaviorTag("seek_support")
@Flavor("gentle_suggestion")
def support_user(*args, **kwargs):
    """Example stub behavior used for annotation discovery."""
    return {"status": "ok", "tag": "gentle_suggestion"}

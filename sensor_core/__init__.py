__all__ = ["SensorManager"]

def __getattr__(name):
    if name == "SensorManager":
        from .sensor_manager import SensorManager
        return SensorManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


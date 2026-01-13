__all__ = ["PlotManager"]

def __getattr__(name):
    if name == "PlotManager":
        from .plot_manager import PlotManager
        return PlotManager
    raise AttributeError(name)

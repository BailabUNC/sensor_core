"""Packaging: the compiled extension ships inside the package, and imports stay lightweight."""
import os
import subprocess
import sys


def test_compiled_ring_extension_ships_inside_the_package():
    import sensor_core
    import sensor_core._fastring as extension

    assert os.path.dirname(extension.__file__) == os.path.dirname(sensor_core.__file__)


def test_importing_sensor_core_does_not_load_the_plotting_stack(tmp_path):
    # fastplotlib needs a GPU-capable environment, so it must only load when a figure is created.
    code = (
        "import sys\n"
        "from sensor_core import SensorManager\n"
        "from sensor_core.plot import PlotManager\n"
        "print('fastplotlib' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"

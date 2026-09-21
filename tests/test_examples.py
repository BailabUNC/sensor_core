"""The examples run as documented."""
import os
import pathlib
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
if not EXAMPLES.is_dir():
    pytest.skip("the examples directory is not available", allow_module_level=True)
NOTEBOOKS = sorted(path.name for path in EXAMPLES.glob("*.ipynb"))


@pytest.mark.processes
def test_the_headless_example_stores_every_sample(tmp_path):
    result = subprocess.run([sys.executable, str(EXAMPLES / "virtual_serial_port_line.py"),
                             "--seconds", "2", "--database", str(tmp_path / "db.sqlite3")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Every sample stored, in order: True" in result.stdout


@pytest.mark.skipif(os.environ.get("SENSOR_CORE_RENDER_TESTS") != "1",
                    reason="set SENSOR_CORE_RENDER_TESTS=1 to run the notebooks, which render figures")
@pytest.mark.processes
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_example_notebook_runs(name, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()

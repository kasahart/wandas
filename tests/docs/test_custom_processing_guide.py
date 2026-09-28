"""Execute the user-facing extension path, including its serialized artifact."""

import re
import subprocess
import sys
from pathlib import Path


def test_custom_processing_guide_runs_in_fresh_process(tmp_path: Path) -> None:
    guide = Path(__file__).resolve().parents[2] / "docs/src/how-to/custom-processing.md"
    blocks = re.findall(r"```python\n(.*?)\n```", guide.read_text(), flags=re.S)
    assert blocks, "The custom processing guide must contain runnable examples"
    script = "\n\n".join(blocks)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "gain.recipe.json").is_file()

    # A Recipe transports intent, so a fresh process imports the extension again.
    extension = next(block for block in blocks if "class ProjectFrame" in block)
    (tmp_path / "my_processing.py").write_text(extension)
    replay = subprocess.run(
        [
            sys.executable,
            "-c",
            "from my_processing import np, ProjectFrame, RecipePlan, registry\n"
            "loaded = RecipePlan.load('gain.recipe.json', registry=registry)\n"
            "source = ProjectFrame.from_numpy(np.ones((2, 3)), sampling_rate=8000)\n"
            "result = loaded.apply({'signal': source}, registry=registry)\n"
            "np.testing.assert_allclose(result.to_numpy(), np.full((2, 3), 2.0))\n",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr

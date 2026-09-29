"""Execute the user-facing extension path, including its serialized artifact."""

import re
import subprocess
import sys
from pathlib import Path


def test_custom_processing_guide_runs_in_fresh_process(tmp_path: Path) -> None:
    guide = Path(__file__).resolve().parents[2] / "docs/src/how-to/custom-processing.md"
    blocks = re.findall(r"```python\n(.*?)\n```", guide.read_text(encoding="utf-8"), flags=re.S)
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
    (tmp_path / "my_processing.py").write_text(extension, encoding="utf-8")
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

    boundaries = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import dask.array as da
from my_processing import np, Gain, ProjectFrame, RecipePlan, registry
from wandas.pipeline.errors import RecipeSerializationError
from wandas.processing.semantic import freeze_params, value_to_json

for dtype in (np.int16, np.float32, np.float64):
    samples = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    lazy = Gain(8000, factor=0.5).process(da.from_array(samples))
    assert lazy.dtype == np.dtype(np.float64)
    values = lazy.compute()
    assert values.dtype == lazy.dtype
    np.testing.assert_allclose(values, [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
    frame = ProjectFrame.from_numpy(samples, sampling_rate=8000)
    processed = frame.gain(0.5)
    plan = RecipePlan.from_frame(processed, input_names=('signal',), registry=registry)
    restored = RecipePlan.from_dict(plan.to_dict(), registry=registry)
    np.testing.assert_allclose(restored.apply({'signal': frame}, registry=registry).to_numpy(), values)

for params in ({'factor': 2.0, 'unknown': 1}, {}, {'factor': True}, {'factor': '2'}):
    payload = plan.to_dict()
    payload['nodes'][0]['params'] = value_to_json(freeze_params(params))
    try:
        RecipePlan.from_dict(payload, registry=registry)
    except RecipeSerializationError:
        pass
    else:
        raise AssertionError(f'Invalid gain parameters accepted: {params}')
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert boundaries.returncode == 0, boundaries.stdout + boundaries.stderr

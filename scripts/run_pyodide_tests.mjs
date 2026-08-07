import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

function parseArguments(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Invalid arguments: ${argv.join(" ")}`);
    }
    parsed[key.slice(2)] = value;
  }
  return parsed;
}

function writeVirtualFile(pyodide, source, destination) {
  pyodide.FS.mkdirTree(path.posix.dirname(destination));
  pyodide.FS.writeFile(destination, fs.readFileSync(source));
}

function copyTree(pyodide, source, destination) {
  const stat = fs.statSync(source);
  if (stat.isDirectory()) {
    pyodide.FS.mkdirTree(destination);
    for (const entry of fs.readdirSync(source)) {
      if (entry === "__pycache__" || entry.endsWith(".pyc")) {
        continue;
      }
      copyTree(pyodide, path.join(source, entry), path.posix.join(destination, entry));
    }
    return;
  }
  writeVirtualFile(pyodide, source, destination);
}

const options = parseArguments(process.argv.slice(2));
const mode = options.mode;
const repositoryRoot = path.resolve(options["repository-root"]);
const runtimeDir = path.resolve(options["runtime-dir"]);
const expectedPyodideVersion = options["expected-pyodide-version"];
const wheelModes = new Set(["wheel-guide-smoke", "source-tests"]);
const guideModes = new Set(["wheel-guide-smoke", "published-guide-smoke"]);
if (!["html-smoke", ...wheelModes, "published-guide-smoke"].includes(mode)) {
  throw new Error(`Unsupported Pyodide harness mode: ${mode}`);
}
if (wheelModes.has(mode) && !options.wheel) {
  throw new Error(`${mode} requires --wheel PATH`);
}
if (mode === "source-tests" && !options["expected-wandas-version"]) {
  throw new Error("source-tests requires an expected Wandas version");
}
if (mode === "html-smoke") {
  const expectedWandasVersion = options["expected-wandas-version"];
  if (!expectedWandasVersion) {
    throw new Error("html-smoke requires an expected Wandas version");
  }

  const browserExample = fs.readFileSync(
    path.join(repositoryRoot, "examples", "pyodide", "index.html"),
    "utf8",
  );
  const requiredFragments = [
    `https://cdn.jsdelivr.net/pyodide/v${expectedPyodideVersion}/full/pyodide.js`,
    `const PYODIDE_VERSION = "${expectedPyodideVersion}";`,
    `const WANDAS_VERSION = "${expectedWandasVersion}";`,
    'await pyodide.loadPackage("micropip")',
    'await micropip.install([',
    'import wandas as wd',
    "wd.from_numpy(",
    "low_pass_filter(cutoff=1_000)",
    "wd.read(",
    "pyfetch(",
    'type="file"',
    "CORS",
    "URL.revokeObjectURL(",
  ];
  const missingFragments = requiredFragments.filter(
    (fragment) => !browserExample.includes(fragment),
  );
  if (missingFragments.length > 0) {
    throw new Error(
      `Browser example is missing required fragments: ${missingFragments.join(", ")}`,
    );
  }
  console.log(
    "Browser example source consistency: required Pyodide, Wandas, WAV, and browser fragments found",
  );
  process.exitCode = 0;
  process.exit();
}
const lockedRequirements = fs
  .readFileSync(path.join(repositoryRoot, "scripts", "pyodide", "requirements.txt"), "utf8")
  .split("\n")
  .map((line) => line.trim())
  .filter((line) => line && !line.startsWith("#"));

const pyodideModulePath = path.join(runtimeDir, "node_modules", "pyodide", "pyodide.mjs");
const pyodideModule = await import(pathToFileURL(pyodideModulePath).href);
if (pyodideModule.version !== expectedPyodideVersion) {
  throw new Error(
    `Loaded Pyodide ${pyodideModule.version}, expected ${expectedPyodideVersion}`,
  );
}

const pyodide = await pyodideModule.loadPyodide();
console.log(`Runtime: Pyodide ${pyodideModule.version} / ${pyodide.runPython("import sys; sys.version")}`);

await pyodide.loadPackage("micropip");
pyodide.globals.set("wandas_locked_requirements_json", JSON.stringify(lockedRequirements));

let localWheelUri;
if (wheelModes.has(mode)) {
  const wheelPath = path.resolve(options.wheel);
  const virtualWheel = `/work/dist/${path.basename(wheelPath)}`;
  writeVirtualFile(pyodide, wheelPath, virtualWheel);
  localWheelUri = `emfs:${virtualWheel}`;
}

if (guideModes.has(mode)) {
  const wandasInstallSpec =
    mode === "wheel-guide-smoke" ? localWheelUri : options["wandas-install-spec"];
  const expectedWandasVersion = options["expected-wandas-version"];
  if (!wandasInstallSpec || !expectedWandasVersion) {
    throw new Error(`${mode} requires a Wandas install spec and expected version`);
  }

  pyodide.globals.set("wandas_install_spec", wandasInstallSpec);
  pyodide.globals.set("expected_wandas_version", expectedWandasVersion);
  await pyodide.runPythonAsync(`
import importlib.metadata
import json
from io import BytesIO

import micropip

await micropip.install(
    [*json.loads(wandas_locked_requirements_json), wandas_install_spec]
)

import matplotlib.pyplot as plt
import numpy as np
import soundfile
import wandas as wd

actual_version = importlib.metadata.version("wandas")
if actual_version != expected_wandas_version:
    raise RuntimeError(
        f"Installed Wandas {actual_version}, expected {expected_wandas_version}"
    )

sampling_rate = 8_000
time = np.arange(sampling_rate, dtype=np.float64) / sampling_rate
samples = (
    0.6 * np.sin(2 * np.pi * 440 * time)
    + 0.2 * np.sin(2 * np.pi * 1_800 * time)
)
original = wd.from_numpy(
    samples,
    sampling_rate=sampling_rate,
    label="browser guide install smoke",
    ch_labels=["mono"],
)
filtered = original.low_pass_filter(cutoff=1_000)
filtered_values = filtered.to_numpy()
if (
    filtered.sampling_rate != sampling_rate
    or filtered.n_channels != 1
    or filtered_values.shape != samples.shape
    or not np.isfinite(filtered_values).all()
    or np.allclose(filtered_values, samples)
):
    raise RuntimeError("Wandas artifact failed the browser guide smoke")

figure, axis = plt.subplots(figsize=(4, 2))
original.plot(ax=axis, color="0.65", label="original")
filtered.plot(ax=axis, color="C0", label="1 kHz low-pass")
png = BytesIO()
figure.savefig(png, format="png")
plt.close(figure)
if not png.getvalue().startswith(b"\\x89PNG\\r\\n\\x1a\\n"):
    raise RuntimeError("Wandas artifact failed to render a PNG")

generated_wav_path = "/tmp/wandas-guide-smoke.wav"
filtered.to_wav(generated_wav_path)
round_trip = wd.read(generated_wav_path)
if (
    round_trip.sampling_rate != sampling_rate
    or round_trip.n_channels != 1
    or round_trip.to_numpy().shape != samples.shape
):
    raise RuntimeError("Wandas artifact failed the WAV round trip")

print(
    "Browser guide install: "
    f"Wandas {actual_version}, soundfile {soundfile.__version__}, "
    f"Matplotlib {importlib.metadata.version('matplotlib')}"
)
`);
  process.exitCode = 0;
} else {
  copyTree(pyodide, path.join(repositoryRoot, "tests", "core"), "/work/tests/core");
  copyTree(
    pyodide,
    path.join(repositoryRoot, "tests", "pyodide"),
    "/work/tests/pyodide",
  );
  writeVirtualFile(
    pyodide,
    path.join(repositoryRoot, "tests", "__init__.py"),
    "/work/tests/__init__.py",
  );
  writeVirtualFile(
    pyodide,
    path.join(repositoryRoot, "tests", "frame_helpers.py"),
    "/work/tests/frame_helpers.py",
  );
  writeVirtualFile(
    pyodide,
    path.join(repositoryRoot, "scripts", "run_pyodide_tests.py"),
    "/work/run_pyodide_tests.py",
  );

  await pyodide.loadPackage("pytest");
  pyodide.globals.set("wandas_wheel_uri", localWheelUri);
  pyodide.globals.set("expected_wandas_version", options["expected-wandas-version"]);
  await pyodide.runPythonAsync(`
import importlib.metadata
import json
import micropip

await micropip.install(
    [*json.loads(wandas_locked_requirements_json), wandas_wheel_uri]
)

actual_version = importlib.metadata.version("wandas")
if actual_version != expected_wandas_version:
    raise RuntimeError(
        f"Installed Wandas {actual_version}, expected {expected_wandas_version}"
    )
`);
  pyodide.globals.delete("wandas_wheel_uri");
  pyodide.globals.delete("expected_wandas_version");

  const exitCode = pyodide.runPython(`
import runpy

runpy.run_path("/work/run_pyodide_tests.py")["main"]()
`);
  process.exitCode = Number(exitCode);
}

pyodide.globals.delete("wandas_locked_requirements_json");

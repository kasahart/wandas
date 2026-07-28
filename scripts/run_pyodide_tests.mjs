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
if (!["guide-smoke", "source-tests"].includes(mode)) {
  throw new Error(`Unsupported Pyodide harness mode: ${mode}`);
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

if (mode === "guide-smoke") {
  const wandasInstallSpec = options["wandas-install-spec"];
  const expectedWandasVersion = options["expected-wandas-version"];
  if (!wandasInstallSpec || !expectedWandasVersion) {
    throw new Error("guide-smoke requires a Wandas install spec and expected version");
  }

  pyodide.globals.set("wandas_install_spec", wandasInstallSpec);
  pyodide.globals.set("expected_wandas_version", expectedWandasVersion);
  await pyodide.runPythonAsync(`
import importlib.metadata
import json

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

frame = wd.from_numpy(
    np.array([0.0, 0.25, -0.25], dtype=np.float64),
    sampling_rate=8_000,
    label="browser guide install smoke",
    ch_labels=["mono"],
)
if frame.sampling_rate != 8_000 or frame.n_channels != 1:
    raise RuntimeError("Published Wandas artifact failed the browser guide smoke")

print(
    "Browser guide install: "
    f"Wandas {actual_version}, soundfile {soundfile.__version__}, "
    f"Matplotlib {importlib.metadata.version('matplotlib')}"
)
`);
  process.exitCode = 0;
} else {
  const wheelPath = path.resolve(options.wheel);

  pyodide.FS.mkdirTree("/work/dist");
  const virtualWheel = `/work/dist/${path.basename(wheelPath)}`;
  writeVirtualFile(pyodide, wheelPath, virtualWheel);

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
  pyodide.globals.set("wandas_wheel_uri", `emfs:${virtualWheel}`);
  await pyodide.runPythonAsync(`
import json
import micropip

await micropip.install(
    [*json.loads(wandas_locked_requirements_json), wandas_wheel_uri]
)
`);
  pyodide.globals.delete("wandas_wheel_uri");

  const exitCode = pyodide.runPython(`
import runpy

runpy.run_path("/work/run_pyodide_tests.py")["main"]()
`);
  process.exitCode = Number(exitCode);
}

pyodide.globals.delete("wandas_locked_requirements_json");

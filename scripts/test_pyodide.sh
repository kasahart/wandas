#!/usr/bin/env bash
set -euo pipefail

PYODIDE_VERSION="314.0.3"
WANDAS_GUIDE_VERSION="0.6.1"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_manifest_dir="${repository_root}/scripts/pyodide"
cache_root="${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}/wandas/pyodide"
runtime_dir="${PYODIDE_RUNTIME_DIR:-${cache_root}/${PYODIDE_VERSION}}"
temporary_dir="$(mktemp -d)"

cleanup() {
    rm -rf "${temporary_dir}"
}
trap cleanup EXIT

for command in node npm uv; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: ${command} is required to run the Pyodide test harness" >&2
        exit 2
    fi
done

node_major="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
if ((node_major < 20)); then
    echo "error: Node.js 20 or newer is required; found $(node --version)" >&2
    exit 2
fi

mkdir -p "${runtime_dir}"
runtime_manifest_hash="$(
    node -e '
        const crypto = require("node:crypto");
        const fs = require("node:fs");
        const path = require("node:path");
        const hash = crypto.createHash("sha256");
        for (const manifest of process.argv.slice(1)) {
            hash.update(path.basename(manifest));
            hash.update("\0");
            hash.update(fs.readFileSync(manifest));
            hash.update("\0");
        }
        console.log(hash.digest("hex"));
    ' \
        "${runtime_manifest_dir}/package.json" \
        "${runtime_manifest_dir}/package-lock.json"
)"
installed_manifest_hash=""
if [[ -f "${runtime_dir}/.wandas-runtime-manifest.sha256" ]]; then
    installed_manifest_hash="$(<"${runtime_dir}/.wandas-runtime-manifest.sha256")"
fi
installed_version=""
if [[ -f "${runtime_dir}/node_modules/pyodide/package.json" ]]; then
    installed_version="$(
        node -e 'console.log(require(process.argv[1]).version)' \
            "${runtime_dir}/node_modules/pyodide/package.json"
    )"
fi

if [[
    "${installed_version}" != "${PYODIDE_VERSION}" ||
    "${installed_manifest_hash}" != "${runtime_manifest_hash}"
]]; then
    echo "Installing the locked Pyodide ${PYODIDE_VERSION} runtime into ${runtime_dir}"
    install -m 0644 "${runtime_manifest_dir}/package.json" "${runtime_dir}/package.json"
    install -m 0644 "${runtime_manifest_dir}/package-lock.json" "${runtime_dir}/package-lock.json"
    npm ci \
        --prefix "${runtime_dir}" \
        --ignore-scripts
    printf '%s\n' "${runtime_manifest_hash}" >"${runtime_dir}/.wandas-runtime-manifest.sha256"
fi

wheel_dir="${temporary_dir}/dist"
mkdir -p "${wheel_dir}"
(
    cd "${repository_root}"
    uv build --wheel --out-dir "${wheel_dir}"
)

shopt -s nullglob
wheel_paths=("${wheel_dir}"/wandas-*.whl)
shopt -u nullglob
if ((${#wheel_paths[@]} != 1)); then
    echo "error: uv build must produce exactly one Wandas wheel; found ${#wheel_paths[@]}" >&2
    exit 2
fi
wheel_path="${wheel_paths[0]}"

echo "Repository: ${repository_root}"
echo "Wheel: ${wheel_path}"
echo "Node.js: $(node --version)"
echo "Pyodide: ${PYODIDE_VERSION}"

echo "Validating the published browser-guide installation"
node \
    "${repository_root}/scripts/run_pyodide_tests.mjs" \
    --mode "guide-smoke" \
    --repository-root "${repository_root}" \
    --runtime-dir "${runtime_dir}" \
    --expected-pyodide-version "${PYODIDE_VERSION}" \
    --wandas-install-spec "wandas==${WANDAS_GUIDE_VERSION}" \
    --expected-wandas-version "${WANDAS_GUIDE_VERSION}"

echo "Validating the wheel built from the current checkout"
node \
    "${repository_root}/scripts/run_pyodide_tests.mjs" \
    --mode "source-tests" \
    --repository-root "${repository_root}" \
    --runtime-dir "${runtime_dir}" \
    --wheel "${wheel_path}" \
    --expected-pyodide-version "${PYODIDE_VERSION}"

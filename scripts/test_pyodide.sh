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
lock_hash="$(sha256sum "${runtime_manifest_dir}/package-lock.json" | cut -d ' ' -f 1)"
installed_lock_hash=""
if [[ -f "${runtime_dir}/.wandas-package-lock.sha256" ]]; then
    installed_lock_hash="$(<"${runtime_dir}/.wandas-package-lock.sha256")"
fi
installed_version=""
if [[ -f "${runtime_dir}/node_modules/pyodide/package.json" ]]; then
    installed_version="$(
        node -e 'console.log(require(process.argv[1]).version)' \
            "${runtime_dir}/node_modules/pyodide/package.json"
    )"
fi

if [[ "${installed_version}" != "${PYODIDE_VERSION}" || "${installed_lock_hash}" != "${lock_hash}" ]]; then
    echo "Installing the locked Pyodide ${PYODIDE_VERSION} runtime into ${runtime_dir}"
    install -m 0644 "${runtime_manifest_dir}/package.json" "${runtime_dir}/package.json"
    install -m 0644 "${runtime_manifest_dir}/package-lock.json" "${runtime_dir}/package-lock.json"
    npm ci \
        --prefix "${runtime_dir}" \
        --ignore-scripts
    printf '%s\n' "${lock_hash}" >"${runtime_dir}/.wandas-package-lock.sha256"
fi

wheel_dir="${temporary_dir}/dist"
mkdir -p "${wheel_dir}"
(
    cd "${repository_root}"
    uv build --wheel --out-dir "${wheel_dir}"
)

wheel_path="$(
    find "${wheel_dir}" -maxdepth 1 -type f -name 'wandas-*.whl' -print -quit
)"
if [[ -z "${wheel_path}" ]]; then
    echo "error: uv build did not produce a Wandas wheel" >&2
    exit 2
fi

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

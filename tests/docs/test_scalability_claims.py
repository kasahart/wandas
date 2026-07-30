from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_EN = REPO_ROOT / "README.md"
README_JA = REPO_ROOT / "README.ja.md"
DOCS_HOME = REPO_ROOT / "docs" / "src" / "index.md"
LEARNING_INTRO = REPO_ROOT / "learning-path" / "00_why_wandas.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_top_level_scalability_claims_link_to_canonical_contract() -> None:
    """Every broad entry point should lead readers to the exact execution contract."""
    assert "(docs/src/explanation/scalability-contract.md)" in _read(README_EN)
    assert "(docs/src/explanation/scalability-contract.md)" in _read(README_JA)
    assert "(explanation/scalability-contract.md)" in _read(DOCS_HOME)
    assert "https://kasahart.github.io/wandas/explanation/scalability-contract/" in _read(LEARNING_INTRO)


def test_top_level_claims_distinguish_all_materialization_boundaries() -> None:
    """Lazy construction must not be presented as an unbounded-memory guarantee."""
    english_surfaces = (_read(README_EN), _read(DOCS_HOME))
    japanese_surfaces = (_read(README_JA), _read(LEARNING_INTRO))

    for text in english_surfaces:
        assert "lazy Dask graph" in text
        assert "kernel" in text
        assert "materialize" in text
        assert "NumPy" in text
        assert "tensor" in text

    for text in japanese_surfaces:
        assert "遅延" in text
        assert "graph" in text
        assert "kernel" in text
        assert "実体化" in text
        assert "NumPy" in text
        assert "tensor" in text


def test_top_level_guidance_is_select_first_and_recording_bounded() -> None:
    """Examples should select files before processing bounded recordings."""
    english = _read(README_EN)
    japanese = _read(README_JA)
    learning = _read(LEARNING_INTRO)

    assert english.index("selected = dataset.select") < english.index("Only then load or process")
    assert japanese.index("selected = dataset.select") < japanese.index("その後で、選択済み")
    assert learning.index("selected = dataset.select") < learning.index("dataset = (selected")
    assert "processing before selection" not in english
    assert "選択前の Dataset 一括処理" not in japanese

    for text in (english, japanese, learning):
        assert "サイズを制御" in text or "bounded recording" in text


def test_top_level_docs_do_not_claim_unbounded_single_frame_distribution() -> None:
    """The former broad Dask claims must not return without the single-Frame limit."""
    all_surfaces = " ".join(
        "\n".join(_read(path) for path in (README_EN, README_JA, DOCS_HOME, LEARNING_INTRO)).split()
    )

    prohibited_claims = (
        "Efficiently process large data using dask",
        "daskを活用した効率的な大規模データ処理",
        "Dask統合で大規模データ対応",
        "Daskでメモリ効率的に処理",
    )
    for claim in prohibited_claims:
        assert claim not in all_surfaces

    assert "not by treating one enormous Frame as arbitrarily distributed" in all_surfaces
    assert "単一の巨大な Frame を自由に分散" in all_surfaces

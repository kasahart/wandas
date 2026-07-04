# Pipeline Recipe Requirements Check Notebook

This page points to the executable notebook for checking the Pipeline Recipe requirements with small synthetic frames.
このページは、小さな合成Frameで Pipeline Recipe 要件を確認する実行可能Notebookへの入口です。

- [Open the requirements check notebook / 要件確認Notebookを開く](pipeline-recipe-requirements-check.ipynb)
- [Read the requirements / 要件定義を読む](../explanation/pipeline-recipe-requirements.md)
- [Try the UX tutorial notebook / UXチュートリアルNotebookを試す](pipeline-recipes.md)

The notebook is assert-driven. It checks R1-R8 from the requirements document, including replay, input immutability, Dask laziness, sklearn adapter behavior when available, and optional dependency boundaries.
Notebookはassert中心です。要件定義のR1-R8について、replay、入力Frameの不変性、Dask laziness、sklearn adapterの挙動、optional dependencyの境界を確認します。

# Documentation consistency gates

Documentation is publishable only when its source examples, public API claims,
numerical meaning, exported learning applications, and completed site agree. CI runs
the same ordered gate once in its dedicated documentation job; deployment reruns the
site-producing portion and cannot reach the publish action after a failed stage.

## Automated contract

| Inconsistency class | Canonical automated evidence |
| --- | --- |
| Navigation, source-body links, repository links, and generated internal links | `tests/docs/test_docs_links.py`, strict MkDocs, and `scripts/check_docs_site.py` |
| Assets, fragments, canonical URLs, edit links, project prefix, and sitemap | `scripts/check_docs_site.py` plus deliberately broken generated-site fixtures |
| README and Markdown Python examples | executable docs tests and `markdown-exec` during strict MkDocs |
| All learning applications | exact 00–08 inventory, `marimo check`, execution/export of every app, finalization, and completed-site crawl |
| Undefined names and private, compatibility, or removed learner APIs | marimo reactive checks and the learning-path source/API policy tests |
| Canonical public inventory, package exports, API pages, and classification drift | the inventory from Issue #369 and its deliberate export mutation test |
| Public docstring parser/render/style | `scripts/check_public_docstrings.py`, focused malformed-style fixtures, and strict MkDocs rendering |
| FFT, Welch, and IFFT quantities, units, scaling, and reconstruction | the independent processing/Frame numerical tests and documentation contract tests from Issue #365 |
| Supported Python and optional extras | package metadata tests, learning-material version checks, public API policy, and the core-only wheel smoke job |

The orchestrator accepts the audit-baseline repository as a standalone profile so its
own PR can be checked. As soon as any prerequisite checker is integrated, it requires
the complete #365/#373/#369/#372/#367 cohort; partial integration is a hard failure.
Deployment always requires that final profile.

## Explicit manual checks

Automation does not make live third-party websites, browser-specific interactive
widgets, prose translation quality, or visual teaching clarity deterministic. Review
those items when their content changes. External HTTP availability is deliberately
excluded from CI to avoid making releases depend on unrelated services. These manual
checks do not replace any internal link, executable example, API, numerical, metadata,
or generated-site gate listed above.

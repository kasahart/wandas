# Documentation consistency gates

Documentation is publishable only when its source examples, public API claims,
numerical meaning, exported learning applications, and completed site agree. CI and
deployment both run the complete ordered gate, and neither can reach the publish
action after a source, API, numerical, learning, build, or site-validation failure.

## Automated contract

| Inconsistency class | Canonical automated evidence |
| --- | --- |
| Navigation, source-body links, repository links, and generated internal links | `tests/docs/test_docs_links.py`, strict MkDocs, and `scripts/check_docs_site.py` |
| Assets, fragments, canonical URLs, edit links, project prefix, and sitemap | `scripts/check_docs_site.py` plus deliberately broken generated-site fixtures |
| README and Markdown Python examples | executable docs tests and `markdown-exec` during strict MkDocs |
| All learning applications | exact numbered inventory (currently 00–08), isolated offline execution with checked-in fixtures, `marimo check`, export of every app, finalization, and completed-site crawl |
| Undefined names and private, compatibility, or removed learner APIs | marimo reactive checks and the learning-path source/API policy tests |
| Canonical public inventory, package exports, API pages, and classification drift | the inventory from Issue #369 and its deliberate export mutation test |
| Public docstring parser/render/style | `scripts/check_public_docstrings.py`, focused malformed-style fixtures, and strict MkDocs rendering |
| FFT, Welch, and IFFT quantities, units, scaling, and reconstruction | the independent processing/Frame numerical tests and documentation contract tests from Issue #365 |
| Supported Python and optional extras | package metadata tests, learning-material version checks, public API policy, and the core-only wheel smoke job |

The orchestrator accepts the audit-baseline repository as a standalone profile so its
own PR can be checked. During the ordered predecessor merges, an integration profile
accepts each fully installed checker while rejecting a half-installed multi-file
checker. PR #390 installs `.github/documentation-gate-finalized` as an irreversible
state-transition ledger. Its own PR base lacks that sentinel and may use the integration
profile; once merged, PR CI finds the sentinel in the base commit and main-push CI finds
it in the current commit, so deleting a whole checker group cannot downgrade CI back to
integration. Both then require the complete #365/#373/#369/#372/#367 final profile.
Deployment also always requires that final profile and does not use the
source-test-skipping `--site-only` mode. That mode is
accepted only for manual reruns of a final-profile checkout; standalone and integration
profiles reject it. Finalization and crawling read the canonical origin from the same
top-level `site_url` used by `docs/mkdocs.yml` rather than maintaining a second URL.

## Explicit manual checks

Learning-app execution uses checked-in fixtures from a temporary workspace, so it does
not write generated files into the repository or depend on external HTTP availability.
Automation does not make live third-party websites, browser-specific interactive
widgets, prose translation quality, or visual teaching clarity deterministic. Review
those items when their content changes. External HTTP availability is deliberately
excluded from CI to avoid making releases depend on unrelated services. These manual
checks do not replace any internal link, executable example, API, numerical, metadata,
or generated-site gate listed above.

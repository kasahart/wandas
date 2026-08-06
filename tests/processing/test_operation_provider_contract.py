from __future__ import annotations

import abc
import importlib
import threading
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any, cast

import pytest

import wandas.processing.base as base_module
from wandas.processing.base import (
    _OPERATION_CACHE,
    _OPERATION_PROVIDERS,
    AudioOperation,
    _EagerOperationProvider,
    _LazyOperationProvider,
    create_operation,
    get_operation,
    register_lazy_operation,
    register_operation,
)


@pytest.fixture(autouse=True)
def _restore_operation_state() -> Iterator[None]:
    """Keep provider and resolved-class state isolated between contract tests."""
    providers = dict(_OPERATION_PROVIDERS)
    cache = dict(_OPERATION_CACHE)
    yield
    _OPERATION_PROVIDERS.clear()
    _OPERATION_PROVIDERS.update(providers)
    _OPERATION_CACHE.clear()
    _OPERATION_CACHE.update(cache)


def _make_operation_class(
    operation_name: Any,
    *,
    module_name: str = "tests.fake_operation_implementations",
) -> type[AudioOperation[Any, Any]]:
    """Create a concrete operation with a controllable registry name."""

    class FakeOperation(AudioOperation[Any, Any]):
        name = operation_name

        def _process(self, data: Any) -> Any:
            return data

    FakeOperation.__module__ = module_name
    return FakeOperation


def _make_non_operation_class() -> type[object]:
    class NotAnAudioOperation:
        name = "not_an_audio_operation"

    return NotAnAudioOperation


def _make_abstract_operation(operation_name: str) -> type[AudioOperation[Any, Any]]:
    class AbstractOperation(AudioOperation[Any, Any], abc.ABC):
        name = operation_name

        @abc.abstractmethod
        def _process(self, data: Any) -> Any:
            raise NotImplementedError

    return AbstractOperation


def _module_with_operation(
    module_name: str,
    attribute_name: str,
    operation_class: object,
) -> ModuleType:
    module = ModuleType(module_name)
    setattr(module, attribute_name, operation_class)
    return module


def _patch_import_module(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    def fake_import_module(requested_name: str, *_args: Any, **_kwargs: Any) -> ModuleType:
        assert requested_name == module.__name__
        return module

    monkeypatch.setattr(base_module.importlib, "import_module", fake_import_module)


def _assert_error_mentions(error: pytest.ExceptionInfo[BaseException], *parts: str) -> None:
    message = str(error.value)
    assert all(part in message for part in parts), message


def test_register_operation_publishes_one_eager_provider_and_cache_entry() -> None:
    operation_class = _make_operation_class("contract_eager_registration")

    register_operation(operation_class)

    provider = _OPERATION_PROVIDERS[operation_class.name]
    assert isinstance(provider, _EagerOperationProvider)
    assert provider.operation_class is operation_class
    assert _OPERATION_CACHE[operation_class.name] is operation_class
    assert get_operation(operation_class.name) is operation_class


def test_register_operation_same_class_object_is_idempotent() -> None:
    operation_class = _make_operation_class("contract_eager_idempotence")

    register_operation(operation_class)
    provider_before = _OPERATION_PROVIDERS[operation_class.name]
    cache_before = _OPERATION_CACHE[operation_class.name]

    register_operation(operation_class)

    assert _OPERATION_PROVIDERS[operation_class.name] is provider_before
    assert _OPERATION_CACHE[operation_class.name] is cache_before is operation_class


def test_register_operation_rejects_same_module_and_qualname_from_factory() -> None:
    first_class = _make_operation_class("contract_eager_identity_conflict")
    second_class = _make_operation_class("contract_eager_identity_conflict")
    assert first_class is not second_class
    assert first_class.__module__ == second_class.__module__
    assert first_class.__qualname__ == second_class.__qualname__

    register_operation(first_class)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises(ValueError) as error:
        register_operation(second_class)

    _assert_error_mentions(error, "contract_eager_identity_conflict")
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_register_operation_rejects_non_class_without_changing_state() -> None:
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((TypeError, ValueError)) as error:
        register_operation(cast(Any, object()))

    _assert_error_mentions(error, "AudioOperation")
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


@pytest.mark.parametrize(
    ("operation_class_factory", "expected_message"),
    [
        (_make_non_operation_class, "AudioOperation"),
        (lambda: _make_abstract_operation("contract_abstract_eager"), "abstract"),
        (lambda: _make_operation_class(""), "non-blank str"),
        (lambda: _make_operation_class(" "), "non-blank str"),
    ],
    ids=["non-subclass", "abstract", "blank-name", "whitespace-name"],
)
def test_register_operation_validates_candidate_before_mutating_state(
    operation_class_factory: Callable[[], object],
    expected_message: str,
) -> None:
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((TypeError, ValueError)) as error:
        register_operation(cast(Any, operation_class_factory()))

    _assert_error_mentions(error, expected_message)
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_register_lazy_operation_requires_explicit_keyword_attribute_name() -> None:
    register_without_attribute = cast(Callable[..., None], register_lazy_operation)

    with pytest.raises(TypeError):
        register_without_attribute(
            "contract_missing_attribute",
            "tests.fake_lazy_module",
        )


@pytest.mark.parametrize(
    ("operation_name", "module_name", "attribute_name"),
    [
        ("", "tests.fake_lazy_module", "Operation"),
        (" ", "tests.fake_lazy_module", "Operation"),
        ("contract_invalid_lazy", "", "Operation"),
        ("contract_invalid_lazy", " ", "Operation"),
        ("contract_invalid_lazy", "tests.fake_lazy_module", ""),
        ("contract_invalid_lazy", "tests.fake_lazy_module", " "),
        (None, "tests.fake_lazy_module", "Operation"),
    ],
    ids=[
        "blank-name",
        "whitespace-name",
        "blank-module",
        "whitespace-module",
        "blank-attribute",
        "whitespace-attribute",
        "non-string-name",
    ],
)
def test_register_lazy_operation_validates_provider_fields_atomically(
    operation_name: Any,
    module_name: str,
    attribute_name: str,
) -> None:
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((TypeError, ValueError)):
        register_lazy_operation(
            cast(str, operation_name),
            module_name,
            attribute_name=attribute_name,
        )

    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_register_lazy_operation_records_explicit_provider_without_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_called = False

    def fail_import(*_args: Any, **_kwargs: Any) -> ModuleType:
        nonlocal import_called
        import_called = True
        raise AssertionError("lazy provider registration must not import its module")

    monkeypatch.setattr(base_module.importlib, "import_module", fail_import)
    register_lazy_operation(
        "contract_lazy_registration",
        "tests.module_for_contract_lazy_registration",
        attribute_name="ExplicitOperation",
    )

    provider = _OPERATION_PROVIDERS["contract_lazy_registration"]
    assert isinstance(provider, _LazyOperationProvider)
    assert provider.module_name == "tests.module_for_contract_lazy_registration"
    assert provider.attribute_name == "ExplicitOperation"
    assert "contract_lazy_registration" not in _OPERATION_CACHE
    assert import_called is False


def test_register_lazy_operation_same_provider_is_idempotent() -> None:
    register_lazy_operation(
        "contract_lazy_idempotence",
        "tests.module_for_contract_lazy_idempotence",
        attribute_name="ExplicitOperation",
    )
    provider_before = _OPERATION_PROVIDERS["contract_lazy_idempotence"]
    cache_before = dict(_OPERATION_CACHE)

    register_lazy_operation(
        "contract_lazy_idempotence",
        "tests.module_for_contract_lazy_idempotence",
        attribute_name="ExplicitOperation",
    )

    assert _OPERATION_PROVIDERS["contract_lazy_idempotence"] is provider_before
    assert _OPERATION_CACHE == cache_before


@pytest.mark.parametrize(
    ("conflicting_module", "conflicting_attribute"),
    [
        ("tests.other_lazy_module", "ExplicitOperation"),
        ("tests.module_for_contract_lazy_conflict", "OtherOperation"),
    ],
    ids=["module-conflict", "attribute-conflict"],
)
def test_register_lazy_operation_rejects_different_provider(
    conflicting_module: str,
    conflicting_attribute: str,
) -> None:
    operation_name = "contract_lazy_provider_conflict"
    original_module = "tests.module_for_contract_lazy_conflict"
    original_attribute = "ExplicitOperation"
    register_lazy_operation(operation_name, original_module, attribute_name=original_attribute)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises(ValueError) as error:
        register_lazy_operation(operation_name, conflicting_module, attribute_name=conflicting_attribute)

    _assert_error_mentions(error, operation_name, conflicting_module, conflicting_attribute)
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_register_eager_then_lazy_conflict_preserves_eager_provider_and_cache() -> None:
    operation_name = "contract_eager_then_lazy_conflict"
    operation_class = _make_operation_class(operation_name)
    register_operation(operation_class)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises(ValueError) as error:
        register_lazy_operation(
            operation_name,
            "tests.module_for_eager_then_lazy_conflict",
            attribute_name="OtherOperation",
        )

    _assert_error_mentions(error, operation_name)
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_register_lazy_then_eager_conflict_preserves_lazy_provider_and_cache() -> None:
    operation_name = "contract_lazy_then_eager_conflict"
    module_name = "tests.module_for_lazy_then_eager_conflict"
    attribute_name = "ExplicitOperation"
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises(ValueError) as error:
        register_operation(_make_operation_class(operation_name))

    _assert_error_mentions(error, operation_name)
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_get_operation_resolves_explicit_attribute_even_when_class_module_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_name = "contract_explicit_attribute_resolution"
    module_name = "tests.provider_module_for_explicit_attribute"
    attribute_name = "PublishedOperation"
    operation_class = _make_operation_class(operation_name, module_name="tests.implementation_module")
    module = _module_with_operation(module_name, attribute_name, operation_class)
    _patch_import_module(monkeypatch, module)
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    provider_before = _OPERATION_PROVIDERS[operation_name]

    resolved = get_operation(operation_name)

    assert resolved is operation_class
    assert operation_class.__module__ != module_name
    assert _OPERATION_PROVIDERS[operation_name] is provider_before
    assert _OPERATION_CACHE[operation_name] is operation_class


def test_get_operation_preserves_import_exception_and_allows_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    operation_name = "contract_import_exception"
    module_name = "tests.module_for_import_exception"
    attribute_name = "ExplicitOperation"
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)
    import_error = ImportError("dependency unavailable for provider module")

    def fail_import(requested_name: str, *_args: Any, **_kwargs: Any) -> ModuleType:
        assert requested_name == module_name
        raise import_error

    monkeypatch.setattr(base_module.importlib, "import_module", fail_import)
    with pytest.raises(ImportError) as error:
        get_operation(operation_name)

    assert error.value is import_error
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before

    operation_class = _make_operation_class(operation_name)
    module = _module_with_operation(module_name, attribute_name, operation_class)
    _patch_import_module(monkeypatch, module)
    assert get_operation(operation_name) is operation_class
    assert _OPERATION_CACHE[operation_name] is operation_class


def test_get_operation_reports_missing_explicit_attribute_without_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_name = "contract_missing_attribute"
    module_name = "tests.module_for_missing_attribute"
    attribute_name = "MissingOperation"
    module = ModuleType(module_name)
    _patch_import_module(monkeypatch, module)
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((AttributeError, ValueError)) as error:
        get_operation(operation_name)

    _assert_error_mentions(error, module_name, attribute_name)
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_get_operation_rejects_non_class_without_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    operation_name = "contract_non_class_candidate"
    module_name = "tests.module_for_non_class_candidate"
    attribute_name = "PublishedOperation"
    module = _module_with_operation(module_name, attribute_name, object())
    _patch_import_module(monkeypatch, module)
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((TypeError, ValueError)) as error:
        get_operation(operation_name)

    _assert_error_mentions(error, operation_name, module_name, attribute_name, "AudioOperation")
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_get_operation_rejects_non_subclass_without_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    operation_name = "contract_non_subclass_candidate"
    module_name = "tests.module_for_non_subclass_candidate"
    attribute_name = "PublishedOperation"
    module = _module_with_operation(module_name, attribute_name, _make_non_operation_class())
    _patch_import_module(monkeypatch, module)
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((TypeError, ValueError)) as error:
        get_operation(operation_name)

    _assert_error_mentions(error, operation_name, module_name, attribute_name, "AudioOperation")
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_get_operation_rejects_abstract_class_without_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    operation_name = "contract_abstract_candidate"
    module_name = "tests.module_for_abstract_candidate"
    attribute_name = "PublishedOperation"
    module = _module_with_operation(
        module_name,
        attribute_name,
        _make_abstract_operation(operation_name),
    )
    _patch_import_module(monkeypatch, module)
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((TypeError, ValueError)) as error:
        get_operation(operation_name)

    _assert_error_mentions(error, operation_name, module_name, attribute_name, "abstract")
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_get_operation_rejects_class_name_mismatch_without_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    operation_name = "contract_provider_name_mismatch"
    module_name = "tests.module_for_provider_name_mismatch"
    attribute_name = "PublishedOperation"
    module = _module_with_operation(
        module_name,
        attribute_name,
        _make_operation_class("different_operation_name"),
    )
    _patch_import_module(monkeypatch, module)
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    with pytest.raises((TypeError, ValueError)) as error:
        get_operation(operation_name)

    _assert_error_mentions(error, operation_name, module_name, attribute_name, "different_operation_name")
    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_get_operation_cache_hit_returns_same_class_without_reimporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_name = "contract_cache_hit"
    module_name = "tests.module_for_cache_hit"
    attribute_name = "PublishedOperation"
    operation_class = _make_operation_class(operation_name)
    module = _module_with_operation(module_name, attribute_name, operation_class)
    import_calls = 0

    def fake_import(requested_name: str, *_args: Any, **_kwargs: Any) -> ModuleType:
        nonlocal import_calls
        assert requested_name == module_name
        import_calls += 1
        return module

    monkeypatch.setattr(base_module.importlib, "import_module", fake_import)
    register_lazy_operation(operation_name, module_name, attribute_name=attribute_name)

    first = get_operation(operation_name)
    second = get_operation(operation_name)

    assert first is second is operation_class
    assert import_calls == 1


def test_create_operation_uses_get_operation_as_its_resolution_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_name = "contract_create_operation_resolution"
    operation_class = _make_operation_class(operation_name)
    register_operation(operation_class)
    calls: list[str] = []
    real_get_operation = base_module.get_operation

    def recording_get_operation(name: str) -> type[AudioOperation[Any, Any]]:
        calls.append(name)
        return real_get_operation(name)

    monkeypatch.setattr(base_module, "get_operation", recording_get_operation)
    created = create_operation(operation_name, 16000)

    assert isinstance(created, operation_class)
    assert calls == [operation_name]


@pytest.mark.parametrize(
    ("operation_name", "module_name", "attribute_name"),
    [
        ("cepstrum", "wandas.processing.cepstral", "Cepstrum"),
        ("coherence", "wandas.processing.spectral", "Coherence"),
        ("roughness_dw", "wandas.processing.psychoacoustic", "RoughnessDw"),
    ],
    ids=["cepstrum", "coherence", "roughness-dw"],
)
def test_public_lazy_export_direct_import_and_get_operation_share_identity(
    operation_name: str,
    module_name: str,
    attribute_name: str,
) -> None:
    import wandas.processing as processing

    module = importlib.import_module(module_name)
    direct_class = getattr(module, attribute_name)

    assert get_operation(operation_name) is direct_class
    assert getattr(processing, attribute_name) is direct_class


def test_builtin_lazy_and_private_recipe_declarations_use_explicit_providers() -> None:
    import wandas.processing as processing

    declarations = (*processing._LAZY_OPERATION_CLASSES.values(), *processing._PRIVATE_LAZY_OPERATION_PROVIDERS)
    for operation_name, module_name, attribute_name in declarations:
        provider = _OPERATION_PROVIDERS[operation_name]
        assert provider == _LazyOperationProvider(module_name, attribute_name)


def test_builtin_eager_private_recipe_operations_have_explicit_providers() -> None:
    from wandas.processing.temporal import _RecipeRmsTrendV1, _RecipeSoundLevelV1

    for operation_class in (_RecipeRmsTrendV1, _RecipeSoundLevelV1):
        provider = _OPERATION_PROVIDERS[operation_class.name]
        assert provider == _EagerOperationProvider(operation_class)


@pytest.mark.parametrize(
    "module_name",
    [
        "wandas.processing.cepstral",
        "wandas.processing.spectral",
        "wandas.processing.psychoacoustic",
    ],
    ids=["cepstral", "spectral", "psychoacoustic"],
)
def test_direct_lazy_module_import_does_not_mutate_provider_or_cache(module_name: str) -> None:
    providers_before = dict(_OPERATION_PROVIDERS)
    cache_before = dict(_OPERATION_CACHE)

    importlib.import_module(module_name)

    assert _OPERATION_PROVIDERS == providers_before
    assert _OPERATION_CACHE == cache_before


def test_concurrent_lazy_resolution_converges_and_does_not_block_other_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow_name = "contract_concurrent_slow"
    fast_name = "contract_concurrent_fast"
    slow_module = _module_with_operation(
        "tests.module_for_concurrent_slow",
        "SlowOperation",
        _make_operation_class(slow_name),
    )
    fast_module = _module_with_operation(
        "tests.module_for_concurrent_fast",
        "FastOperation",
        _make_operation_class(fast_name),
    )
    register_lazy_operation(slow_name, slow_module.__name__, attribute_name="SlowOperation")
    register_lazy_operation(fast_name, fast_module.__name__, attribute_name="FastOperation")

    slow_import_started = threading.Event()
    release_slow_import = threading.Event()
    fast_import_finished = threading.Event()
    outcomes: dict[str, object] = {}

    def fake_import(requested_name: str, *_args: Any, **_kwargs: Any) -> ModuleType:
        if requested_name == slow_module.__name__:
            slow_import_started.set()
            if not release_slow_import.wait(timeout=5):
                raise TimeoutError("slow provider import was not released")
            return slow_module
        if requested_name == fast_module.__name__:
            fast_import_finished.set()
            return fast_module
        raise AssertionError(f"unexpected provider module: {requested_name}")

    monkeypatch.setattr(base_module.importlib, "import_module", fake_import)

    def resolve(label: str, operation_name: str) -> None:
        try:
            outcomes[label] = get_operation(operation_name)
        except BaseException as error:  # pragma: no cover - reported below
            outcomes[label] = error

    slow_first = threading.Thread(target=resolve, args=("slow-first", slow_name))
    slow_second = threading.Thread(target=resolve, args=("slow-second", slow_name))
    fast = threading.Thread(target=resolve, args=("fast", fast_name))
    slow_first.start()
    assert slow_import_started.wait(timeout=2)
    slow_second.start()
    fast.start()

    try:
        assert fast_import_finished.wait(timeout=2), "a distinct provider import was blocked by the slow import"
    finally:
        release_slow_import.set()
        slow_first.join(timeout=5)
        slow_second.join(timeout=5)
        fast.join(timeout=5)

    assert not any(isinstance(outcome, BaseException) for outcome in outcomes.values())
    assert outcomes["slow-first"] is slow_module.SlowOperation
    assert outcomes["slow-second"] is slow_module.SlowOperation
    assert outcomes["fast"] is fast_module.FastOperation

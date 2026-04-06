"""Test suite for introspection utilities module."""

from wandas.utils.introspection import accepted_kwargs, filter_kwargs


class TestAcceptedKwargs:
    """Test suite for accepted_kwargs — parameter inspection utility."""

    def test_accepted_kwargs_no_kwargs_returns_all_params(self) -> None:
        def func(a, b, c):
            return a + b + c

        params, has_var_kwargs = accepted_kwargs(func)
        assert params == {"a", "b", "c"}
        assert not has_var_kwargs

    def test_accepted_kwargs_with_var_kwargs_detects_star_star(self) -> None:
        def func(a, b, **kwargs):
            return a + b + sum(kwargs.values())

        params, has_var_kwargs = accepted_kwargs(func)
        assert params == {"a", "b"}
        assert has_var_kwargs

    def test_accepted_kwargs_caching_returns_consistent_results(self) -> None:
        def func(a, b, c=1):
            return a + b + c

        result1 = accepted_kwargs(func)
        result2 = accepted_kwargs(func)
        assert result1[0] == {"a", "b", "c"}
        assert result1[1] is False
        # Subsequent calls return same values (cache hit)
        assert result2[0] == result1[0]
        assert result2[1] is result1[1]


class TestFilterKwargs:
    """Test suite for filter_kwargs — keyword argument filtering utility."""

    def test_filter_kwargs_no_var_kwargs_drops_unknown(self) -> None:
        def func(a, b, c=1):
            return a + b + c

        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        filtered = filter_kwargs(func, kwargs)
        assert filtered == {"a": 1, "b": 2, "c": 3}
        assert "d" not in filtered

    def test_filter_kwargs_with_var_kwargs_passes_all(self) -> None:
        def func(a, b, **kwargs):
            return a + b + sum(kwargs.values())

        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        filtered = filter_kwargs(func, kwargs)
        assert filtered == kwargs

    def test_filter_kwargs_strict_mode_ignores_var_kwargs(self) -> None:
        def func(a, b, **kwargs):
            return a + b + sum(kwargs.values())

        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        filtered = filter_kwargs(func, kwargs, strict_mode=True)
        assert filtered == {"a": 1, "b": 2}
        assert "c" not in filtered
        assert "d" not in filtered

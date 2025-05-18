"""introspection utilities module tests."""

from wandas.utils.introspection import accepted_kwargs, filter_kwargs


def test_accepted_kwargs_with_no_kwargs():
    """Test accepted_kwargs with a function that has no kwargs."""

    def func(a, b, c):
        return a + b + c

    params, has_var_kwargs = accepted_kwargs(func)
    assert params == {"a", "b", "c"}
    assert not has_var_kwargs


def test_accepted_kwargs_with_var_kwargs():
    """Test accepted_kwargs with a function that has var kwargs."""

    def func(a, b, **kwargs):
        return a + b + sum(kwargs.values())

    params, has_var_kwargs = accepted_kwargs(func)
    assert params == {"a", "b"}
    assert has_var_kwargs


def test_filter_kwargs_with_no_var_kwargs():
    """Test filter_kwargs with a function that has no var kwargs."""

    def func(a, b, c=1):
        return a + b + c

    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    filtered = filter_kwargs(func, kwargs)
    assert filtered == {"a": 1, "b": 2, "c": 3}
    assert "d" not in filtered


def test_filter_kwargs_with_var_kwargs():
    """Test filter_kwargs with a function that has var kwargs."""

    def func(a, b, **kwargs):
        return a + b + sum(kwargs.values())

    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    filtered = filter_kwargs(func, kwargs)
    assert filtered == kwargs


def test_filter_kwargs_with_var_kwargs_strict_mode():
    """Test filter_kwargs with a function that has var kwargs in strict mode."""

    def func(a, b, **kwargs):
        return a + b + sum(kwargs.values())

    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    filtered = filter_kwargs(func, kwargs, strict_mode=True)
    assert filtered == {"a": 1, "b": 2}
    assert "c" not in filtered
    assert "d" not in filtered


def test_filter_kwargs_caching():
    """Test the caching behavior of filter_kwargs."""

    def func(a, b, c=1):
        return a + b + c

    # First call to accepted_kwargs initializes the cache
    cached_result1 = accepted_kwargs(func)
    expected_params = {"a", "b", "c"}
    expected_has_var_kwargs = False
    assert cached_result1[0] == expected_params
    assert cached_result1[1] is expected_has_var_kwargs

    # Subsequent calls should return the same values
    cached_result2 = accepted_kwargs(func)
    assert cached_result2[0] == cached_result1[0]
    assert cached_result2[1] is cached_result1[1]

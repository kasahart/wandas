# Recipe v2 support matrix

| Capability | Representation |
| --- | --- |
| unary same-frame operation | one `frame` binding |
| typed Frame transition | one `frame` binding and a Frame-returning handler |
| scalar arithmetic | one `frame` binding plus a canonical scalar parameter |
| Frame arithmetic | two ordered `frame` bindings |
| external NumPy/Dask arithmetic | ordered `frame` and `array` bindings |
| indexing and multidimensional slicing | one `frame` binding plus canonical selectors |
| channel addition | ordered `base` and `data` bindings |
| signal mixing | ordered `base` and `other` bindings |
| extension operation | explicit `@recipe_operation` plus an immutable registry value |

All supported rows compile to the same schema-2 `RecipePlan`. External arrays stay
named inputs and do not expose their NumPy/Dask implementation in history or schema.
`Frame.apply(callable)` is runtime-only; portable extensions declare a stable public
operation. Unknown operations and versions fail closed rather than producing partial
Recipes.

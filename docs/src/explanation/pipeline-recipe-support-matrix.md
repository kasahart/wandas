# Recipe v2 support matrix

| Capability | Representation |
| --- | --- |
| unary same-frame operation | explicit-opt-in `AudioReplay` |
| typed frame transition | `MethodReplay` |
| scalar/frame/external-array arithmetic | `BinaryReplay` |
| indexing and multidimensional slicing | `IndexReplay` |
| frame or external-array channel addition | `AddChannelReplay` |
| importable custom function | `CustomReplay` |
| true ordered multi-input operation | `MultiInputReplay` |

All supported rows compile to the same `RecipePlan`. External arrays stay named inputs.
Unknown operations and versions fail closed rather than producing partial Recipes.

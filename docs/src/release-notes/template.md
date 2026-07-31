# Wandas X.Y.Z

Use this source template for a feature release or any release containing a
compatibility change, then remove guidance that does not apply. An ordinary patch
release with no compatibility change may state that none occurred without copying
the full table. User-visible compatibility changes must follow the classification and
exception process in the
[public API stability policy](../explanation/public-api-stability.md).

## Highlights

- Describe the release outcome.

## Changes

- Link each included pull request.

## Compatibility

For every removal or incompatible semantic change, complete one row. Write `None`
only when the linked decision explicitly approves an exception.

| Affected surface or artifact | Classification | Deprecation start | Replacement or migration | Removal/change version | Exception reason and decision link |
| --- | --- | --- | --- | --- | --- |
| `name` | Stable / Experimental / Serialized / Internal-only | `X.Y.Z` or `None` | `replacement` | `X.Y.Z` | `Not applicable` or reason + issue/PR |

State explicitly when the release contains no compatibility changes.

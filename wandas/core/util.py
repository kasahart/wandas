def unit_to_ref(unit: str) -> float:
    """
    単位を参照値に変換します。
    """
    if unit == "Pa":
        return 2e-5

    else:
        return 1.0

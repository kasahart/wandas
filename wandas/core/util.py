from typing import Callable


def transform_method(target_class=None):
    """
    デコレータ: 変換メソッドをラップし、共通の処理を適用します。
    """

    def decorator(func: Callable):
        def wrapper(self, *args, **kwargs):
            # target_class が文字列の場合に実際のクラスオブジェクトに解決

            target_cls = self.__class__ if target_class is None else target_class

            # データ変換を実行
            result = func(self, *args, **kwargs)
            return target_cls(
                data=result.pop("data"),
                sampling_rate=result.pop("sampling_rate", self.sampling_rate),
                label=result.pop("label", self.label),
                unit=result.pop("unit", self.unit),
                calibration_value=result.pop(
                    "calibration_value", self.calibration_value
                ),
                metadata=result.pop("metadata", self.metadata.copy()),
                **result,  # target_classに必要な追加の引数
            )

        return wrapper

    return decorator


def unit_to_ref(unit: str) -> float:
    """
    単位を参照値に変換します。
    """
    if unit == "Pa":
        return 2e-5

    else:
        return 1.0

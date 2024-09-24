from typing import Callable, Dict, Any

# from wandas.core.base_channel import BaseChannel
# from wandas.core.channel import Channel
# from wandas.core.frequency_channel import FrequencyChannel
# from wandas.core.time_frequency_channel import TimeFrequencyChannel


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
                metadata=result.pop("metadata", self.metadata.copy()),
                **result,  # target_classに必要な追加の引数
            )

        return wrapper

    return decorator


def transform_channel(org, target_class, **kwargs):
    # データ変換を実行
    return target_class(
        data=kwargs.pop("data"),
        sampling_rate=kwargs.pop("sampling_rate", org.sampling_rate),
        label=kwargs.pop("label", org.label),
        unit=kwargs.pop("unit", org.unit),
        metadata=kwargs.pop("metadata", org.metadata.copy()),
        **kwargs,  # target_classに必要な追加の引数
    )


# @transform_method(Channel)
# def toChannel(ch: BaseChannel, target_Channel, dict: Dict[str, Any]):
#     """
#     自身を Channel オブジェクトに変換して返します。
#     """


# @transform_method(FrequencyChannel)
# def toFrequencyChannel(self):
#     """
#     自身を FrequencyChannel オブジェクトに変換して返します。
#     """
#     pass


# @transform_method(TimeFrequencyChannel)
# def toTimeFrequencyChannel(self):
#     """
#     自身を TimeFrequencyChannel オブジェクトに変換して返します。
#     """
#     pass


def unit_to_ref(unit: str) -> float:
    """
    単位を参照値に変換します。
    """
    if unit == "Pa":
        return 2e-5

    else:
        return 1.0

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast, overload

from tqdm.auto import tqdm

from wandas.core.lazy.channel_frame import ChannelFrame
from wandas.core.lazy.spectrogram_frame import SpectrogramFrame

logger = logging.getLogger(__name__)

FrameType = Union[ChannelFrame, SpectrogramFrame]
F = TypeVar("F", bound=FrameType)
F_out = TypeVar("F_out", bound=FrameType)


@dataclass
class LazyFrame(Generic[F]):
    """
    フレームとそのロード状態をカプセル化するクラス。

    Attributes:
        file_path: フレームに関連付けられたファイルパス
        frame: ロードされたフレームオブジェクト（未ロードの場合はNone）
        is_loaded: フレームがロード済みかどうかのフラグ
        load_attempted: ロードを試みたかどうかのフラグ（エラー検出用）
    """

    file_path: Path
    frame: Optional[F] = None
    is_loaded: bool = False
    load_attempted: bool = False

    def ensure_loaded(self, loader: Callable[[Path], Optional[F]]) -> Optional[F]:
        """
        フレームがロードされていることを確認し、必要であればロードする。

        Args:
            loader: ファイルパスからフレームをロードする関数

        Returns:
            ロードされたフレーム、またはロードに失敗した場合はNone
        """
        # すでにロード済みの場合は現在のフレームを返す
        if self.is_loaded:
            return self.frame

        # まだロードしていない場合はロードを試みる
        try:
            self.load_attempted = True
            self.frame = loader(self.file_path)
            self.is_loaded = True
            return self.frame
        except Exception as e:
            logger.error(f"ファイル {self.file_path} の読み込みに失敗: {str(e)}")
            self.is_loaded = True  # ロードは試みた
            self.frame = None
            return None

    def reset(self) -> None:
        """
        フレームの状態をリセットする。
        """
        self.frame = None
        self.is_loaded = False
        self.load_attempted = False


class FrameDataset(Generic[F], ABC):
    """
    フォルダ内のファイルを処理するための抽象基底データセットクラス。
    遅延ロード機能を持ち、大規模データセットを効率的に扱います。
    サブクラスで具体的なフレームタイプ（ChannelFrame, SpectrogramFrameなど）を扱います。
    """

    def __init__(
        self,
        folder_path: str,
        sampling_rate: Optional[int] = None,
        signal_length: Optional[int] = None,
        file_extensions: Optional[list[str]] = None,
        lazy_loading: bool = True,
        recursive: bool = False,
        source_dataset: Optional["FrameDataset[Any]"] = None,
        transform: Optional[Callable[[Any], Optional[F]]] = None,
    ):
        self.folder_path = Path(folder_path)
        if source_dataset is None and not self.folder_path.exists():
            raise FileNotFoundError(f"フォルダが存在しません: {self.folder_path}")

        self.sampling_rate = sampling_rate
        self.signal_length = signal_length
        self.file_extensions = file_extensions or [".wav"]
        self._recursive = recursive
        self._lazy_loading = lazy_loading

        # LazyFrameのリストに変更
        self._lazy_frames: list[LazyFrame[F]] = []

        self._source_dataset = source_dataset
        self._transform = transform

        if self._source_dataset:
            self._initialize_from_source()
        else:
            self._initialize_from_folder()

    def _initialize_from_source(self) -> None:
        """ソースデータセットを元に初期化します。"""
        if self._source_dataset is None:
            return

        # ソースからファイルパスをコピー
        file_paths = self._source_dataset._get_file_paths()
        self._lazy_frames = [LazyFrame(file_path) for file_path in file_paths]

        # 他のプロパティを継承
        self.sampling_rate = self.sampling_rate or self._source_dataset.sampling_rate
        self.signal_length = self.signal_length or self._source_dataset.signal_length
        self.file_extensions = (
            self.file_extensions or self._source_dataset.file_extensions
        )
        self._recursive = self._source_dataset._recursive
        self.folder_path = self._source_dataset.folder_path

    def _initialize_from_folder(self) -> None:
        """フォルダを元に初期化します。"""
        self._discover_files()
        if not self._lazy_loading:
            self._load_all_files()

    def _discover_files(self) -> None:
        """フォルダ内のファイルを探してLazyFrameのリストに格納します。"""
        file_paths = []
        for ext in self.file_extensions:
            pattern = f"**/*{ext}" if self._recursive else f"*{ext}"
            file_paths.extend(
                sorted(p for p in self.folder_path.glob(pattern) if p.is_file())
            )

        # 重複を排除して並べ替え
        file_paths = sorted(list(set(file_paths)))

        # LazyFrameのリストを作成
        self._lazy_frames = [LazyFrame(file_path) for file_path in file_paths]

    def _load_all_files(self) -> None:
        """すべてのファイルをロードします。"""
        for i in tqdm(range(len(self._lazy_frames)), desc="ロード/変換中"):
            try:
                self._ensure_loaded(i)
            except Exception as e:
                filepath = self._lazy_frames[i].file_path
                logger.warning(
                    f"インデックス {i} ({filepath}) のロード/変換に失敗: {str(e)}"
                )
        self._lazy_loading = False

    @abstractmethod
    def _load_file(self, file_path: Path) -> Optional[F]:
        """ファイルからフレームをロードするための抽象メソッド。"""
        pass

    def _load_from_source(self, index: int) -> Optional[F]:
        """ソースデータセットからフレームをロードし、必要に応じて変換します。"""
        if self._source_dataset is None or self._transform is None:
            return None

        source_frame = self._source_dataset._ensure_loaded(index)
        if source_frame is None:
            return None

        try:
            return self._transform(source_frame)
        except Exception as e:
            logger.warning(f"インデックス {index} の変換に失敗: {str(e)}")
            return None

    def _ensure_loaded(self, index: int) -> Optional[F]:
        """インデックスに対応するフレームがロードされていることを確認します。"""
        if not (0 <= index < len(self._lazy_frames)):
            raise IndexError(
                f"インデックス {index} は範囲外です (0-{len(self._lazy_frames) - 1})"
            )

        lazy_frame = self._lazy_frames[index]

        # 既にロード済みならそれを返す
        if lazy_frame.is_loaded:
            return lazy_frame.frame

        try:
            # ソースデータセットから変換する場合
            if self._transform and self._source_dataset:
                lazy_frame.load_attempted = True
                frame = self._load_from_source(index)
                lazy_frame.frame = frame
                lazy_frame.is_loaded = True
                return frame
            # ファイルから直接ロードする場合
            else:
                return lazy_frame.ensure_loaded(self._load_file)
        except Exception as e:
            f_path = lazy_frame.file_path
            logger.error(
                f"index {index} ({f_path}) の読み込みまたは初期処理に失敗: {str(e)}"
            )
            lazy_frame.frame = None
            lazy_frame.is_loaded = True
            lazy_frame.load_attempted = True
            return None

    def _get_file_paths(self) -> list[Path]:
        """ファイルパスのリストを取得します。"""
        return [lazy_frame.file_path for lazy_frame in self._lazy_frames]

    def __len__(self) -> int:
        """データセット内のファイル数を返します。"""
        return len(self._lazy_frames)

    def __getitem__(self, index: int) -> Optional[F]:
        """指定インデックスのフレームを取得します。"""
        return self._ensure_loaded(index)

    @overload
    def apply(self, func: Callable[[F], Optional[F_out]]) -> "FrameDataset[F_out]": ...

    @overload
    def apply(self, func: Callable[[F], Optional[Any]]) -> "FrameDataset[Any]": ...

    def apply(self, func: Callable[[F], Optional[Any]]) -> "FrameDataset[Any]":
        """関数をデータセット全体に適用して新しいデータセットを作成します。"""
        new_dataset = type(self)(
            folder_path=str(self.folder_path),
            lazy_loading=True,
            source_dataset=self,
            transform=func,
            sampling_rate=self.sampling_rate,
            signal_length=self.signal_length,
            file_extensions=self.file_extensions,
            recursive=self._recursive,
        )
        return cast("FrameDataset[Any]", new_dataset)

    def save(self, output_folder: str, filename_prefix: str = "") -> None:
        """処理済みフレームをファイルに保存します。"""
        raise NotImplementedError("saveメソッドは現在実装されていません。")

    def sample(
        self,
        n: Optional[int] = None,
        ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "FrameDataset[F]":
        """データセットからサンプルを取得します。"""
        if seed is not None:
            random.seed(seed)

        total = len(self._lazy_frames)
        if total == 0:
            return type(self)(
                str(self.folder_path),
                sampling_rate=self.sampling_rate,
                signal_length=self.signal_length,
                file_extensions=self.file_extensions,
                lazy_loading=self._lazy_loading,
                recursive=self._recursive,
            )

        # サンプル数の決定
        if n is None and ratio is None:
            n = max(1, min(10, int(total * 0.1)))
        elif n is None and ratio is not None:
            n = max(1, int(total * ratio))
        elif n is not None:
            n = max(1, n)
        else:
            n = 1

        n = min(n, total)

        # ランダムにインデックスを選択
        sampled_indices = sorted(random.sample(range(total), n))

        return _SampledFrameDataset(self, sampled_indices)

    def get_metadata(self) -> dict[str, Any]:
        """データセットのメタデータを取得します。"""
        actual_sr: Optional[Union[int, float]] = self.sampling_rate
        frame_type_name = "Unknown"

        # ロード済みのフレーム数をカウント
        loaded_count = sum(
            1 for lazy_frame in self._lazy_frames if lazy_frame.is_loaded
        )

        # 最初のフレームからメタデータを取得（可能な場合）
        first_frame: Optional[F] = None
        if len(self._lazy_frames) > 0:
            try:
                if self._lazy_frames[0].is_loaded:
                    first_frame = self._lazy_frames[0].frame

                if first_frame:
                    actual_sr = getattr(
                        first_frame, "sampling_rate", self.sampling_rate
                    )
                    frame_type_name = type(first_frame).__name__
            except Exception as e:
                logger.warning(
                    f"メタデータ取得中に最初のフレームのアクセスでエラー: {e}"
                )

        return {
            "folder_path": str(self.folder_path),
            "file_count": len(self._lazy_frames),
            "loaded_count": loaded_count,
            "target_sampling_rate": self.sampling_rate,
            "actual_sampling_rate": actual_sr,
            "signal_length": self.signal_length,
            "file_extensions": self.file_extensions,
            "lazy_loading": self._lazy_loading,
            "recursive": self._recursive,
            "frame_type": frame_type_name,
            "has_transform": self._transform is not None,
            "is_sampled": isinstance(self, _SampledFrameDataset),
        }


class _SampledFrameDataset(FrameDataset[F]):
    """
    データセットのサブセットを表すクラス。
    元のデータセットから選択されたインデックスのみを含みます。
    """

    def __init__(
        self,
        original_dataset: "FrameDataset[F]",
        sampled_indices: list[int],
    ):
        """
        サンプリングされたデータセットを初期化します。

        Args:
            original_dataset: 元のデータセット
            sampled_indices: 選択されたインデックスのリスト
        """
        # 基底クラスの初期化
        super().__init__(
            folder_path=str(original_dataset.folder_path),
            lazy_loading=True,  # サンプリングされたデータセットは常に遅延ロード
            sampling_rate=original_dataset.sampling_rate,
            signal_length=original_dataset.signal_length,
            file_extensions=original_dataset.file_extensions,
            recursive=original_dataset._recursive,
        )

        # オリジナルデータセットを保持
        self._original_dataset = original_dataset

        # サンプリングされたインデックスのマッピング
        self._original_indices = sampled_indices

        # 元のデータセットからファイルパスを取得して新しいLazyFrameを作成
        original_file_paths = original_dataset._get_file_paths()
        try:
            sampled_file_paths = [original_file_paths[i] for i in sampled_indices]
            self._lazy_frames = [
                LazyFrame(file_path) for file_path in sampled_file_paths
            ]
        except IndexError as e:
            logger.error("サンプリングされたインデックスが元データセットの範囲外です")
            logger.error(f"  元データセットのファイル数: {len(original_file_paths)}")
            logger.error(f"  サンプリングされたインデックス: {sampled_indices}")
            raise IndexError(
                "インデックスが元データセットの範囲外です。元データセット数:"
                f"{len(original_file_paths)}, インデックス: {sampled_indices}"
            ) from e

    def _load_file(self, file_path: Path) -> Optional[F]:
        """このクラスでは、ファイルからの直接ロードは行わず、元のデータセットからロードします。"""
        raise NotImplementedError(
            "_SampledFrameDatasetは直接ファイルをロードしません。"
        )

    def _ensure_loaded(self, index: int) -> Optional[F]:
        """
        サンプリングデータセット内のインデックスに対応するフレームをロードします。
        元のデータセットからフレームを取得します。
        """
        # インデックスの範囲チェック
        if not (0 <= index < len(self._lazy_frames)):
            raise IndexError(
                f"インデックス {index} はサンプルされたデータセットの範囲外です "
                f"(0-{len(self._lazy_frames) - 1})"
            )

        lazy_frame = self._lazy_frames[index]

        # 既にロード済みであればキャッシュから返す
        if lazy_frame.is_loaded:
            return lazy_frame.frame

        # 元のデータセット内のインデックス
        original_index = self._original_indices[index]

        try:
            # 元のデータセットからフレームを取得
            frame = self._original_dataset[original_index]

            # LazyFrameを更新
            lazy_frame.frame = frame
            lazy_frame.is_loaded = True
            lazy_frame.load_attempted = True

            return frame

        except Exception as e:
            logger.error(
                "サンプリングデータセットでのフレームのロード中にエラー"
                f"(インデックス {index}, 元インデックス {original_index}): {str(e)}"
            )
            lazy_frame.frame = None
            lazy_frame.is_loaded = True
            lazy_frame.load_attempted = True
            return None

    @overload
    def apply(self, func: Callable[[F], Optional[F_out]]) -> "FrameDataset[F_out]": ...

    @overload
    def apply(self, func: Callable[[F], Optional[Any]]) -> "FrameDataset[Any]": ...

    def apply(self, func: Callable[[F], Optional[Any]]) -> "FrameDataset[Any]":
        """
        関数をサンプリングデータセット全体に適用します。
        元のデータセットに対して新しい変換として追加され、サンプリングは維持されます。
        """
        # 元のデータセットに変換を適用した新しいデータセット
        transformed_dataset = self._original_dataset.apply(func)

        # 同じサンプリングインデックスで新しいサンプリングデータセットを作成
        return _SampledFrameDataset(transformed_dataset, self._original_indices)


class ChannelFrameDataset(FrameDataset[ChannelFrame]):
    """
    フォルダ内の音声ファイルをChannelFrameとして扱うためのデータセットクラス。
    """

    def __init__(
        self,
        folder_path: str,
        sampling_rate: Optional[int] = None,
        signal_length: Optional[int] = None,
        file_extensions: Optional[list[str]] = None,
        lazy_loading: bool = True,
        recursive: bool = False,
        source_dataset: Optional["FrameDataset[Any]"] = None,
        transform: Optional[Callable[[Any], Optional[ChannelFrame]]] = None,
    ):
        _file_extensions = file_extensions or [
            ".wav",
            ".mp3",
            ".flac",
            ".csv",
        ]

        super().__init__(
            folder_path=folder_path,
            sampling_rate=sampling_rate,
            signal_length=signal_length,
            file_extensions=_file_extensions,
            lazy_loading=lazy_loading,
            recursive=recursive,
            source_dataset=source_dataset,
            transform=transform,
        )

    def _load_file(self, file_path: Path) -> Optional[ChannelFrame]:
        """音声ファイルをロードしてChannelFrameを返します。"""
        try:
            frame = ChannelFrame.from_file(file_path)
            if self.sampling_rate and frame.sampling_rate != self.sampling_rate:
                logger.info(
                    f"ファイル {file_path.name} ({frame.sampling_rate} Hz) を "
                    f"データセットのレート ({self.sampling_rate} Hz)"
                    "にリサンプリングします。"
                )
                frame = frame.resampling(target_sr=self.sampling_rate)
            return frame
        except Exception as e:
            logger.error(
                f"ファイル {file_path} の読み込みまたは初期処理に失敗: {str(e)}"
            )
            return None

    def resample(self, target_sr: int) -> "ChannelFrameDataset":
        """データセット内のすべてのフレームをリサンプリングします。"""

        def _resample_func(frame: ChannelFrame) -> Optional[ChannelFrame]:
            if frame is None:
                return None
            try:
                return frame.resampling(target_sr=target_sr)
            except Exception as e:
                logger.warning(f"リサンプリングエラー (target_sr={target_sr}): {e}")
                return None

        new_dataset = self.apply(_resample_func)
        return cast(ChannelFrameDataset, new_dataset)

    def trim(self, start: float, end: float) -> "ChannelFrameDataset":
        """データセット内のすべてのフレームをトリミングします。"""

        def _trim_func(frame: ChannelFrame) -> Optional[ChannelFrame]:
            if frame is None:
                return None
            try:
                return frame.trim(start=start, end=end)
            except Exception as e:
                logger.warning(f"トリミングエラー (start={start}, end={end}): {e}")
                return None

        new_dataset = self.apply(_trim_func)
        return cast(ChannelFrameDataset, new_dataset)

    def normalize(self, **kwargs: Any) -> "ChannelFrameDataset":
        """データセット内のすべてのフレームを正規化します。"""

        def _normalize_func(frame: ChannelFrame) -> Optional[ChannelFrame]:
            if frame is None:
                return None
            try:
                return frame.normalize(**kwargs)
            except Exception as e:
                logger.warning(f"正規化エラー ({kwargs}): {e}")
                return None

        new_dataset = self.apply(_normalize_func)
        return cast(ChannelFrameDataset, new_dataset)

    def stft(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
    ) -> "SpectrogramFrameDataset":
        """データセット内のすべてのフレームにSTFTを適用します。"""
        _hop = hop_length or n_fft // 4

        def _stft_func(frame: ChannelFrame) -> Optional[SpectrogramFrame]:
            if frame is None:
                return None
            try:
                return frame.stft(
                    n_fft=n_fft,
                    hop_length=_hop,
                    win_length=win_length,
                    window=window,
                )
            except Exception as e:
                logger.warning(f"STFTエラー (n_fft={n_fft}, hop={_hop}): {e}")
                return None

        new_dataset = SpectrogramFrameDataset(
            folder_path=str(self.folder_path),
            lazy_loading=True,
            source_dataset=self,
            transform=_stft_func,
            sampling_rate=self.sampling_rate,
        )
        return new_dataset

    @classmethod
    def from_folder(
        cls,
        folder_path: str,
        sampling_rate: Optional[int] = None,
        file_extensions: Optional[list[str]] = None,
        recursive: bool = False,
        lazy_loading: bool = True,
    ) -> "ChannelFrameDataset":
        """フォルダからChannelFrameDatasetを作成するクラスメソッド。"""
        extensions = (
            file_extensions
            if file_extensions is not None
            else [".wav", ".mp3", ".flac", ".csv"]
        )

        return cls(
            folder_path,
            sampling_rate=sampling_rate,
            file_extensions=extensions,
            lazy_loading=lazy_loading,
            recursive=recursive,
        )


class SpectrogramFrameDataset(FrameDataset[SpectrogramFrame]):
    """
    フォルダ内のスペクトログラムデータをSpectrogramFrameとして扱うデータセットクラス。
    主にChannelFrameDataset.stft()の結果として生成されることを想定。
    """

    def __init__(
        self,
        folder_path: str,
        sampling_rate: Optional[int] = None,
        signal_length: Optional[int] = None,
        file_extensions: Optional[list[str]] = None,
        lazy_loading: bool = True,
        recursive: bool = False,
        source_dataset: Optional["FrameDataset[Any]"] = None,
        transform: Optional[Callable[[Any], Optional[SpectrogramFrame]]] = None,
    ):
        super().__init__(
            folder_path=folder_path,
            sampling_rate=sampling_rate,
            signal_length=signal_length,
            file_extensions=file_extensions,
            lazy_loading=lazy_loading,
            recursive=recursive,
            source_dataset=source_dataset,
            transform=transform,
        )

    def _load_file(self, file_path: Path) -> Optional[SpectrogramFrame]:
        """現在直接ファイルからのロードはサポートされていません。"""
        logger.warning(
            "直接SpectrogramFrameをロードする方法は定義されていません。通常、"
            "ChannelFrameDataset.stft()から作成されます。"
        )
        raise NotImplementedError(
            "直接SpectrogramFrameをロードする方法は定義されていません"
        )

    def plot(self, index: int, **kwargs: Any) -> None:
        """指定インデックスのスペクトログラムをプロットします。"""
        try:
            frame = self._ensure_loaded(index)

            if frame is None:
                logger.warning(
                    f"index {index} はロード/変換に失敗していたためプロットできません。"
                )
                return

            plot_method = getattr(frame, "plot", None)
            if callable(plot_method):
                plot_method(**kwargs)
            else:
                logger.warning(
                    f"フレーム (インデックス {index}, タイプ {type(frame).__name__}) に"
                    f"plotメソッドが実装されていません。"
                )
        except Exception as e:
            logger.error(
                f"インデックス {index} のプロット中にエラーが発生しました: {e}"
            )

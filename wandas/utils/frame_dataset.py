import logging
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, cast, overload

from tqdm.auto import tqdm

from wandas.frames.channel import ChannelFrame
from wandas.frames.spectrogram import SpectrogramFrame

logger = logging.getLogger(__name__)

FrameType = ChannelFrame | SpectrogramFrame
F = TypeVar("F", bound=FrameType)
F_out = TypeVar("F_out", bound=FrameType)


@dataclass
class LazyFrame(Generic[F]):
    """
    A class that encapsulates a frame and its loading state.

    Attributes:
        file_path: File path associated with the frame
        frame: Loaded frame object (None if not loaded)
        is_loaded: Flag indicating if the frame is loaded
        load_attempted: Flag indicating if loading was attempted (for error detection)
    """

    file_path: Path
    frame: F | None = None
    is_loaded: bool = False
    load_attempted: bool = False

    def ensure_loaded(self, loader: Callable[[Path], F | None]) -> F | None:
        """
        Ensures the frame is loaded, loading it if necessary.

        Args:
            loader: Function to load a frame from a file path

        Returns:
            The loaded frame, or None if loading failed
        """
        # Return the current frame if already loaded
        if self.is_loaded:
            return self.frame

        # Attempt to load if not loaded yet
        try:
            self.load_attempted = True
            self.frame = loader(self.file_path)
            self.is_loaded = True
            return self.frame
        except Exception as e:
            logger.error(f"Failed to load file {self.file_path}: {str(e)}")
            self.is_loaded = True  # Loading was attempted
            self.frame = None
            return None

    def reset(self) -> None:
        """
        Reset the frame state.
        """
        self.frame = None
        self.is_loaded = False
        self.load_attempted = False


class FrameDataset(Generic[F], ABC):
    """
    Abstract base dataset class for processing files in a folder.
    Includes lazy loading capability to efficiently handle large datasets.
    Subclasses handle specific frame types (ChannelFrame, SpectrogramFrame, etc.).
    """

    def __init__(
        self,
        folder_path: str,
        sampling_rate: int | None = None,
        signal_length: int | None = None,
        file_extensions: list[str] | None = None,
        lazy_loading: bool = True,
        recursive: bool = False,
        source_dataset: Optional["FrameDataset[Any]"] = None,
        transform: Callable[[Any], F | None] | None = None,
    ):
        self.folder_path = Path(folder_path)
        if source_dataset is None and not self.folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")

        self.sampling_rate = sampling_rate
        self.signal_length = signal_length
        self.file_extensions = file_extensions or [".wav"]
        self._recursive = recursive
        self._lazy_loading = lazy_loading

        # Changed to a list of LazyFrame
        self._lazy_frames: list[LazyFrame[F]] = []

        self._source_dataset = source_dataset
        self._transform = transform

        if self._source_dataset:
            self._initialize_from_source()
        else:
            self._initialize_from_folder()

    def _initialize_from_source(self) -> None:
        """Initialize from a source dataset."""
        if self._source_dataset is None:
            return

        # Copy file paths from source
        file_paths = self._source_dataset._get_file_paths()
        self._lazy_frames = [LazyFrame(file_path) for file_path in file_paths]

        # Inherit other properties
        self.sampling_rate = self.sampling_rate or self._source_dataset.sampling_rate
        self.signal_length = self.signal_length or self._source_dataset.signal_length
        self.file_extensions = (
            self.file_extensions or self._source_dataset.file_extensions
        )
        self._recursive = self._source_dataset._recursive
        self.folder_path = self._source_dataset.folder_path

    def _initialize_from_folder(self) -> None:
        """Initialize from a folder."""
        self._discover_files()
        if not self._lazy_loading:
            self._load_all_files()

    def _discover_files(self) -> None:
        """Discover files in the folder and store them in a list of LazyFrame."""
        file_paths = []
        for ext in self.file_extensions:
            pattern = f"**/*{ext}" if self._recursive else f"*{ext}"
            file_paths.extend(
                sorted(p for p in self.folder_path.glob(pattern) if p.is_file())
            )

        # Remove duplicates and sort
        file_paths = sorted(list(set(file_paths)))

        # Create a list of LazyFrame
        self._lazy_frames = [LazyFrame(file_path) for file_path in file_paths]

    def _load_all_files(self) -> None:
        """Load all files."""
        for i in tqdm(range(len(self._lazy_frames)), desc="Loading/transforming"):
            try:
                self._ensure_loaded(i)
            except Exception as e:
                filepath = self._lazy_frames[i].file_path
                logger.warning(
                    f"Failed to load/transform index {i} ({filepath}): {str(e)}"
                )
        self._lazy_loading = False

    @abstractmethod
    def _load_file(self, file_path: Path) -> F | None:
        """Abstract method to load a frame from a file."""
        pass

    def _load_from_source(self, index: int) -> F | None:
        """Load a frame from the source dataset and transform it if necessary."""
        if self._source_dataset is None or self._transform is None:
            return None

        source_frame = self._source_dataset._ensure_loaded(index)
        if source_frame is None:
            return None

        try:
            return self._transform(source_frame)
        except Exception as e:
            msg = f"Failed to transform index {index}: {str(e)}"
            logger.warning(msg)
            # Also emit to the root logger to improve capture reliability
            # in test runners and across different logging configurations
            logging.getLogger().warning(msg)
            return None

    def _ensure_loaded(self, index: int) -> F | None:
        """Ensure the frame at the given index is loaded."""
        if not (0 <= index < len(self._lazy_frames)):
            raise IndexError(
                f"Index {index} is out of range (0-{len(self._lazy_frames) - 1})"
            )

        lazy_frame = self._lazy_frames[index]

        # Return if already loaded
        if lazy_frame.is_loaded:
            return lazy_frame.frame

        try:
            # Convert from source dataset
            if self._transform and self._source_dataset:
                lazy_frame.load_attempted = True
                frame = self._load_from_source(index)
                lazy_frame.frame = frame
                lazy_frame.is_loaded = True
                return frame
            # Load directly from file
            else:
                return lazy_frame.ensure_loaded(self._load_file)
        except Exception as e:
            f_path = lazy_frame.file_path
            logger.error(
                f"Failed to load or initialize index {index} ({f_path}): {str(e)}"
            )
            lazy_frame.frame = None
            lazy_frame.is_loaded = True
            lazy_frame.load_attempted = True
            return None

    def _get_file_paths(self) -> list[Path]:
        """Get a list of file paths."""
        return [lazy_frame.file_path for lazy_frame in self._lazy_frames]

    def __len__(self) -> int:
        """Return the number of files in the dataset."""
        return len(self._lazy_frames)

    def get_by_label(self, label: str) -> F | None:
        """
        Get a frame by its label (filename).

        Parameters
        ----------
        label : str
            The filename (label) to search for (e.g., 'sample_1.wav').

        Returns
        -------
        Optional[F]
            The frame if found, otherwise None.

        Examples
        --------
        >>> frame = dataset.get_by_label("sample_1.wav")
        >>> if frame:
        ...     print(frame.label)
        """
        # Keep for backward compatibility: return the first match but emit
        # a DeprecationWarning recommending `get_all_by_label`.
        all_matches = self.get_all_by_label(label)
        if len(all_matches) > 0:
            warnings.warn(
                "get_by_label() returns the first matching frame and is deprecated; "
                "use get_all_by_label() to obtain all matches.",
                DeprecationWarning,
                stacklevel=2,
            )
            return all_matches[0]
        return None

    def get_all_by_label(self, label: str) -> list[F]:
        """
        Get all frames matching the given label (filename).

        Parameters
        ----------
        label : str
            The filename (label) to search for (e.g., 'sample_1.wav').

        Returns
        -------
        list[F]
            A list of frames matching the label.
            If none are found, returns an empty list.

        Notes
        -----
        - Search is performed against the filename portion only (i.e. Path.name).
        - Each matched frame will be loaded (triggering lazy load) via `_ensure_loaded`.
        """
        matches: list[F] = []
        for i, lazy_frame in enumerate(self._lazy_frames):
            if lazy_frame.file_path.name == label:
                loaded = self._ensure_loaded(i)
                if loaded is not None:
                    matches.append(loaded)
        return matches

    @overload
    def __getitem__(self, key: int) -> F | None: ...

    @overload
    def __getitem__(self, key: str) -> list[F]: ...

    def __getitem__(self, key: int | str) -> F | None | list[F]:
        """
        Get the frame by index (int) or label (str).

        Parameters
        ----------
        key : int or str
            Index (int) or filename/label (str).

        Returns
        -------
        Optional[F] or list[F]
            If `key` is an int, returns the frame or None. If `key` is a str,
            returns a list of matching frames (may be empty).

        Examples
        --------
        >>> frame = dataset[0]  # by index
        >>> frames = dataset["sample_1.wav"]  # list of matches by filename
        """
        if isinstance(key, int):
            return self._ensure_loaded(key)
        if isinstance(key, str):
            # pandas-like behaviour: return all matches for the label as a list
            return self.get_all_by_label(key)
        raise TypeError(f"Invalid key type: {type(key)}. Must be int or str.")

    @overload
    def apply(self, func: Callable[[F], F_out | None]) -> "FrameDataset[F_out]": ...

    @overload
    def apply(self, func: Callable[[F], Any | None]) -> "FrameDataset[Any]": ...

    def apply(self, func: Callable[[F], Any | None]) -> "FrameDataset[Any]":
        """Apply a function to the entire dataset to create a new dataset."""
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
        """Save processed frames to files."""
        raise NotImplementedError("The save method is not currently implemented.")

    def sample(
        self,
        n: int | None = None,
        ratio: float | None = None,
        seed: int | None = None,
    ) -> "FrameDataset[F]":
        """Get a sample from the dataset."""
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

        # Determine sample size
        if n is None and ratio is None:
            n = max(1, min(10, int(total * 0.1)))
        elif n is None and ratio is not None:
            n = max(1, int(total * ratio))
        elif n is not None:
            n = max(1, n)
        else:
            n = 1

        n = min(n, total)

        # Randomly select indices
        sampled_indices = sorted(random.sample(range(total), n))

        return _SampledFrameDataset(self, sampled_indices)

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata for the dataset."""
        actual_sr: int | float | None = self.sampling_rate
        frame_type_name = "Unknown"

        # Count loaded frames
        loaded_count = sum(
            1 for lazy_frame in self._lazy_frames if lazy_frame.is_loaded
        )

        # Get metadata from the first frame (if possible)
        first_frame: F | None = None
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
                    f"Error accessing the first frame during metadata retrieval: {e}"
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
    A class representing a subset of a dataset.
    Contains only the indices selected from the original dataset.
    """

    def __init__(
        self,
        original_dataset: "FrameDataset[F]",
        sampled_indices: list[int],
    ):
        """
        Initialize a sampled dataset.

        Args:
            original_dataset: The original dataset
            sampled_indices: List of selected indices
        """
        # Initialize base class
        super().__init__(
            folder_path=str(original_dataset.folder_path),
            lazy_loading=True,  # Sampled datasets always use lazy loading
            sampling_rate=original_dataset.sampling_rate,
            signal_length=original_dataset.signal_length,
            file_extensions=original_dataset.file_extensions,
            recursive=original_dataset._recursive,
        )

        # Store the original dataset
        self._original_dataset = original_dataset

        # Mapping of sampled indices
        self._original_indices = sampled_indices

        # Get file paths from the original dataset and create new LazyFrames
        original_file_paths = original_dataset._get_file_paths()
        try:
            sampled_file_paths = [original_file_paths[i] for i in sampled_indices]
            self._lazy_frames = [
                LazyFrame(file_path) for file_path in sampled_file_paths
            ]
        except IndexError as e:
            logger.error("Sampled indices are out of range for the original dataset")
            logger.error(f"  Original dataset file count: {len(original_file_paths)}")
            logger.error(f"  Sampled indices: {sampled_indices}")
            raise IndexError(
                "Indices are out of range for the original dataset. Original dataset count: "  # noqa: E501
                f"{len(original_file_paths)}, indices: {sampled_indices}"
            ) from e

    def _load_file(self, file_path: Path) -> F | None:
        """This class does not load directly from files but from the original dataset."""  # noqa: E501
        raise NotImplementedError("_SampledFrameDataset does not load files directly.")

    def _ensure_loaded(self, index: int) -> F | None:
        """
        Load the frame corresponding to the index in the sampled dataset.
        Get the frame from the original dataset.
        """
        # Check index range
        if not (0 <= index < len(self._lazy_frames)):
            raise IndexError(
                f"Index {index} is out of range for the sampled dataset "
                f"(0-{len(self._lazy_frames) - 1})"
            )

        lazy_frame = self._lazy_frames[index]

        # Return from cache if already loaded
        if lazy_frame.is_loaded:
            return lazy_frame.frame

        # Index in the original dataset
        original_index = self._original_indices[index]

        try:
            # Get frame from the original dataset
            frame = self._original_dataset[original_index]

            # Update LazyFrame
            lazy_frame.frame = frame
            lazy_frame.is_loaded = True
            lazy_frame.load_attempted = True

            return frame

        except Exception as e:
            logger.error(
                "Error loading frame in sampled dataset "
                f"(index {index}, original index {original_index}): {str(e)}"
            )
            lazy_frame.frame = None
            lazy_frame.is_loaded = True
            lazy_frame.load_attempted = True
            return None

    @overload
    def apply(self, func: Callable[[F], F_out | None]) -> "FrameDataset[F_out]": ...

    @overload
    def apply(self, func: Callable[[F], Any | None]) -> "FrameDataset[Any]": ...

    def apply(self, func: Callable[[F], Any | None]) -> "FrameDataset[Any]":
        """
        Apply a function to the entire sampled dataset.
        Added as a new transformation to the original dataset, maintaining sampling.
        """
        # Apply transformation to the original dataset
        transformed_dataset = self._original_dataset.apply(func)

        # Create a new sampled dataset with the same sampling indices
        return _SampledFrameDataset(transformed_dataset, self._original_indices)


class ChannelFrameDataset(FrameDataset[ChannelFrame]):
    """
    Dataset class for handling audio files as ChannelFrames in a folder.
    """

    def __init__(
        self,
        folder_path: str,
        sampling_rate: int | None = None,
        signal_length: int | None = None,
        file_extensions: list[str] | None = None,
        lazy_loading: bool = True,
        recursive: bool = False,
        source_dataset: Optional["FrameDataset[Any]"] = None,
        transform: Callable[[Any], ChannelFrame | None] | None = None,
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

    def _load_file(self, file_path: Path) -> ChannelFrame | None:
        """Load an audio file and return a ChannelFrame."""
        try:
            frame = ChannelFrame.from_file(file_path)
            if self.sampling_rate and frame.sampling_rate != self.sampling_rate:
                logger.info(
                    f"Resampling file {file_path.name} ({frame.sampling_rate} Hz) to "
                    f"dataset rate ({self.sampling_rate} Hz)."
                )
                frame = frame.resampling(target_sr=self.sampling_rate)
            return frame
        except Exception as e:
            logger.error(f"Failed to load or initialize file {file_path}: {str(e)}")
            return None

    def resample(self, target_sr: int) -> "ChannelFrameDataset":
        """Resample all frames in the dataset."""

        def _resample_func(frame: ChannelFrame) -> ChannelFrame | None:
            if frame is None:
                return None
            try:
                return frame.resampling(target_sr=target_sr)
            except Exception as e:
                logger.warning(f"Resampling error (target_sr={target_sr}): {e}")
                return None

        new_dataset = self.apply(_resample_func)
        return cast(ChannelFrameDataset, new_dataset)

    def trim(self, start: float, end: float) -> "ChannelFrameDataset":
        """Trim all frames in the dataset."""

        def _trim_func(frame: ChannelFrame) -> ChannelFrame | None:
            if frame is None:
                return None
            try:
                return frame.trim(start=start, end=end)
            except Exception as e:
                logger.warning(f"Trimming error (start={start}, end={end}): {e}")
                return None

        new_dataset = self.apply(_trim_func)
        return cast(ChannelFrameDataset, new_dataset)

    def normalize(self, **kwargs: Any) -> "ChannelFrameDataset":
        """Normalize all frames in the dataset."""

        def _normalize_func(frame: ChannelFrame) -> ChannelFrame | None:
            if frame is None:
                return None
            try:
                return frame.normalize(**kwargs)
            except Exception as e:
                logger.warning(f"Normalization error ({kwargs}): {e}")
                return None

        new_dataset = self.apply(_normalize_func)
        return cast(ChannelFrameDataset, new_dataset)

    def stft(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
    ) -> "SpectrogramFrameDataset":
        """Apply STFT to all frames in the dataset."""
        _hop = hop_length or n_fft // 4

        def _stft_func(frame: ChannelFrame) -> SpectrogramFrame | None:
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
                logger.warning(f"STFT error (n_fft={n_fft}, hop={_hop}): {e}")
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
        sampling_rate: int | None = None,
        file_extensions: list[str] | None = None,
        recursive: bool = False,
        lazy_loading: bool = True,
    ) -> "ChannelFrameDataset":
        """Class method to create a ChannelFrameDataset from a folder."""
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
    Dataset class for handling spectrogram data as SpectrogramFrames.
    Expected to be generated mainly as a result of ChannelFrameDataset.stft().
    """

    def __init__(
        self,
        folder_path: str,
        sampling_rate: int | None = None,
        signal_length: int | None = None,
        file_extensions: list[str] | None = None,
        lazy_loading: bool = True,
        recursive: bool = False,
        source_dataset: Optional["FrameDataset[Any]"] = None,
        transform: Callable[[Any], SpectrogramFrame | None] | None = None,
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

    def _load_file(self, file_path: Path) -> SpectrogramFrame | None:
        """Direct loading from files is not currently supported."""
        logger.warning(
            "No method defined for directly loading SpectrogramFrames. Normally "
            "created from ChannelFrameDataset.stft()."
        )
        raise NotImplementedError(
            "No method defined for directly loading SpectrogramFrames"
        )

    def plot(self, index: int, **kwargs: Any) -> None:
        """Plot the spectrogram at the specified index."""
        try:
            frame = self._ensure_loaded(index)

            if frame is None:
                logger.warning(
                    f"Cannot plot index {index} as it failed to load/transform."
                )
                return

            plot_method = getattr(frame, "plot", None)
            if callable(plot_method):
                plot_method(**kwargs)
            else:
                logger.warning(
                    f"Frame (index {index}, type {type(frame).__name__}) does not "
                    f"have a plot method implemented."
                )
        except Exception as e:
            logger.error(f"An error occurred while plotting index {index}: {e}")

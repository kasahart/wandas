import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import soundfile as sf
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class FileReader(ABC):
    """Base class for audio file readers."""

    # Class attribute for supported file extensions
    supported_extensions: list[str] = []

    @classmethod
    @abstractmethod
    def get_file_info(cls, path: Union[str, Path], **kwargs: Any) -> dict[str, Any]:
        """Get basic information about the audio file.

        Args:
            path: Path to the file.
            **kwargs: Additional parameters specific to the file reader.

        Returns:
            Dictionary containing file information including:
            - samplerate: Sampling rate in Hz
            - channels: Number of channels
            - frames: Total number of frames
            - format: File format
            - duration: Duration in seconds
        """
        pass

    @classmethod
    @abstractmethod
    def get_data(
        cls,
        path: Union[str, Path],
        channels: list[int],
        start_idx: int,
        frames: int,
        **kwargs: Any,
    ) -> ArrayLike:
        """Read audio data from the file.

        Args:
            path: Path to the file.
            channels: List of channel indices to read.
            start_idx: Starting frame index.
            frames: Number of frames to read.
            **kwargs: Additional parameters specific to the file reader.

        Returns:
            Array of shape (channels, frames) containing the audio data.
        """
        pass

    @classmethod
    def can_read(cls, path: Union[str, Path]) -> bool:
        """Check if this reader can handle the file based on extension."""
        ext = Path(path).suffix.lower()
        return ext in cls.supported_extensions


class SoundFileReader(FileReader):
    """Audio file reader using SoundFile library."""

    # SoundFile supported formats
    supported_extensions = [".wav", ".flac", ".ogg", ".aiff", ".aif", ".snd"]

    @classmethod
    def get_file_info(cls, path: Union[str, Path], **kwargs: Any) -> dict[str, Any]:
        """Get basic information about the audio file."""
        info = sf.info(str(path))
        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
            "duration": info.frames / info.samplerate,
        }

    @classmethod
    def get_data(
        cls,
        path: Union[str, Path],
        channels: list[int],
        start_idx: int,
        frames: int,
        **kwargs: Any,
    ) -> ArrayLike:
        """Read audio data from the file."""
        logger.debug(f"Reading {frames} frames from {path} starting at {start_idx}")

        with sf.SoundFile(str(path)) as f:
            if start_idx > 0:
                f.seek(start_idx)
            data = f.read(frames=frames, dtype="float32", always_2d=True)

            # Select requested channels
            if len(channels) < f.channels:
                data = data[:, channels]

            # Transpose to get (channels, samples) format
            result: ArrayLike = data.T
            if not isinstance(result, np.ndarray):
                raise ValueError("Unexpected data type after reading file")

        _shape = result.shape
        logger.debug(f"File read complete, returning data with shape {_shape}")
        return result


class CSVFileReader(FileReader):
    """CSV file reader for time series data."""

    # CSV supported formats
    supported_extensions = [".csv"]

    @classmethod
    def get_file_info(  # type: ignore[override]
        cls,
        path: Union[str, Path],
        delimiter: str = ",",
        header: Optional[int] = 0,
        time_column: Union[int, str] = 0,
    ) -> dict[str, Any]:
        """Get basic information about the CSV file.

        Args:
            path: Path to the CSV file.
            delimiter: Delimiter character. Default is ",".
            header: Row number to use as header. Default is 0 (first row).
                Set to None if no header.
            time_column: Index or name of the time column. Default is 0.

        Returns:
            Dictionary containing file information.
        """
        # Read first few lines to determine structure
        df = pd.read_csv(path, delimiter=delimiter, header=header)

        # Estimate sampling rate from first column (assuming it's time)
        try:
            # Get time column as Series
            if isinstance(time_column, str):
                time_series = df[time_column]
            else:
                time_series = df.iloc[:, time_column]
            time_values = np.array(time_series.values)
            if len(time_values) > 1:
                estimated_sr = int(1 / np.mean(np.diff(time_values)))
            else:
                estimated_sr = 0  # Cannot determine from single row
        except Exception:
            estimated_sr = 0  # Default if can't calculate

        frames = df.shape[0]
        duration = frames / estimated_sr if estimated_sr > 0 else None

        # Return file info
        return {
            "samplerate": estimated_sr,
            "channels": df.shape[1] - 1,  # Assuming first column is time
            "frames": frames,
            "format": "CSV",
            "duration": duration,
            "ch_labels": df.columns[1:].tolist(),  # Assuming first column is time
        }

    @classmethod
    def get_data(  # type: ignore[override]
        cls,
        path: Union[str, Path],
        channels: list[int],
        start_idx: int,
        frames: int,
        time_column: Union[int, str] = 0,
        delimiter: str = ",",
        header: Optional[int] = 0,
    ) -> ArrayLike:
        """Read data from the CSV file.

        Args:
            path: Path to the CSV file.
            channels: List of channel indices to read.
            start_idx: Starting frame index.
            frames: Number of frames to read.
            time_column: Index or name of the time column. Default is 0.
            delimiter: Delimiter character. Default is ",".
            header: Row number to use as header. Default is 0.

        Returns:
            Array of shape (channels, frames) containing the data.
        """
        logger.debug(f"Reading CSV data from {path} starting at {start_idx}")

        # Read the CSV file
        df = pd.read_csv(path, delimiter=delimiter, header=header)

        # Remove time column
        df = df.drop(
            columns=[time_column]
            if isinstance(time_column, str)
            else df.columns[time_column]
        )

        # Select requested channels - adjust indices to account for time column removal
        if channels:
            try:
                data_df = df.iloc[:, channels]
            except IndexError:
                raise ValueError(f"Requested channels {channels} out of range")
        else:
            data_df = df

        # Handle start_idx and frames for partial reading
        end_idx = start_idx + frames if frames > 0 else None
        data_df = data_df.iloc[start_idx:end_idx]

        # Convert to numpy array and transpose to (channels, samples) format
        result = data_df.values.T

        if not isinstance(result, np.ndarray):
            raise ValueError("Unexpected data type after reading file")

        _shape = result.shape
        logger.debug(f"CSV read complete, returning data with shape {_shape}")
        return result


# Registry of available file readers
_file_readers = [SoundFileReader(), CSVFileReader()]


def get_file_reader(path: Union[str, Path]) -> FileReader:
    """Get an appropriate file reader for the given path."""
    path_str = str(path)
    ext = Path(path).suffix.lower()

    # Try each reader in order
    for reader in _file_readers:
        if ext in reader.__class__.supported_extensions:
            logger.debug(f"Using {reader.__class__.__name__} for {path_str}")
            return reader

    # If no reader found, raise error
    raise ValueError(f"No suitable file reader found for {path_str}")


def register_file_reader(reader_class: type) -> None:
    """Register a new file reader."""
    reader = reader_class()
    _file_readers.append(reader)
    logger.debug(f"Registered new file reader: {reader_class.__name__}")

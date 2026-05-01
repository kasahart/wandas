import io
import logging
import tempfile
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, TypedDict, cast

import numpy as np
import pandas as pd
import soundfile as sf
from numpy.typing import ArrayLike
from scipy.io import wavfile

logger = logging.getLogger(__name__)

URL_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_URL_DOWNLOAD_BYTES = 256 * 1024 * 1024


@dataclass
class DownloadedTemporaryFile:
    """Temporary file created for streamed URL downloads."""

    path: Path
    temp_dir: tempfile.TemporaryDirectory[str]

    def __post_init__(self) -> None:
        self._finalizer = weakref.finalize(
            self,
            tempfile.TemporaryDirectory.cleanup,
            self.temp_dir,
        )

    def __enter__(self) -> "DownloadedTemporaryFile":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._finalizer.alive:
            self._finalizer()


class CSVFileInfoParams(TypedDict, total=False):
    """Type definition for CSV file reader parameters in get_file_info.

    Parameters
    ----------
    delimiter : str
        Delimiter character. Default is ",".
    header : Optional[int]
        Row number to use as header. Default is 0 (first row).
        Set to None if no header.
    time_column : Union[int, str]
        Index or name of the time column. Default is 0.
    """

    delimiter: str
    header: int | None
    time_column: int | str


class CSVGetDataParams(TypedDict, total=False):
    """Type definition for CSV file reader parameters in get_data.

    Parameters
    ----------
    delimiter : str
        Delimiter character. Default is ",".
    header : Optional[int]
        Row number to use as header. Default is 0.
    time_column : Union[int, str]
        Index or name of the time column. Default is 0.
    """

    delimiter: str
    header: int | None
    time_column: int | str


class FileReader(ABC):
    """Base class for audio file readers."""

    # Class attribute for supported file extensions
    supported_extensions: ClassVar[list[str]] = []

    @classmethod
    @abstractmethod
    def get_file_info(
        cls,
        path: str | Path | bytes | bytearray | memoryview | BinaryIO,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
        # pragma: no cover

    @classmethod
    @abstractmethod
    def get_data(
        cls,
        path: str | Path | bytes | bytearray | memoryview | BinaryIO,
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
        # pragma: no cover

    @classmethod
    def can_read(cls, path: str | Path) -> bool:
        """Check if this reader can handle the file based on extension."""
        ext = Path(path).suffix.lower()
        return ext in cls.supported_extensions


class SoundFileReader(FileReader):
    """Audio file reader using SoundFile library."""

    # SoundFile supported formats
    supported_extensions: ClassVar[list[str]] = [".wav", ".flac", ".ogg", ".aiff", ".aif", ".snd"]

    @classmethod
    def get_file_info(
        cls,
        path: str | Path | bytes | bytearray | memoryview | BinaryIO,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get basic information about the audio file."""
        info = sf.info(_prepare_file_source(path))
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
        path: str | Path | bytes | bytearray | memoryview | BinaryIO,
        channels: list[int],
        start_idx: int,
        frames: int,
        normalize: bool = False,
        **kwargs: Any,
    ) -> ArrayLike:
        """Read audio data from the file.

        Args:
            normalize: When False (default) and the source is a WAV file path,
                return raw integer PCM samples cast to float32 via
                scipy.io.wavfile.read. For non-WAV formats or in-memory sources,
                always uses soundfile (returning float32 normalized to [-1.0, 1.0]).
                When True, return float32 data normalized to [-1.0, 1.0] via soundfile.
        """
        logger.debug(f"Reading {frames} frames from {path!r} starting at {start_idx}")

        is_wav = isinstance(path, (str, Path)) and Path(path).suffix.lower() == ".wav"
        if not normalize and is_wav:
            # Use scipy to return raw integer samples (no normalization), cast to float32.
            source = _prepare_file_source(path)
            _sr, raw = wavfile.read(source)
            raw = np.expand_dims(raw, axis=0) if raw.ndim == 1 else raw.T

            # Only reindex channels when the requested selection is not the identity.
            if channels != list(range(raw.shape[0])):
                raw = raw[channels]

            result: ArrayLike = raw[:, start_idx : start_idx + frames].astype(
                np.float32,
                copy=False,
            )
            if not isinstance(result, np.ndarray):
                raise ValueError("Unexpected data type after reading file")
            logger.debug(f"File read complete (raw), returning data with shape {result.shape}")
            return result

        with sf.SoundFile(_prepare_file_source(path)) as f:
            if start_idx > 0:
                f.seek(start_idx)
            data = f.read(frames=frames, dtype="float32", always_2d=True)

            # Select requested channels
            data = data[:, channels]

            # Transpose to get (channels, samples) format
            result = data.T
            if not isinstance(result, np.ndarray):
                raise ValueError("Unexpected data type after reading file")

        _shape = result.shape
        logger.debug(f"File read complete, returning data with shape {_shape}")
        return result


class CSVFileReader(FileReader):
    """CSV file reader for time series data."""

    # CSV supported formats
    supported_extensions: ClassVar[list[str]] = [".csv"]

    @classmethod
    def get_file_info(
        cls,
        path: str | Path | bytes | bytearray | memoryview | BinaryIO,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get basic information about the CSV file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the CSV file.
        **kwargs : Any
            Additional parameters for CSV reading. Supported parameters:

            - delimiter : str, default=","
                Delimiter character.
            - header : Optional[int], default=0
                Row number to use as header. Set to None if no header.
            - time_column : Union[int, str], default=0
                Index or name of the time column.

        Returns
        -------
        dict[str, Any]
            Dictionary containing file information including:
            - samplerate: Estimated sampling rate in Hz
            - channels: Number of data channels (excluding time column)
            - frames: Total number of frames
            - format: "CSV"
            - duration: Duration in seconds (or None if cannot be calculated)
            - ch_labels: List of channel labels

        Notes
        -----
        This method accepts CSV-specific parameters through kwargs.
        See CSVFileInfoParams for supported parameter types.
        """
        # Extract parameters with defaults
        delimiter: str = kwargs.get("delimiter", ",")
        header: int | None = kwargs.get("header", 0)
        time_column: int | str = kwargs.get("time_column", 0)

        # Read first few lines to determine structure
        df = pd.read_csv(_prepare_file_source(path), delimiter=delimiter, header=header)

        # Estimate sampling rate from first column (assuming it's time)
        try:
            # Get time column as Series
            time_series = df[time_column] if isinstance(time_column, str) else df.iloc[:, time_column]
            time_values = np.array(time_series.values)
            if len(time_values) > 1:
                # Use round() instead of int() to handle floating-point precision issues
                estimated_sr = round(1 / np.mean(np.diff(time_values)))
            else:
                estimated_sr = 0  # Cannot determine from single row
        except Exception:
            estimated_sr = 0  # Default if can't calculate

        frames = df.shape[0]
        duration = frames / estimated_sr if estimated_sr else None

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
    def get_data(
        cls,
        path: str | Path | bytes | bytearray | memoryview | BinaryIO,
        channels: list[int],
        start_idx: int,
        frames: int,
        **kwargs: Any,
    ) -> ArrayLike:
        """Read data from the CSV file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the CSV file.
        channels : list[int]
            List of channel indices to read.
        start_idx : int
            Starting frame index.
        frames : int
            Number of frames to read.
        **kwargs : Any
            Additional parameters for CSV reading. Supported parameters:

            - delimiter : str, default=","
                Delimiter character.
            - header : Optional[int], default=0
                Row number to use as header.
            - time_column : Union[int, str], default=0
                Index or name of the time column.

        Returns
        -------
        ArrayLike
            Array of shape (channels, frames) containing the data.

        Notes
        -----
        This method accepts CSV-specific parameters through kwargs.
        See CSVGetDataParams for supported parameter types.
        """
        # Extract parameters with defaults
        time_column: int | str = kwargs.get("time_column", 0)
        delimiter: str = kwargs.get("delimiter", ",")
        header: int | None = kwargs.get("header", 0)

        logger.debug(f"Reading CSV data from {path!r} starting at {start_idx}")

        # Read the CSV file
        df = pd.read_csv(_prepare_file_source(path), delimiter=delimiter, header=header)

        # Remove time column
        df = df.drop(columns=[time_column] if isinstance(time_column, str) else df.columns[time_column])

        # Select requested channels - adjust indices to account for time column removal
        if channels:
            try:
                data_df = df.iloc[:, channels]
            except IndexError as e:
                raise ValueError(f"Requested channels {channels} out of range") from e
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


def _normalize_extension(file_type: str | None) -> str | None:
    if not file_type:
        return None
    ext = file_type.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _get_validated_content_length_or_none(
    response: Any,
    *,
    url: str,
    resource_name: str,
) -> int | None:
    headers = getattr(response, "headers", {})
    raw_value = headers.get("Content-Length") if hasattr(headers, "get") else None
    if raw_value in (None, ""):
        return None

    try:
        content_length = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise OSError(
            f"Invalid Content-Length for {resource_name} download\n"
            f"  URL: {url}\n"
            f"  Content-Length: {raw_value!r}\n"
            f"Use a valid URL or download the file locally before loading."
        ) from exc

    if content_length < 0:
        raise OSError(
            f"Invalid Content-Length for {resource_name} download\n"
            f"  URL: {url}\n"
            f"  Content-Length: {content_length}\n"
            f"Use a valid URL or download the file locally before loading."
        )
    return content_length


def download_url_to_temporary_file(
    url: str,
    *,
    timeout: float,
    suffix: str | None = None,
    resource_name: str = "file",
    max_bytes: int | None = None,
    chunk_size: int | None = None,
) -> DownloadedTemporaryFile:
    import urllib.error
    import urllib.request

    effective_max_bytes = MAX_URL_DOWNLOAD_BYTES if max_bytes is None else max_bytes
    effective_chunk_size = URL_DOWNLOAD_CHUNK_SIZE if chunk_size is None else chunk_size
    if effective_max_bytes <= 0:
        raise ValueError(
            f"Download size limit must be greater than zero\n"
            f"  Resource: {resource_name}\n"
            f"  URL: {url}\n"
            f"  Got: {effective_max_bytes} bytes\n"
            f"Provide a positive max_bytes value."
        )
    if effective_chunk_size <= 0:
        raise ValueError(
            f"Download chunk size must be greater than zero\n"
            f"  Resource: {resource_name}\n"
            f"  URL: {url}\n"
            f"  Got: {effective_chunk_size} bytes\n"
            f"Provide a positive chunk_size value."
        )
    normalized_suffix = _normalize_extension(suffix) or ""
    downloaded_bytes = 0
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    downloaded_file: DownloadedTemporaryFile | None = None

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content_length = _get_validated_content_length_or_none(
                response,
                url=url,
                resource_name=resource_name,
            )
            if content_length is not None and content_length > effective_max_bytes:
                raise OSError(
                    f"Declared size of {resource_name} exceeds download limit\n"
                    f"  URL: {url}\n"
                    f"  Declared size: {content_length} bytes\n"
                    f"  Limit: {effective_max_bytes} bytes\n"
                    f"Use a smaller file or download it locally before loading."
                )

            temp_dir = tempfile.TemporaryDirectory()
            temp_path = Path(temp_dir.name) / f"download{normalized_suffix}"
            downloaded_file = DownloadedTemporaryFile(path=temp_path, temp_dir=temp_dir)
            with temp_path.open("wb") as temp_file:
                while True:
                    chunk = response.read(effective_chunk_size)
                    if not chunk:
                        break
                    next_downloaded_bytes = downloaded_bytes + len(chunk)
                    if next_downloaded_bytes > effective_max_bytes:
                        raise OSError(
                            f"Streaming {resource_name} would exceed size limit\n"
                            f"  URL: {url}\n"
                            f"  Attempted size: {next_downloaded_bytes} bytes\n"
                            f"  Limit: {effective_max_bytes} bytes\n"
                            f"Use a smaller file or download it locally before loading."
                        )
                    downloaded_bytes = next_downloaded_bytes
                    temp_file.write(chunk)
        assert downloaded_file is not None
        return downloaded_file
    except urllib.error.URLError as exc:
        raise OSError(
            f"Failed to download {resource_name} from URL\n"
            f"  URL: {url}\n"
            f"  Error: {exc}\n"
            f"Verify the URL is accessible and try again."
        ) from exc
    except Exception:
        if downloaded_file is not None:
            downloaded_file.cleanup()
        elif temp_dir is not None:
            temp_dir.cleanup()
        raise


def _prepare_file_source(
    source: str | Path | bytes | bytearray | memoryview | BinaryIO,
) -> str | BinaryIO:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(source))
    if hasattr(source, "read"):
        file_obj = cast(BinaryIO, source)
        try:
            file_obj.seek(0)
        except Exception:
            # Some file-like objects are not seekable or may reject seek(0).
            # In that case, continue using the current position without failing.
            logger.debug(
                "Could not seek to start of file-like object; continuing from current position",
                exc_info=True,
            )
        return file_obj
    return str(source)


def get_file_reader(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    *,
    file_type: str | None = None,
) -> FileReader:
    """Get an appropriate file reader for the given path or file type."""
    path_str = str(path)
    ext = _normalize_extension(file_type)
    if ext is None and isinstance(path, (str, Path)):
        ext = Path(path).suffix.lower()
    if not ext:
        raise ValueError(
            "File type is required when the extension is missing\n"
            "  Cannot determine format without an extension\n"
            "  Provide file_type like '.wav' or '.csv'"
        )

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

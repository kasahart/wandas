import copy
import logging
import numbers
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from re import Pattern
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes
from pydantic import ValidationError

from wandas.utils.types import NDArrayComplex, NDArrayReal

from .metadata import ChannelMetadata, FrameMetadata

# IPython display types for visualize_graph return type
# TYPE_CHECKING ブロック内では型エイリアスとして定義し、実行時は Any とする
if TYPE_CHECKING:
    from typing import TypeAlias

    from IPython.display import Image as IPythonImage

    VisualizeReturnType: TypeAlias = IPythonImage | None
else:
    # 実行時は Any として扱い、mypy のエラーを回避
    VisualizeReturnType = Any

logger = logging.getLogger(__name__)

T = TypeVar("T", NDArrayComplex, NDArrayReal)
S = TypeVar("S", bound="BaseFrame[Any]")
S_Out = TypeVar("S_Out", bound="BaseFrame[Any]")
QueryType = str | Pattern[str] | Callable[["ChannelMetadata"], bool] | dict[str, Any]


class BaseFrame(ABC, Generic[T]):
    """
    Abstract base class for all signal frame types.

    This class provides the common interface and functionality for all frame types
    used in signal processing. It implements basic operations like indexing, iteration,
    and data manipulation that are shared across all frame types.

    Parameters
    ----------
    data : DaArray
        The signal data to process. Must be a dask array.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    label : str, optional
        A label for the frame. If not provided, defaults to "unnamed_frame".
    metadata : FrameMetadata | dict, optional
        Additional metadata for the frame. Plain dicts are automatically
        converted to FrameMetadata.
    operation_history : list[dict], optional
        History of operations performed on this frame.
    channel_metadata : list[ChannelMetadata | dict], optional
        Metadata for each channel in the frame. Can be ChannelMetadata objects
        or dicts that will be validated by Pydantic.
    previous : BaseFrame, optional
        The frame that this frame was derived from.

    Attributes
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    label : str
        The label of the frame.
    metadata : FrameMetadata
        Additional metadata for the frame.
    operation_history : list[dict]
        History of operations performed on this frame.
    """

    _data: DaArray
    sampling_rate: float
    label: str
    metadata: FrameMetadata
    operation_history: list[dict[str, Any]]
    _channel_metadata: list[ChannelMetadata]
    _previous: "BaseFrame[Any] | None"

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: str | None = None,
        metadata: "FrameMetadata | dict[str, Any] | None" = None,
        operation_history: list[dict[str, Any]] | None = None,
        channel_metadata: list[ChannelMetadata] | list[dict[str, Any]] | None = None,
        previous: "BaseFrame[Any] | None" = None,
    ):
        # Default rechunk: prefer channel-wise chunking so the 0th axis
        # (channels) will be processed per-channel for parallelism.
        # For (channels, samples) arrays use (1, -1). For spectrograms
        # and higher-dim arrays (channels, ..) we preserve channel-wise
        # first-axis chunking: (1, -1, -1, ...)
        try:
            # 正規化：data が 1D の場合は (1, -1) にする
            if data.ndim == 1:
                normalized = data.reshape((1, -1))
            else:
                normalized = data

            # チャンク設定：常に先頭軸をチャネル単位にし、残りはフラット
            if normalized.ndim >= 2:
                chunks = tuple([1] + [-1] * (normalized.ndim - 1))
            else:
                chunks = tuple([-1] * normalized.ndim)

            self._data = normalized.rechunk(chunks)
        except Exception as e:
            # Fall back to previous behavior if Dask rechunk fails.
            logger.warning(f"Rechunk failed: {e!r}. Falling back to chunks=-1.")
            self._data = data.rechunk(chunks=-1)  # type: ignore [unused-ignore]

        self.sampling_rate = sampling_rate
        self.label = label or "unnamed_frame"
        if isinstance(metadata, FrameMetadata):
            self.metadata = metadata
        elif metadata is not None:
            self.metadata = FrameMetadata(metadata)
        else:
            self.metadata = FrameMetadata()
        self.operation_history = operation_history or []
        self._previous = previous

        if channel_metadata:
            # Pydantic handles both ChannelMetadata objects and dicts
            def _to_channel_metadata(ch: ChannelMetadata | dict[str, Any], index: int) -> ChannelMetadata:
                if isinstance(ch, ChannelMetadata):
                    return copy.deepcopy(ch)
                elif isinstance(ch, dict):
                    try:
                        return ChannelMetadata(**ch)
                    except ValidationError as e:
                        raise ValueError(
                            f"Invalid channel_metadata at index {index}\n"
                            f"  Got: {ch}\n"
                            f"  Validation error: {e}\n"
                            f"Ensure all dict keys match ChannelMetadata fields "
                            f"(label, unit, ref, extra) and have correct types."
                        ) from e
                else:
                    raise TypeError(
                        f"Invalid type in channel_metadata at index {index}\n"
                        f"  Got: {type(ch).__name__} ({ch!r})\n"
                        f"  Expected: ChannelMetadata or dict\n"
                        f"Use ChannelMetadata objects or dicts with valid fields."
                    )

            self._channel_metadata = [
                _to_channel_metadata(cast(ChannelMetadata | dict[str, Any], ch), i)
                for i, ch in enumerate(channel_metadata)
            ]
        else:
            self._channel_metadata = [
                ChannelMetadata(label=f"ch{i}", unit="", extra={}) for i in range(self._n_channels)
            ]

        try:
            # Display information for newer dask versions
            logger.debug(f"Dask graph layers: {list(self._data.dask.layers.keys())}")
            logger.debug(f"Dask graph dependencies: {len(self._data.dask.dependencies)}")
        except Exception as e:
            logger.debug(f"Dask graph visualization details unavailable: {e}")

    @property
    @abstractmethod
    def _n_channels(self) -> int:
        """Returns the number of channels."""

    @property
    def n_channels(self) -> int:
        """Returns the number of channels."""
        return self._n_channels

    @property
    def channels(self) -> list[ChannelMetadata]:
        """Property to access channel metadata."""
        return self._channel_metadata

    @property
    def previous(self) -> "BaseFrame[Any] | None":
        """
        Returns the previous frame.
        """
        return self._previous

    def get_channel(
        self: S,
        channel_idx: int | list[int] | tuple[int, ...] | npt.NDArray[np.int_] | npt.NDArray[np.bool_] | None = None,
        query: QueryType | None = None,
        validate_query_keys: bool = True,
    ) -> S:
        """
        Get channel(s) by index.

        Parameters
        ----------
        channel_idx : int or sequence of int
            Single channel index or sequence of channel indices.
            Supports negative indices (e.g., -1 for the last channel).
        query : str, re.Pattern, callable, or dict, optional
            If a query is provided, use it to derive indices and ignore the positional channel_idx argument.
            Query to select channels based on metadata. Supported types:
            - str: exact label match
            - re.Pattern: regex search against label
            - callable(ChannelMetadata) -> bool: predicate on channel metadata
            - dict: attribute equality on ChannelMetadata (values may be re.Pattern)
        validate_query_keys : bool, default True
            If True (default), dict queries that contain unknown keys (neither
            model fields nor any channel `extra` keys) will raise `KeyError`.
            Set to False to disable this strict validation and allow callers
            to attempt matches without pre-validation.
        Returns
        -------
        S
            New instance containing the selected channel(s).

        Examples
        --------
        >>> frame.get_channel(0)  # Single channel
        >>> frame.get_channel([0, 2, 3])  # Multiple channels
        >>> frame.get_channel((-1, -2))  # Last two channels
        >>> frame.get_channel(np.array([1, 2]))  # NumPy array of indices
        """  # noqa: E501

        def _indices_from_query(q: Any) -> list[int]:
            if isinstance(q, str):
                return [i for i, ch in enumerate(self._channel_metadata) if ch.label == q]

            # re.Pattern compatibility
            if hasattr(q, "search") and callable(q.search):
                return [i for i, ch in enumerate(self._channel_metadata) if q.search(ch.label)]

            if callable(q):
                return [i for i, ch in enumerate(self._channel_metadata) if bool(q(ch))]

            if isinstance(q, dict):
                # Validate dict keys: accept model fields or keys present in any
                # channel `extra` dict. If a key is not recognized at all,
                # raise KeyError to surface likely user mistakes.
                try:
                    model_keys = set(getattr(ChannelMetadata, "model_fields").keys())
                except Exception:
                    # Fallback for pydantic v1
                    model_keys = set(getattr(ChannelMetadata, "__fields__").keys())

                extra_keys: set[str] = set()
                for ch in self._channel_metadata:
                    if isinstance(ch.extra, dict):
                        extra_keys.update(ch.extra.keys())

                allowed_keys = model_keys | extra_keys
                unknown_keys = [k for k in q.keys() if k not in allowed_keys]
                if unknown_keys:
                    if validate_query_keys:
                        names_str = ", ".join(map(str, unknown_keys))
                        raise KeyError("Unknown channel metadata key(s): " + names_str)
                    # If validation is disabled, skip raising and let matching
                    # treat unknown keys as non-matching attributes.

                matches: list[int] = []
                for i, ch in enumerate(self._channel_metadata):
                    ok = True
                    for key, val in q.items():
                        # Prefer attribute access, fall back to dict-like access
                        attr = getattr(ch, key, None)
                        if attr is None:
                            # ChannelMetadata may support mapping-style access
                            try:
                                attr = ch[key]
                            except Exception:
                                ok = False
                                break

                        # If expected value is a regex pattern
                        if hasattr(val, "search") and callable(val.search):
                            if not (isinstance(attr, str) and val.search(attr)):
                                ok = False
                                break
                        else:
                            if attr != val:
                                ok = False
                                break
                    if ok:
                        matches.append(i)
                return matches

            raise TypeError(f"Unsupported query type: {type(q).__name__}")

        if query is not None:
            indices = _indices_from_query(query)
            if not indices:
                raise KeyError(f"No channels match query: {query!r}")
            channel_idx_list = indices
        else:
            if channel_idx is None:
                raise TypeError("Either 'channel_idx' or 'query' must be provided.")

            if isinstance(channel_idx, int):
                # Convert single channel to a list.
                channel_idx_list = [channel_idx]
            else:
                channel_idx_list = list(channel_idx)

        new_data = self._data[channel_idx_list]
        new_channel_metadata = [self._channel_metadata[i] for i in channel_idx_list]

        # Preserve operation_history (copy for immutability) but do not
        # append a selection operation so higher-level semantic operations
        # (e.g., 'fade') remain the last recorded operation.
        new_history = self.operation_history.copy() if self.operation_history else []

        return self._create_new_instance(
            data=new_data,
            operation_history=new_history,
            channel_metadata=new_channel_metadata,
        )

    def __len__(self) -> int:
        """
        Returns the number of channels.
        """
        return len(self._channel_metadata)

    def __iter__(self: S) -> Iterator[S]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(
        self: S,
        key: int
        | str
        | slice
        | list[int]
        | list[str]
        | tuple[
            int | str | slice | list[int] | list[str] | npt.NDArray[np.int_] | npt.NDArray[np.bool_],
            ...,
        ]
        | npt.NDArray[np.int_]
        | npt.NDArray[np.bool_],
    ) -> S:
        """
        Get channel(s) by index, label, or advanced indexing.

        This method supports multiple indexing patterns similar to NumPy and pandas:

        - Single channel by index: `frame[0]`
        - Single channel by label: `frame["ch0"]`
        - Slice of channels: `frame[0:3]`
        - Multiple channels by indices: `frame[[0, 2, 5]]`
        - Multiple channels by labels: `frame[["ch0", "ch2"]]`
        - NumPy integer array: `frame[np.array([0, 2])]`
        - Boolean mask: `frame[mask]` where mask is a boolean array
        - Multidimensional indexing: `frame[0, 100:200]` (channel + time)

        Parameters
        ----------
        key : int, str, slice, list, tuple, or ndarray
            - int: Single channel index (supports negative indexing)
            - str: Single channel label
            - slice: Range of channels
            - list[int]: Multiple channel indices
            - list[str]: Multiple channel labels
            - tuple: Multidimensional indexing (channel_key, time_key, ...)
            - ndarray[int]: NumPy array of channel indices
            - ndarray[bool]: Boolean mask for channel selection

        Returns
        -------
        S
            New instance containing the selected channel(s).

        Raises
        ------
        ValueError
            If the key length is invalid for the shape or if boolean mask
            length doesn't match number of channels.
        IndexError
            If the channel index is out of range.
        TypeError
            If the key type is invalid or list contains mixed types.
        KeyError
            If a channel label is not found.

        Examples
        --------
        >>> # Single channel selection
        >>> frame[0]  # First channel
        >>> frame["acc_x"]  # By label
        >>> frame[-1]  # Last channel
        >>>
        >>> # Multiple channel selection
        >>> frame[[0, 2, 5]]  # Multiple indices
        >>> frame[["acc_x", "acc_z"]]  # Multiple labels
        >>> frame[0:3]  # Slice
        >>>
        >>> # NumPy array indexing
        >>> frame[np.array([0, 2, 4])]  # Integer array
        >>> mask = np.array([True, False, True])
        >>> frame[mask]  # Boolean mask
        >>>
        >>> # Time slicing (multidimensional)
        >>> frame[0, 100:200]  # Channel 0, samples 100-200
        >>> frame[[0, 1], ::2]  # Channels 0-1, every 2nd sample
        """

        # Single index (int)
        if isinstance(key, numbers.Integral):
            # Ensure we pass a plain Python int to satisfy the type checker
            return self.get_channel(int(key))

        # Single label (str)
        if isinstance(key, str):
            index = self.label2index(key)
            return self.get_channel(index)

        # Phase 2: NumPy array support (bool mask and int array)
        if isinstance(key, np.ndarray):
            if key.dtype == bool or key.dtype == np.bool_:
                # Boolean mask
                if len(key) != self.n_channels:
                    raise ValueError(
                        f"Boolean mask length {len(key)} does not match number of channels {self.n_channels}"
                    )
                indices = np.where(key)[0]
                return self.get_channel(indices)
            elif np.issubdtype(key.dtype, np.integer):
                # Integer array
                return self.get_channel(key)
            else:
                raise TypeError(f"NumPy array must be of integer or boolean type, got {key.dtype}")

        # Phase 1: List support (int or str)
        if isinstance(key, list):
            if len(key) == 0:
                raise ValueError("Cannot index with an empty list")

            # Check if all elements are strings
            if all(isinstance(k, str) for k in key):
                # Multiple labels - type narrowing for mypy
                str_list = cast(list[str], key)
                indices_from_labels = [self.label2index(label) for label in str_list]
                return self.get_channel(indices_from_labels)

            # Check if all elements are integers
            elif all(isinstance(k, int | np.integer) for k in key):
                # Multiple indices - convert to list[int] for type safety
                int_list = [int(k) for k in key]
                return self.get_channel(int_list)

            else:
                raise TypeError(
                    f"List must contain all str or all int, got mixed types: {[type(k).__name__ for k in key]}"
                )

        # Tuple: multidimensional indexing
        if isinstance(key, tuple):
            return self._handle_multidim_indexing(key)

        # Slice
        if isinstance(key, slice):
            new_data = self._data[key]
            new_channel_metadata = self._channel_metadata[key]
            if isinstance(new_channel_metadata, ChannelMetadata):
                new_channel_metadata = [new_channel_metadata]
            return self._create_new_instance(
                data=new_data,
                operation_history=self.operation_history,
                channel_metadata=new_channel_metadata,
            )

        raise TypeError(f"Invalid key type: {type(key).__name__}. Expected int, str, slice, list, tuple, or ndarray.")

    def _handle_multidim_indexing(
        self: S,
        key: tuple[
            int | str | slice | list[int] | list[str] | npt.NDArray[np.int_] | npt.NDArray[np.bool_],
            ...,
        ],
    ) -> S:
        """
        Handle multidimensional indexing (channel + time axis).

        Parameters
        ----------
        key : tuple
            Tuple of indices where the first element selects channels
            and subsequent elements select along other dimensions (e.g., time).

        Returns
        -------
        S
            New instance with selected channels and time range.

        Raises
        ------
        ValueError
            If the key length exceeds the data dimensions.
        """
        if len(key) > self._data.ndim:
            raise ValueError(f"Invalid key length: {len(key)} for shape {self.shape}")

        # First element: channel selection
        channel_key = key[0]
        time_keys = key[1:] if len(key) > 1 else ()

        # Select channels first (recursively call __getitem__)
        if isinstance(channel_key, list | np.ndarray):
            selected = self[channel_key]
        elif isinstance(channel_key, int | str | slice):
            selected = self[channel_key]
        else:
            raise TypeError(f"Invalid channel key type in tuple: {type(channel_key).__name__}")

        # Apply time indexing if present
        if time_keys:
            new_data = selected._data[(slice(None),) + time_keys]
            return selected._create_new_instance(
                data=new_data,
                operation_history=selected.operation_history,
                channel_metadata=selected._channel_metadata,
            )

        return selected

    def label2index(self, label: str) -> int:
        """
        Get the index from a channel label.

        Parameters
        ----------
        label : str
            Channel label.

        Returns
        -------
        int
            Corresponding index.

        Raises
        ------
        KeyError
            If the channel label is not found.
        """
        for idx, ch in enumerate(self._channel_metadata):
            if ch.label == label:
                return idx
        raise KeyError(f"Channel label '{label}' not found.")

    @property
    def shape(self) -> tuple[int, ...]:
        _shape: tuple[int, ...] = self._data.shape
        if _shape[0] == 1:
            return _shape[1:]
        return _shape

    @property
    def data(self) -> T:
        """
        Returns the computed data.
        Calculation is executed the first time this is accessed.
        """
        data = self.compute()
        if self.n_channels == 1:
            return data.squeeze(axis=0)
        return data

    @property
    def labels(self) -> list[str]:
        """Get a list of all channel labels."""
        return [ch.label for ch in self._channel_metadata]

    def compute(self) -> T:
        """
        Compute and return the data.
        This method materializes lazily computed data into a concrete NumPy array.

        Returns
        -------
        NDArrayReal
            The computed data.

        Raises
        ------
        ValueError
            If the computed result is not a NumPy array.
        """
        logger.debug("COMPUTING DASK ARRAY - This will trigger file reading and all processing")
        result = self._data.compute()

        if not isinstance(result, np.ndarray):
            raise ValueError(f"Computed result is not a np.ndarray: {type(result)}")

        logger.debug(f"Computation complete, result shape: {result.shape}")
        return cast(T, result)

    @abstractmethod
    def plot(self, plot_type: str = "default", ax: Axes | None = None, **kwargs: Any) -> Axes | Iterator[Axes]:
        """Plot the data"""
        pass

    def persist(self: S) -> S:
        """Persist the data in memory"""
        persisted_data = self._data.persist()
        return self._create_new_instance(data=persisted_data)

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Return additional keyword arguments for ``_create_new_instance``.

        Subclasses that require extra constructor parameters (e.g. ``n_fft``,
        ``hop_length``) should override this method.  The default returns an
        empty dict, which is correct for frames with no extra init args
        (e.g. ``ChannelFrame``).
        """
        return {}

    def _create_new_instance(self: S, data: DaArray, **kwargs: Any) -> S:
        """
        Create a new channel instance based on an existing channel.
        Keyword arguments can override or extend the original attributes.
        """

        sampling_rate = kwargs.pop("sampling_rate", self.sampling_rate)
        # if not isinstance(sampling_rate, int):
        #     raise TypeError("Sampling rate must be an integer")

        label = kwargs.pop("label", self.label)
        if not isinstance(label, str):
            raise TypeError("Label must be a string")

        metadata = kwargs.pop("metadata", copy.deepcopy(self.metadata))
        if not isinstance(metadata, (dict, FrameMetadata)):
            raise TypeError("Metadata must be a dictionary or FrameMetadata")

        channel_metadata = kwargs.pop("channel_metadata", copy.deepcopy(self._channel_metadata))
        if not isinstance(channel_metadata, list):
            raise TypeError("Channel metadata must be a list")

        # Get additional initialization arguments from derived classes
        additional_kwargs = self._get_additional_init_kwargs()
        kwargs.update(additional_kwargs)

        return type(self)(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            channel_metadata=channel_metadata,
            previous=self,
            **kwargs,
        )

    def __array__(self, dtype: npt.DTypeLike = None) -> NDArrayReal:
        """Implicit conversion to NumPy array"""
        result = self.compute()
        if dtype is not None:
            return result.astype(dtype)
        return result

    def visualize_graph(self, filename: str | None = None) -> VisualizeReturnType:
        """
        Visualize the computation graph and save it to a file.

        This method creates a visual representation of the Dask computation graph.
        In Jupyter notebooks, it returns an IPython.display.Image object that
        will be displayed inline. In other environments, it saves the graph to
        a file and returns None.

        Parameters
        ----------
        filename : str, optional
            Output filename for the graph image. If None, a unique filename
            is generated using UUID. The file is saved in the current working
            directory.

        Returns
        -------
        IPython.display.Image or None
            In Jupyter environments: Returns an IPython.display.Image object
            that can be displayed inline.
            In other environments: Returns None after saving the graph to file.

        Notes
        -----
        This method requires graphviz to be installed on your system:
        - Ubuntu/Debian: `sudo apt-get install graphviz`
        - macOS: `brew install graphviz`
        - Windows: Download from https://graphviz.org/download/

        The graph displays operation names (e.g., 'normalize', 'lowpass_filter')
        making it easier to understand the processing pipeline.

        Examples
        --------
        >>> import wandas as wd
        >>> signal = wd.read_wav("audio.wav")
        >>> processed = signal.normalize().low_pass_filter(cutoff=1000)
        >>> # In Jupyter: displays graph inline
        >>> processed.visualize_graph()
        >>> # Save to specific file
        >>> processed.visualize_graph("my_graph.png")

        See Also
        --------
        debug_info : Print detailed debug information about the frame
        """
        try:
            filename = filename or f"graph_{uuid.uuid4().hex[:8]}.png"
            return self._data.visualize(filename=filename)
        except Exception as e:
            logger.warning(f"Failed to visualize the graph: {e}")
            return None

    def _binary_op(
        self: S,
        other: S | int | float | complex | NDArrayReal | DaArray,
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> S:
        """Default implementation of binary operations using dask's lazy evaluation.

        Handles both frame-frame and frame-scalar/array operations with
        metadata propagation and history tracking.  Uses
        ``_create_new_instance`` so that subclass-specific constructor
        parameters are automatically forwarded.

        Subclasses may override this entirely (e.g. ``RoughnessFrame``).
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        metadata: FrameMetadata = self.metadata.copy() if self.metadata is not None else FrameMetadata()
        operation_history: list[dict[str, Any]] = self.operation_history.copy() if self.operation_history else []

        # Same-type frame operation
        if isinstance(other, type(self)):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    f"Sampling rate mismatch\n"
                    f"  Left operand: {self.sampling_rate} Hz\n"
                    f"  Right operand: {other.sampling_rate} Hz\n"
                    f"Resample one frame to match the other before performing "
                    f"{symbol} operation."
                )
            if self.n_channels != other.n_channels:
                raise ValueError(
                    f"Channel count mismatch\n"
                    f"  Left operand: {self.n_channels} channels\n"
                    f"  Right operand: {other.n_channels} channels\n"
                    f"Binary frame operations require matching channel counts to keep "
                    f"channel metadata aligned.\n"
                    f"Select, duplicate, or remove channels so both operands match "
                    f"before performing {symbol} operation."
                )
            if self.shape != other.shape:
                raise ValueError(
                    f"Frame shape mismatch\n"
                    f"  Left operand: {self.shape}\n"
                    f"  Right operand: {other.shape}\n"
                    f"Binary frame operations require identical shapes to avoid "
                    f"unintended broadcasting.\n"
                    f"Align the frame shapes before performing {symbol} operation."
                )

            result_data = op(self._data, other._data)

            merged_channel_metadata: list[ChannelMetadata] = []
            for self_ch, other_ch in zip(self._channel_metadata, other._channel_metadata):
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch['label']} {symbol} {other_ch['label']})"
                merged_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other.label})

            return self._create_new_instance(
                data=result_data,
                label=f"({self.label} {symbol} {other.label})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=merged_channel_metadata,
            )

        # Scalar or array operand
        result_data = op(self._data, other)
        other_str = self._format_operand_str(other)

        updated_channel_metadata: list[ChannelMetadata] = []
        for self_ch in self._channel_metadata:
            ch = self_ch.model_copy(deep=True)
            ch["label"] = f"({self_ch.label} {symbol} {other_str})"
            updated_channel_metadata.append(ch)

        operation_history.append({"operation": symbol, "with": other_str})

        return self._create_new_instance(
            data=result_data,
            label=f"({self.label} {symbol} {other_str})",
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=updated_channel_metadata,
        )

    @staticmethod
    def _format_operand_str(other: object) -> str:
        """Return a short display string for a binary operand."""
        if isinstance(other, int | float):
            return str(other)
        if isinstance(other, complex):
            return f"complex({other.real}, {other.imag})"
        if isinstance(other, np.ndarray):
            return f"ndarray{other.shape}"
        if hasattr(other, "shape"):
            return f"dask.array{other.shape}"
        return str(type(other).__name__)

    def __add__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Addition operator"""
        return self._binary_op(other, lambda x, y: x + y, "+")

    def __sub__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Subtraction operator"""
        return self._binary_op(other, lambda x, y: x - y, "-")

    def __mul__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Multiplication operator"""
        return self._binary_op(other, lambda x, y: x * y, "*")

    def __truediv__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Division operator"""
        return self._binary_op(other, lambda x, y: x / y, "/")

    def __pow__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Power operator"""
        return self._binary_op(other, lambda x, y: x**y, "**")

    def apply_operation(self: S, operation_name: str, **params: Any) -> S:
        """
        Apply a named operation.

        Parameters
        ----------
        operation_name : str
            Name of the operation to apply.
        **params : Any
            Parameters to pass to the operation.

        Returns
        -------
        S
            A new instance with the operation applied.
        """
        # Apply the operation through abstract method
        return self._apply_operation_impl(operation_name, **params)

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """Default implementation of operation application.

        Creates the named operation, applies it to the data, and returns
        a new frame with updated metadata and operation history.
        Derived classes may override this to add extra behaviour
        (e.g. channel relabelling).
        """
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from wandas.processing import create_operation

        operation = create_operation(operation_name, self.sampling_rate, **params)
        processed_data = operation.process(self._data)

        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = self.metadata.copy()
        new_metadata[operation_name] = params

        return self._create_new_instance(
            data=processed_data,
            metadata=new_metadata,
            operation_history=new_history,
        )

    @overload
    def _apply_operation_instance(
        self: S,
        operation: Any,
        operation_name: str | None = None,
        output_frame_class: None = None,
        output_frame_kwargs: dict[str, Any] | None = None,
    ) -> S: ...

    @overload
    def _apply_operation_instance(
        self: S,
        operation: Any,
        operation_name: str | None = None,
        output_frame_class: type[S_Out] = ...,
        output_frame_kwargs: dict[str, Any] | None = None,
    ) -> S_Out: ...

    def _apply_operation_instance(
        self: S,
        operation: Any,
        operation_name: str | None = None,
        output_frame_class: type[S_Out] | None = None,
        output_frame_kwargs: dict[str, Any] | None = None,
    ) -> S | S_Out:
        """Apply an already-instantiated operation to the frame.

        This method processes data through the operation, updates metadata,
        operation history, and channel labels atomically.  It is the
        entry-point used by ``ChannelProcessingMixin`` and by
        ``ChannelFrame._apply_operation_impl``.

        Parameters
        ----------
        operation : AudioOperation
            Instantiated operation to apply.
        operation_name : str, optional
            Override for the operation name in history.
        output_frame_class : type, optional
            If provided, the result is wrapped in this frame class instead
            of the same type as ``self``.  Enables domain transitions
            (e.g. ChannelFrame -> SpectralFrame) from ``apply()``.
        output_frame_kwargs : dict, optional
            Extra constructor keyword arguments required by *output_frame_class*
            (e.g. ``{"n_fft": 1024, "window": "hann"}``).
        """
        processed_data = operation.process(self._data)

        if operation_name is None:
            operation_name = getattr(operation, "name", "unknown_operation")
        params = getattr(operation, "params", {})

        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = self.metadata.copy()
        new_metadata[operation_name] = params

        metadata_updates = operation.get_metadata_updates()

        display_name = operation.get_display_name()
        new_channel_metadata = self._relabel_channels(operation_name, display_name)

        if output_frame_class is not None:
            if not isinstance(output_frame_class, type) or not issubclass(output_frame_class, BaseFrame):
                raise TypeError(
                    "Invalid output_frame_class\n"
                    f"  Got: {output_frame_class!r}\n"
                    f"  Expected: a BaseFrame subclass\n"
                    f"Pass a compatible Wandas frame class such as "
                    f"SpectralFrame or SpectrogramFrame."
                )

            # Domain transition: build a different frame type
            kw: dict[str, Any] = {
                "data": processed_data,
                "sampling_rate": metadata_updates.pop("sampling_rate", self.sampling_rate),
                "label": self.label,
                "metadata": new_metadata,
                "operation_history": new_history,
                "channel_metadata": new_channel_metadata,
                "previous": self,
            }
            kw.update(metadata_updates)
            if output_frame_kwargs:
                kw.update(output_frame_kwargs)
            try:
                return output_frame_class(**kw)
            except TypeError as exc:
                provided_kwargs = ", ".join(sorted(kw)) or "none"
                raise TypeError(
                    "Invalid output_frame_class constructor\n"
                    f"  Frame class: {output_frame_class.__name__}\n"
                    f"  Provided keyword arguments: {provided_kwargs}\n"
                    f"Ensure output_frame_class accepts these parameters and "
                    f"use output_frame_kwargs to supply any required "
                    f"domain-specific constructor arguments."
                ) from exc

        creation_params: dict[str, Any] = {
            "data": processed_data,
            "metadata": new_metadata,
            "operation_history": new_history,
            "channel_metadata": new_channel_metadata,
        }
        creation_params.update(metadata_updates)

        return self._create_new_instance(**creation_params)

    def _relabel_channels(
        self,
        operation_name: str,
        display_name: str | None = None,
    ) -> list[ChannelMetadata]:
        """
        Update channel labels to reflect applied operation.

        This method creates new channel metadata with labels that include
        the operation name, making it easier to track processing history
        and distinguish frames in plots.

        Parameters
        ----------
        operation_name : str
            Name of the operation (e.g., "normalize", "lowpass_filter")
        display_name : str, optional
            Display name for the operation. If None, uses operation_name.
            This allows operations to provide custom, more readable labels.

        Returns
        -------
        list[ChannelMetadata]
            New channel metadata with updated labels.
            Original metadata is deep-copied and only labels are modified.

        Examples
        --------
        >>> # Original label: "ch0"
        >>> # After normalize: "normalize(ch0)"
        >>> # After chained ops: "lowpass_filter(normalize(ch0))"

        Notes
        -----
        Labels are nested for chained operations, allowing full
        traceability of the processing pipeline.
        """
        display = display_name or operation_name
        new_metadata = []
        for ch in self._channel_metadata:
            # All channel metadata are ChannelMetadata objects at this point
            new_ch = ch.model_copy(deep=True)
            new_ch.label = f"{display}({ch.label})"
            new_metadata.append(new_ch)
        return new_metadata

    def debug_info(self) -> None:
        """Output detailed debug information"""
        logger.debug(f"=== {self.__class__.__name__} Debug Info ===")
        logger.debug(f"Label: {self.label}")
        logger.debug(f"Shape: {self.shape}")
        logger.debug(f"Sampling rate: {self.sampling_rate} Hz")
        logger.debug(f"Operation history: {len(self.operation_history)} operations")
        self._debug_info_impl()
        logger.debug("=== End Debug Info ===")

    def print_operation_history(self) -> None:
        """
        Print the operation history to standard output in a readable format.

        This method writes a human-friendly representation of the
        `operation_history` list to stdout. Each operation is printed on its
        own line with an index, the operation name (if available), and the
        parameters used.

        Examples
        --------
        >>> cf.print_operation_history()
        1: normalize {}
        2: low_pass_filter {'cutoff': 1000}
        """
        if not self.operation_history:
            print("Operation history: <empty>")
            return

        print(f"Operation history ({len(self.operation_history)}):")
        for i, record in enumerate(self.operation_history, start=1):
            # record is expected to be a dict with at least a 'operation' key
            op_name = record.get("operation") or record.get("name") or "<unknown>"
            # Copy params for display - exclude the 'operation'/'name' keys
            params = {k: v for k, v in record.items() if k not in ("operation", "name")}
            print(f"{i}: {op_name} {params}")

    def to_numpy(self) -> T:
        """Convert the frame data to a NumPy array.

        This method computes the Dask array and returns it as a concrete NumPy array.
        The returned array has the same shape as the frame's data.

        Returns
        -------
        T
            NumPy array containing the frame data.

        Examples
        --------
        >>> cf = ChannelFrame.read_wav("audio.wav")
        >>> data = cf.to_numpy()
        >>> print(f"Shape: {data.shape}")  # (n_channels, n_samples)
        """
        return self.data

    def to_tensor(self, framework: str = "torch", device: str | None = None) -> Any:
        """
        Convert the Dask array to a tensor in the specified framework.

        Parameters
        ----------
        framework : str, default="torch"
            The ML framework to use ("torch" or "tensorflow").
        device : str or None, optional
            Device to place the tensor on. For PyTorch, use "cpu", "cuda", "cuda:0",
            etc. For TensorFlow, use "/CPU:0", "/GPU:0", etc. If None, uses the default
            device.

        Returns
        -------
        torch.Tensor or tf.Tensor
            A tensor in the specified framework.

        Raises
        ------
        ImportError
            If the specified framework is not installed.
        ValueError
            If the framework is not supported.
        TypeError
            If self.data is not a Dask array.

        Examples
        --------
        >>> # PyTorch tensor on CPU
        >>> tensor = frame.to_tensor(framework="torch", device="cpu")
        >>> # PyTorch tensor on GPU
        >>> tensor = frame.to_tensor(framework="torch", device="cuda:0")
        >>> # TensorFlow tensor on GPU
        >>> tensor = frame.to_tensor(framework="tensorflow", device="/GPU:0")
        """

        # Compute the Dask array to NumPy array
        numpy_data = self.to_numpy()

        if framework == "torch":
            try:
                import importlib.util

                if importlib.util.find_spec("torch") is None:
                    raise ImportError(
                        "PyTorch is not installed\n"
                        "  Required for: tensor conversion with framework='torch'\n"
                        "  Install with: pip install torch"
                    )
                import torch

                # Convert NumPy array to PyTorch tensor
                tensor = torch.from_numpy(numpy_data)

                # Move to specified device if provided
                if device is not None:
                    tensor = tensor.to(device)

                return tensor

            except ImportError as e:
                raise ImportError(
                    "PyTorch is not installed\n"
                    "  Required for: tensor conversion with framework='torch'\n"
                    "  Install with: pip install torch"
                ) from e

        elif framework == "tensorflow":
            try:
                import importlib.util

                if importlib.util.find_spec("tensorflow") is None:
                    raise ImportError(
                        "TensorFlow is not installed\n"
                        "  Required for: tensor conversion with\n"
                        "  framework='tensorflow'\n"
                        "  Install with: pip install tensorflow"
                    )
                import tensorflow as tf

                # Convert NumPy array to TensorFlow tensor
                if device is not None:
                    with tf.device(device):
                        tensor = tf.convert_to_tensor(numpy_data)
                else:
                    tensor = tf.convert_to_tensor(numpy_data)

                return tensor

            except ImportError as e:
                raise ImportError(
                    "TensorFlow is not installed\n"
                    "  Required for: tensor conversion with framework='tensorflow'\n"
                    "  Install with: pip install tensorflow"
                ) from e

        else:
            raise ValueError(
                f"Unsupported framework\n"
                f"  Got: '{framework}'\n"
                f"  Expected: 'torch' or 'tensorflow'\n"
                f"Use a supported framework for tensor conversion"
            )

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert the frame data to a pandas DataFrame.

        This method provides a common implementation for converting frame data
        to pandas DataFrame. Subclasses can override this method for custom behavior.

        Returns
        -------
        pd.DataFrame
            DataFrame with appropriate index and columns.

        Examples
        --------
        >>> cf = ChannelFrame.read_wav("audio.wav")
        >>> df = cf.to_dataframe()
        >>> print(df.head())
        """
        # Get data as numpy array
        data = self.to_numpy()

        # Get column names from subclass
        columns = self._get_dataframe_columns()

        # Get index from subclass
        index = self._get_dataframe_index()

        # Create DataFrame
        if data.ndim == 1:
            # Single channel case - reshape to 2D
            df = pd.DataFrame(data.reshape(-1, 1), columns=columns, index=index)
        else:
            # Multi-channel case - transpose to (n_samples, n_channels)
            df = pd.DataFrame(data.T, columns=columns, index=index)

        return df

    @abstractmethod
    def _get_dataframe_columns(self) -> list[str]:
        """Get column names for DataFrame.

        This method should be implemented by subclasses to provide
        appropriate column names for the DataFrame.

        Returns
        -------
        list[str]
            List of column names.
        """
        pass

    @abstractmethod
    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """Get index for DataFrame.

        This method should be implemented by subclasses to provide
        appropriate index for the DataFrame based on the frame type.

        Returns
        -------
        pd.Index
            Index for the DataFrame.
        """
        pass

    def _debug_info_impl(self) -> None:
        """Implement derived class-specific debug information"""
        pass

    def _print_operation_history(self) -> None:
        """Print the operation history information.

        This is a helper method for info() implementations to display
        the number of operations applied to the frame in a consistent format.
        """
        if self.operation_history:
            print(f"  Operations Applied: {len(self.operation_history)}")
        else:
            print("  Operations Applied: None")

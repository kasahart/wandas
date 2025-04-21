# wandas/core/file_frame.py
import glob
import os
from collections.abc import Iterator
from typing import Optional, Union

import ipywidgets as widgets

from .channel_frame import ChannelFrame


class FileFrame:
    def __init__(
        self, channel_frames: list["ChannelFrame"], label: Optional[str] = None
    ):
        """
        Initialize a FileFrame object.

        Parameters
        ----------
        channel_frames : list of ChannelFrame
            List of ChannelFrame objects.
        label : str, optional
            Label for the file frame.
        """
        self.channel_frames = channel_frames
        self.label = label

        # Build a dictionary for accessing by file name
        # self.file_dict = {file: file for file in files}
        # if len(self.file_dict) != len(files):
        #     raise ValueError("File labels must be unique.")

    @classmethod
    def from_filelist(
        cls, files: list[str], label: Optional[str] = None
    ) -> "FileFrame":
        """
        Create a FileFrame object from a list of files.

        Parameters
        ----------
        files : list of str
            List of file paths.
        label : str, optional
            Label for the file frame.

        Returns
        -------
        FileFrame
            FileFrame object created from the file list.

        Raises
        ------
        ValueError
            If the file format is not supported.
        """

        # Build a dictionary for accessing by file name
        # self.file_dict = {file: file for file in files}
        # if len(self.file_dict) != len(files):
        #     raise ValueError("File labels must be unique.")

        channel_frames = []
        for file in files:
            # Switch loading function based on file extension
            if file.endswith(".wav"):
                # Load WAV file
                channel_frame = ChannelFrame.read_wav(file)
            else:
                raise ValueError(f"Unsupported file format: {file}")

            channel_frames.append(
                channel_frame,
            )

        return cls(channel_frames, label)

    @classmethod
    def from_dir(
        cls, dir_path: str, label: Optional[str] = None, suffix: Optional[str] = None
    ) -> "FileFrame":
        """
        Create a FileFrame object from a directory.

        Parameters
        ----------
        dir_path : str
            Path to the directory.
        label : str, optional
            Label for the file frame.
        suffix : str, optional
            File extension to load. If None, all files will be loaded.

        Returns
        -------
        FileFrame
            FileFrame object created from the directory.
        """
        pattern = os.path.join(dir_path, "**", ("*" + suffix) if suffix else "*")
        file_list = sorted(glob.glob(pattern, recursive=True))
        return cls.from_filelist(file_list, label)

    def describe(self) -> widgets.VBox:
        """
        Display information about the channels.

        Returns
        -------
        VBox
            Widget containing channel information.
        """
        content = []
        content += [frame.describe() for frame in self.channel_frames]
        # Set layout for center alignment
        layout = widgets.Layout(
            display="flex", justify_content="center", align_items="center"
        )
        return widgets.VBox(content, layout=layout)

    def __iter__(self) -> Iterator["ChannelFrame"]:
        """
        Iterate through the channel frames.

        Returns
        -------
        Iterator[ChannelFrame]
            Iterator of ChannelFrame objects.
        """
        return iter(self.channel_frames)

    def __getitem__(self, key: Union[str, int]) -> "ChannelFrame":
        """
        Get a channel frame by index.

        Parameters
        ----------
        key : int
            Index of the channel frame.

        Returns
        -------
        ChannelFrame
            Corresponding ChannelFrame object.

        Raises
        ------
        IndexError
            If the index is out of range.
        TypeError
            If the key is not an integer.
        """
        if isinstance(key, int):
            # Access by index
            if key < 0 or key >= len(self.channel_frames):
                raise IndexError(f"Channel index {key} out of range.")
            return self.channel_frames[key]
        else:
            raise TypeError(
                "Key must be either a string (channel name) or an integer "
                "(channel index)."
            )

    def __len__(self) -> int:
        """
        Return the number of files.

        Returns
        -------
        int
            Number of files.
        """
        return len(self.channel_frames)

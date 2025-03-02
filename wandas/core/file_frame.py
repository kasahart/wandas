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
        FileFrame オブジェクトを初期化します。

        Parameters:
            files (list of str): ファイルパスのリスト。
            label (str, optional): ファイルのラベル。
        """
        self.channel_frames = channel_frames
        self.label = label

        # ファイル名で辞書のようにアクセスできるようにするための辞書を構築
        # self.file_dict = {file: file for file in files}
        # if len(self.file_dict) != len(files):
        #     raise ValueError("File labels must be unique.")

    @classmethod
    def from_filelist(
        cls, files: list[str], label: Optional[str] = None
    ) -> "FileFrame":
        """
        指定されたファイルリストから FileFrame オブジェクトを作成します。

        Args:
            files (list[str]): ファイルパスのリスト。
            label (Optional[str], optional): ファイルのラベル。デフォルトは None。

        Raises:
            ValueError: サポートされていないファイル形式が含まれている場合。
        """

        # ファイル名で辞書のようにアクセスできるようにするための辞書を構築
        # self.file_dict = {file: file for file in files}
        # if len(self.file_dict) != len(files):
        #     raise ValueError("File labels must be unique.")

        channel_frames = []
        for file in files:
            # ファイルの拡張子に応じて読み込み関数を切り替え
            if file.endswith(".wav"):
                # wav ファイルの読み込み
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
        指定されたディレクトリから FileFrame オブジェクトを作成します。

        Args:
            dir_path (str): ディレクトリのパス。
            label (Optional[str], optional): ファイルのラベル。デフォルトは None。
            suffix (Optional[str], optional): 読み込むファイルの拡張子。
                デフォルトは None。

        Returns:
            FileFrame: 作成された FileFrame オブジェクト。
        """
        pattern = os.path.join(dir_path, "**", ("*" + suffix) if suffix else "*")
        file_list = sorted(glob.glob(pattern, recursive=True))
        return cls.from_filelist(file_list, label)

    def describe(self) -> widgets.VBox:
        """
        チャンネルの情報を表示します。
        """
        content = []
        content += [frame.describe() for frame in self.channel_frames]
        # 中央寄せのレイアウトを設定
        layout = widgets.Layout(
            display="flex", justify_content="center", align_items="center"
        )
        return widgets.VBox(content, layout=layout)

    # forでループを回すためのメソッド
    def __iter__(self) -> Iterator["ChannelFrame"]:
        return iter(self.channel_frames)

    def __getitem__(self, key: Union[str, int]) -> "ChannelFrame":
        """
        チャンネル名またはインデックスでチャンネルを取得するためのメソッド。

        Parameters:
            key (str or int): チャンネルの名前（label）またはインデックス番号。

        Returns:
            Channel: 対応するチャンネル。
        """
        if isinstance(key, int):
            # インデックス番号でアクセス
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
        ファイル数を返します。
        """
        return len(self.channel_frames)

# Unreleased changes

Changes after [Wandas 0.14.0](v0.14.0.md).

0.14.0以降の変更です。

## Changed / 変更

- `save()` to WDF accepts a subclass of a built-in Frame, such as
  `class ProjectFrame(ChannelFrame)`, and saves it as its built-in base class,
  so loading the file returns that built-in class (for example `ChannelFrame`).
  A subclass that defines its own `__init__` is still rejected, because WDF
  cannot store the constructor state it adds
  ([#710](https://github.com/kasahart/wandas/issues/710)).
  WDFへの`save()`は、`class ProjectFrame(ChannelFrame)`のような組み込みFrameのsubclassも受け付け、
  組み込みの親クラスとして保存します。読み込むとその組み込みクラス（例: `ChannelFrame`）が返ります。
  独自の`__init__`を定義したsubclassは、追加したconstructorの状態をWDFに保存できないため、
  今までどおり拒否します。

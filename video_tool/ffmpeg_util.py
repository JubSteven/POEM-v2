from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional, Tuple

import os
import numpy as np
import ffmpeg


def load_frame_buf(in_filename: str, pix_fmt: str, frame_id: str):
    out, _ = (
        ffmpeg.input(in_filename)
        .filter("select", "gte(n,{})".format(frame_id))
        .output("pipe:", vframes=1, format="rawvideo", pix_fmt=pix_fmt)
        .run(capture_stdout=True, quiet=True)
    )
    return out


def load_frame_buf_ranged(in_filename: str, pix_fmt: str, frame_id: str, frame_num: int):
    out, _ = (
        ffmpeg.input(in_filename)
        .filter("select", "gte(n,{})".format(frame_id))
        .output("pipe:", vframes=frame_num, format="rawvideo", pix_fmt=pix_fmt)
        .run(capture_stdout=True, quiet=True)
    )
    return out


class FFMPEGFrameLoader:
    pix_fmt_supported = ["rgb24", "rgba", "gray16le"]
    dtype_mappping = {
        "rgb24": np.uint8,
        "rgba": np.uint8,
        "gray16le": np.uint16,
    }
    shape_mapping = {
        "rgb24": (3,),
        "rgba": (4,),
        "gray16le": (),
    }

    def __init__(
        self,
        in_filename: str,
        pix_fmt: str,
        cache_size: int = 32,
        num_frame: Optional[int] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ):
        if num_frame is None or frame_size is None:
            # probe the infile to get the number of frame
            probe = ffmpeg.probe(in_filename)

        self.in_filename = in_filename
        if pix_fmt not in self.pix_fmt_supported:
            raise ValueError(f"Unsupported pixel format: {pix_fmt}")
        self.pix_fmt = pix_fmt

        self.cache_size = cache_size
        
        if num_frame is None:
            try:
                self.num_frame = int(probe["streams"][0]["nb_frames"])
            except KeyError:
                duration = float(probe["format"]["duration"])
                r_frame_rate = probe["streams"][0]["r_frame_rate"]  # e.g., "30/1"
                numerator, denominator = map(int, r_frame_rate.split('/'))
                fps = numerator / denominator
                self.num_frame = int(duration * fps + 0.5)
        else:
            self.num_frame = num_frame
        if frame_size is None:
            self.frame_size = (int(probe["streams"][0]["width"]), int(probe["streams"][0]["height"]))
        else:
            self.frame_size = frame_size

        self.reset()

    def reset(self):
        self.cache = None
        self.cache_itvl = None
        self.cache_itvl_size = None

    @staticmethod
    def in_interval(fid, itvl):
        return fid >= itvl[0] and fid < itvl[1]

    @staticmethod
    def compute_interval(fid, size, nframe):
        start = max(fid - size // 2, 0)
        end = min(start + size, nframe)
        return (start, end)

    def __getitem__(self, frame_id: int):
        if frame_id < 0 or frame_id >= self.num_frame:
            raise IndexError(f"frame_id out of range: {frame_id} of [0, {self.num_frame})")

        if self.cache_itvl is None or not self.in_interval(frame_id, self.cache_itvl):
            # load cache
            ## compute cache itvl, it should be centered at frame_id and constrained by 0 and self.num_frame
            self.cache_itvl = self.compute_interval(frame_id, self.cache_size, self.num_frame)
            self.cache_itvl_size =  self.cache_itvl[1] - self.cache_itvl[0]
            cache_buf = load_frame_buf_ranged(
                self.in_filename, self.pix_fmt, self.cache_itvl[0], self.cache_itvl_size
            )
            array_shape = (self.cache_itvl_size, self.frame_size[1], self.frame_size[0], *self.shape_mapping[self.pix_fmt])
            self.cache = np.frombuffer(cache_buf, dtype=self.dtype_mappping[self.pix_fmt]).reshape(array_shape)        # TODO
        
        local_offset = frame_id - self.cache_itvl[0]
        return self.cache[local_offset]
    
    def __len__(self):
        return self.num_frame

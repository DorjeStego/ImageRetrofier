import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import io
from PIL import Image, ImageFilter
from decoder_exceptions import InvalidFiletypeError

class Decoder:
    _image_types = frozenset(["JPG", "PNG", "BMP"])

    def __init__(self, path_str:str, init_state:Dict[str, str|int|bool|Any]):
        self.path = path_str
        self.init_state = init_state
        self.image_type = None
        self._image_bytes = Path(path_str).read_bytes()
        self._type_image(self._image_bytes)
        if self.image_type not in Decoder._image_types:
            raise InvalidFiletypeError
        return

    def _type_image(self, image_bytes:bytes):
        if image_bytes.startswith(b"\xFF\xD8\xFF"): # JPEG
            self.image_type = "JPG"
        elif image_bytes.startswith(b"b\x89PNG\r\n\x1a\n"): # PNG
            self.image_type = "PNG"
        elif image_bytes.startswith(b"BM"): # Bitmap
            self.image_type = "BMP"
        return

    def decode_image_to_rows(self, dtype=np.uint8):
        with Image.open(io.BytesIO(self._image_bytes)) as img:
            rgb = img.convert("RGB") # 3 channels
            arr = np.asarray(rgb, dtype=dtype)
        return arr[..., np.newaxis]

    def tile_image_rgb(self, arr:np.ndarray, n:int, mode:str="strict"):
        """
        Convert an image array into NxN tiles that perfectly tile the image.
        :param arr: (H,W,3 or (H,W,3,1)
        :param n: tile size (square, NxN)
        :param mode: strict|crop|pad_edge
        :return: tiles: (tiles_y, tiles_x, N, N, 3)
        """
        if arr.ndim not in (3, 4):
            raise ValueError(f"Expected (H,W,3 or (H,W,3,1), got shape {arr.shape}")
        if arr.shape[2] != 3:
            raise ValueError(f"Expected RGB in axis 2, got shape {arr.shape}")
        if n <= 1:
            raise ValueError("N must be > 1")

        h, w = arr.shape[0], arr.shape[1]
        has_batch = (arr.ndim == 4)
        batch_dim = arr.shape[3] if has_batch else None
        if has_batch and batch_dim != 1:
            raise ValueError("If 4D, expected shape (H,W,3,1)")

        rem_h, rem_w = h % n, w % n

        if rem_h or rem_w:
            if mode == "strict":
                raise ValueError(f"Image size {(h, w)} not divisible by N={n}")
            elif mode == "crop":
                h2 = h - rem_h
                w2 = w - rem_w
                arr = arr[:h2,:w2,...]
                h, w = h2, w2
            elif mode == "pad_edge":
                pad_h = (n - rem_h) % n
                pad_w = (n - rem_w) % n
                pad_spec = ((0, pad_h), (0, pad_w), (0, 0)) + (((0, 0),) if has_batch else ())
                arr = np.pad(arr, pad_spec, mode="edge")
                h, w = arr.shape[0], arr.shape[1]
            else:
                raise ValueError("Mode must be 'strict', 'crop', or 'pad_edge'")

        tiles_y = h // n
        tiles_x = w // n

        if not has_batch:
            tiles = arr.reshape(tiles_y, n, tiles_x, n, 3).transpose(0,2,1,3,4)
        else:
            tiles = arr.reshape(tiles_y, n, tiles_x, n, 3, 1).transpose(0,2,1,3,4,5)
        return tiles

    def untile_image_rgb(self, tiles:np.ndarray) -> np.ndarray:
        if tiles.ndim ==5:
            ty, tx, n1, n2, c = tiles.shape
            if n1 != n2 or c != 3:
                raise ValueError(f"Bad tile shape {tiles.shape}")
            return tiles.transpose(0, 2, 1, 3, 4).reshape(ty * n1, tx * n1, c)
        if tiles.ndim == 6:
            ty, tx, n1, n2, c, b = tiles.shape
            if n1 != n2 or c != 3 or b != 1:
                raise ValueError(f"Bad tile shape {tiles.shape}")
            return tiles.transpose(0, 2, 1, 3, 4, 5).reshape(ty * n1, tx * n1, c, b)
        raise ValueError(f"Expected 5D or 6D tiles, got {tiles.shape}")

    def tile_dot_fill(self, tiles:np.ndarray, out_dtype=np.float32) -> np.ndarray:
        """

        :param tiles: ty, tx, n, n, 3 or ty, tx, n, n, 3, 1
        :param out_dtype: Usually float32 for reconstructing image.
        :return: tiles of the same sape, where each tile is filled with(v.v) for that tile.
        """
        if tiles.ndim not in (5, 6):
            raise ValueError(f"Expected 5D or 6D tiles, got {tiles.shape}")
        if tiles.shape[4] != 3:
            raise ValueError(f"Expected RGB at axis 4, got {tiles.shape}")

        has_batch = (tiles.ndim == 6)
        if has_batch and tiles.shape[5] != 1:
            raise ValueError(f"If 6D, expected trailing batch dim of 1")

        # Work in a safe dtype to avoid uint8 overflow
        t = tiles.astype(np.float64, copy=False)

        if not has_batch:
            flat = t.reshape(t.shape[0], t.shape[1], -1)
            dot = np.einsum("...k,...k->...", flat, flat)
            filled = np.broadcast_to(dot[..., None, None, None], tiles.shape)
            return filled.astype(out_dtype, copy=False)
        else:
            flat = t.reshape(t.shape[0], t.shape[1], -1)
            dot = np.einsum("...k,...k->...", flat, flat)
            filled = np.broadcast_to(dot[...,None, None, None, None], tiles.shape)
            return filled.astype(out_dtype, copy=False)

    def _reduce_rows_dotproduct_divide_conquer(self, tile: np.ndarray) -> np.ndarray:
        """
        :param tile (N, N, 3) float64
        :return: (3,) float64  -- final per-channel value after row reduction
        """
        n = tile.shape[0]
        rows = tile  # (R, N, 3) initially R=N

        first_pass = True
        while rows.shape[0] > 1:
            r = rows.shape[0]

            # Pair up neighbouring rows: (0,1), (2,3), ...
            a = rows[0:r - (r % 2):2]  # (P, N, 3)
            b = rows[1:r - (r % 2):2]  # (P, N, 3)

            # Per-pair, per-channel dot across columns -> (P, 3)
            dots = np.einsum("pnc,pnc->pc", a, b)

            # Turn each scalar dot (per channel) into a full row vector of length N
            # => (P, N, 3)
            new_rows = np.broadcast_to(dots[:, None, :], (dots.shape[0], n, 3)).copy()

            # Odd leftover handling: on the FIRST pass, combine leftover with its neighbour dot-row
            if (r % 2) == 1:
                leftover = rows[-1]  # (N, 3)

                if first_pass:
                    # "taken as the dot product against the neighbouring dot product row after the first pass"
                    # Neighbour is the last produced row (from pairing R-3 with R-2).
                    if new_rows.shape[0] == 0:
                        # Degenerate case: only 1 row existed (won't happen with while), but keep safe.
                        new_rows = leftover[None, :, :]
                    else:
                        neighbour = new_rows[-1]  # (N, 3), constant-per-channel row

                        # Dot neighbour with leftover -> (3,)
                        fix = np.einsum("nc,nc->c", neighbour, leftover)

                        # Replace last produced row with the corrected dot-row
                        new_rows[-1] = np.broadcast_to(fix[None, :], (n, 3))
                else:
                    # After first pass, simplest consistent behaviour is: fold leftover into last row again.
                    # (You can change this rule if you want different adjacency semantics.)
                    if new_rows.shape[0] == 0:
                        new_rows = leftover[None, :, :]
                    else:
                        neighbour = new_rows[-1]
                        fix = np.einsum("nc,nc->c", neighbour, leftover)
                        new_rows[-1] = np.broadcast_to(fix[None, :], (n, 3))

            rows = new_rows
            first_pass = False

        # rows is now (1, N, 3). Usually constant across columns; take mean to be safe.
        return rows[0].mean(axis=0)

    def tile_channel_energy_fill_divconq_rows(self, tiles: np.ndarray, out_dtype=np.float32) -> np.ndarray:
        """
        :param tiles: (ty,tx,N,N,3) or (ty,tx,N,N,3,1)
        :return: same shape, filled per tile with the final per-channel reduction value.
        """
        if tiles.ndim not in (5, 6):
            raise ValueError(f"Expected 5D or 6D tiles, got {tiles.shape}")
        if tiles.shape[4] != 3:
            raise ValueError(f"Expected RGB at axis 4, got {tiles.shape}")
        if tiles.ndim == 6 and tiles.shape[5] != 1:
            raise ValueError("If 6D, expected trailing batch dim of 1")

        has_batch = (tiles.ndim == 6)
        t = tiles.astype(np.float64, copy=False)

        if not has_batch:
            ty, tx, n, n2, c = t.shape
            if n != n2:
                raise ValueError("Tiles must be NxN")

            out = np.empty((ty, tx, n, n, 3), dtype=out_dtype)
            for y in range(ty):
                for x in range(tx):
                    v = self._reduce_rows_dotproduct_divide_conquer(t[y, x])  # (3,)
                    out[y, x] = np.broadcast_to(v[None, None, :], (n, n, 3))
            return out

        else:
            ty, tx, n, n2, c, b = t.shape
            if n != n2:
                raise ValueError("Tiles must be NxN")

            out = np.empty((ty, tx, n, n, 3, 1), dtype=out_dtype)
            for y in range(ty):
                for x in range(tx):
                    v = self._reduce_rows_dotproduct_divide_conquer(t[y, x, :, :, :, 0])  # (3,)
                    out[y, x, :, :, :, 0] = np.broadcast_to(v[None, None, :], (n, n, 3))
            return out

    def tile_channel_energy_fill(self, tiles:np.ndarray, out_dtype=np.float32) -> np.ndarray:
        """

        :param tiles: (ty,tx,n,n,3) or (ty,tx,n,n,3,1)
        :param out_dtype: default np.float32
        :return: same shape filled per tile with [R_energy, G_energy, B_energy]
        """
        has_batch = (tiles.ndim == 6)
        t = tiles.astype(np.float64, copy=False)

        # TODO: Add multiple mathematical pathways - use comments as starting point.
        if not has_batch:
            # area = tiles.shape[-4] * tiles.shape[-3]
            e = np.einsum("...ijc,...ijc->...c", t, t)
            # rms = np.sqrt(e / area)
            # mean = np.mean(t, axis=(-4, -3))
            filled = np.broadcast_to(e[..., None, None, :], tiles.shape)
            return filled.astype(out_dtype, copy=False)
        else:
            t0 = t[..., 0]
            # area = tiles.shape[-4] * tiles.shape[-3]
            e = np.einsum("...ijc,...ijc->...c", t0, t0)
            # rms = np.sqrt(e / area)
            # mean = np.mean(t, axis=(-4, -3))
            filled0 = np.broadcast_to(e[..., None, None, :], t0.shape)
            return filled0[..., None].astype(out_dtype, copy=False)

    def save_rgb_image(self, path:str, img: np.ndarray) -> None:
        """
        :param path: Output file path
        :param img: Output image as np.ndarray
        :return: None; saves to disk
        """
        if img.ndim == 4 and img.shape[3] == 1:
            img = img[..., 0]
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) or (H,W,3,1), got {img.shape}")
        if img.dtype == np.uint8:
            out = img
        else:
            x = img.astype(np.float64, copy=False)
            mn = np.min(x)
            mx = np.max(x)
            if mx == mn:
                out = ((x - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)
            else:
                out = ((x - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)
        Image.fromarray(out, mode="RGB").save(path)

    def save_rgb_image_per_channel(self, path:str|Path, img:np.ndarray):
        if self.init_state.get("verbose") == True:
            print(f"Saving output file to {path}...")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if img.ndim == 4 and img.shape[3] == 1:
            img = img[..., 0]  # (H,W,3)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) or (H,W,3,1), got {img.shape}")

        if img.dtype != np.uint8:
            x = img.astype(np.float64, copy=False)
            mn = x.reshape(-1, 3).min(axis=0)
            mx = x.reshape(-1, 3).max(axis=0)
            denom = np.where(mx == mn, 1.0, (mx - mn))
            out = ((x - mn) * (255.0 / denom)).clip(0, 255).astype(np.uint8)
        else:
            out = img

        Image.fromarray(out, mode="RGB").save(path)

    def _to_uint8_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Coerce (H,W,3) or (H,W,3,1) into (H,W,3) uint8 RGB."""
        if arr.ndim == 4 and arr.shape[3] == 1:
            arr = arr[..., 0]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) or (H,W,3,1), got {arr.shape}")

        if arr.dtype == np.uint8:
            return arr

        x = arr.astype(np.float64, copy=False)

        # If values look like 0..1 floats, scale up; otherwise clip to 0..255.
        mx = float(np.max(x)) if x.size else 0.0
        if mx <= 1.0:
            x = x * 255.0

        return np.clip(x, 0.0, 255.0).astype(np.uint8)

    def edge_preserving_flatten(
        self,
        arr: np.ndarray,
        *,
        median_size: int = 3,
        passes: int = 1,
    ) -> np.ndarray:
        """
        Edge-preserving-ish flattening using a median filter (fast, SBC-friendly).

        median_size: odd integer (3,5,7...). Larger => flatter, more edge chunking.
        passes: apply filter multiple times for stronger effect.
        """
        if median_size < 3 or median_size % 2 == 0:
            raise ValueError("median_size must be an odd integer >= 3")
        if passes < 1:
            raise ValueError("passes must be >= 1")

        rgb = self._to_uint8_rgb(arr)
        img = Image.fromarray(rgb, mode="RGB")

        filt = ImageFilter.MedianFilter(size=median_size)
        for _ in range(passes):
            img = img.filter(filt)

        return np.asarray(img, dtype=np.uint8)

    def pixel_art_dominant_tile_quantize(
        self,
        arr: np.ndarray,
        *,
        tile_size: int,
        n_colours: int,
        dither: bool,
        mode: str = "crop",  # "crop" | "pad_edge" | "strict"
    ) -> np.ndarray:
        """
        1) Replace each NxN tile with its DOMINANT RGB colour (most frequent pixel colour in that tile)
        2) Quantise to n_colours (runtime)
        3) Optional dithering (runtime)

        mode:
          - "strict": require H and W divisible by tile_size
          - "crop": drop right/bottom remainders
          - "pad_edge": pad by repeating edge pixels to reach a multiple of tile_size
        """
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if n_colours < 2 or n_colours > 256:
            raise ValueError("n_colours must be in [2, 256]")
        if mode not in {"strict", "crop", "pad_edge"}:
            raise ValueError("mode must be 'strict', 'crop', or 'pad_edge'")

        rgb = self._to_uint8_rgb(arr)
        h, w, _ = rgb.shape

        rem_h, rem_w = h % tile_size, w % tile_size
        if rem_h or rem_w:
            if mode == "strict":
                raise ValueError(f"Image size {(h, w)} not divisible by tile_size={tile_size}")
            elif mode == "crop":
                h2 = h - rem_h
                w2 = w - rem_w
                rgb_work = rgb[:h2, :w2]
            else:  # pad_edge
                pad_h = (tile_size - rem_h) % tile_size
                pad_w = (tile_size - rem_w) % tile_size
                rgb_work = np.pad(
                    rgb,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="edge",
                )
        else:
            rgb_work = rgb

        h2, w2, _ = rgb_work.shape
        ty, tx = h2 // tile_size, w2 // tile_size

        # Reshape into tiles: (ty, tile, tx, tile, 3) -> (ty, tx, tile, tile, 3)
        tiles = rgb_work.reshape(ty, tile_size, tx, tile_size, 3).transpose(0, 2, 1, 3, 4)

        # Compute dominant colour per tile (loop over tiles, but each tile is small).
        # Pack RGB to 24-bit integer for fast counting.
        small = np.empty((ty, tx, 3), dtype=np.uint8)

        for y in range(ty):
            for x in range(tx):
                t = tiles[y, x].reshape(-1, 3).astype(np.uint32, copy=False)
                packed = (t[:, 0] << 16) | (t[:, 1] << 8) | t[:, 2]
                vals, counts = np.unique(packed, return_counts=True)
                dom = int(vals[np.argmax(counts)])
                small[y, x, 0] = (dom >> 16) & 0xFF
                small[y, x, 1] = (dom >> 8) & 0xFF
                small[y, x, 2] = dom & 0xFF

        # Upscale back to original working size using nearest-neighbor expansion (repeat).
        pixelated = np.repeat(np.repeat(small, tile_size, axis=0), tile_size, axis=1)

        # Quantise + optional dithering
        img = Image.fromarray(pixelated, mode="RGB")
        dither_flag = Image.FLOYDSTEINBERG if dither else Image.NONE
        img_q = img.quantize(colors=n_colours, dither=dither_flag).convert("RGB")

        out = np.asarray(img_q, dtype=np.uint8)

        # If we padded, you may want to crop back to original H,W to “fit” the input frame.
        if mode == "pad_edge" and (out.shape[0] != h or out.shape[1] != w):
            out = out[:h, :w]

        # If we cropped, output is smaller; that’s intentional for a perfect tiling.
        return out

class JPEG:
    def __init__(self):
        return

class Bitmap:
    def __init__(self):
        return

class PNG:
    def __init__(self):
        return

if __name__ == "__main__":
    sys.exit()

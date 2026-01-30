import os
import argparse
from argparse import ArgumentParser
from typing import Dict, Any

from arg_exceptions import ArgError
from decoder_core import Decoder
from src.decoder_core import edge_preserving_flatten, pixel_art_dominant_tile_quantize


def parse_args(program_info:Dict[str,str]):
    parser = build_parser(program_info)
    args = parser.parse_args()
    print(args)
    return args, parser

def build_state(args, parser):
    if not args.input_filename:
        raise ArgError("No input file", parser)
    if not args.output_filename:
        raise ArgError("No output file", parser)
    if not args.verbose:
        print("Not verbose")
    return { "input" : args.input_filename,
              "output" : args.output_filename,
              "verbose": True if args.verbose == True else False,
              "transform": args.transform if args.transform else "dot",
              "tile_size": int(args.tile_size) if args.tile_size else 10,
              "n_colours": int(args.n_colours) if args.n_colours else 32,
              "flatten_passes": int(args.flatten_passes),
              "median_size": int(args.flatten_ms) }

def validate_state(init_state:Dict[str, str|int|bool|Any], parser: ArgumentParser):
    if not init_state.get("input"):
        raise ArgError(
            f"No input filename argument provided.", parser
        )
    if not init_state.get("output"):
        raise ArgError(
            f"No output filename argument provided.", parser
        )
    if init_state.get("transform") not in {"dot", "energy", "pixel"}:
        raise ArgError(
            f"Output type is not valid. Expected one of \"dot\", \"energy\" or \"pixel\" for --transform, got {init_state.get("output_type")}",
            parser
        )
    if not init_state.get("tile_size") or (not (isinstance(init_state.get("tile_size"), int))) or init_state.get("tile_size") <= 0:
        raise ArgError(
            f"Invalid tile size argument provided. Must be positive integer. Got {init_state.get("tile_size")}",
            parser
        )
    if not init_state.get("n_colours") or not isinstance(init_state.get("n_colours"), int) or init_state.get("n_colours") <= 0:
        raise ArgError(
            f"Invalid n_colours argument provided. Must be positive integer. Got {init_state.get("n_colours")}",
            parser
        )
    if not init_state.get("flatten_passes") or not isinstance(init_state.get("flatten_passes"), int) or init_state.get("flatten_passes") <= 0:
        raise ArgError(
            f"Invalid flatten_passes argument provided. Must be positive integer. Got {init_state.get("flatten_passes")}",
            parser
        )
    if not init_state.get("median_size") or not isinstance(init_state.get("median_size"), int) or init_state.get("median_size") <= 0:
        raise ArgError(
            f"Invalid median_size argument provided. Must be positive integer. Got {init_state.get("median_size")}",
            parser
        )
    if init_state.get("verbose") == True:
        print("Validated input successfully. This does not mean filenames have been resolved.")
    return

def build_parser(program_info:Dict[str,str]):
    parser = argparse.ArgumentParser(
        prog=program_info.get("program_name", None),
        description=program_info.get("description", None),
        epilog=program_info.get("epilog", None))
    parser.add_argument("--input-filename", "-if")
    parser.add_argument("--output-filename", "-of")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--transform", "-t")
    parser.add_argument("--tile-size", "-ts")
    parser.add_argument("--n-colours", "-c"),
    parser.add_argument("--flatten-passes", "-p", default="6")
    parser.add_argument("--flatten-ms", "-m", default="3")
    return parser

def main(program_info:Dict[str,str]):
    args, parser = parse_args(program_info)
    init_state = build_state(args, parser)
    validate_state(init_state, parser)
    decoder = Decoder(init_state.get("input"), init_state)
    arr = decoder.decode_image_to_rows()
    tiled = decoder.tile_image_rgb(arr, int(init_state.get("tile_size")), "crop")
    # sanity check
    # decoder.untile_image_rgb(tiled)
    untiled = None
    pix = None
    if init_state.get("transform") == "dot":
        dot_filled = decoder.tile_dot_fill(tiled)
        untiled = decoder.untile_image_rgb(dot_filled)
    elif init_state.get("transform") == "energy":
        energy_filled = decoder.tile_channel_energy_fill(tiled)
        untiled = decoder.untile_image_rgb(energy_filled)
    elif init_state.get("transform") == "pixel":
        flat = edge_preserving_flatten(
            arr,
            median_size=int(init_state.get("median_size")),
            passes=int(init_state.get("flatten_passes"))
        )
        pix = pixel_art_dominant_tile_quantize(
            flat,
            tile_size=int(init_state.get("tile_size")),
            n_colours=int(init_state.get("n_colours")),
            dither=True,
            mode="crop")
        decoder.save_rgb_image_per_channel(init_state.get("output"), pix)
    if untiled is not None:
        decoder.save_rgb_image_per_channel(init_state.get("output"), untiled)
    elif pix is not None:
        pass
    else:
        raise ArgError(f"Neither untiled nor pix are not None", parser)
    return

if __name__ == '__main__':
    prog_info = {"program_name" : "ImageRetrofier",
    "description" : "Converts images to ASCII or pixel art.",
    "epilog": ""}
    main(prog_info)

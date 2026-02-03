import os
import argparse
from argparse import ArgumentParser, Namespace
from typing import Dict, Any, Tuple

from arg_exceptions import ArgError
from decoder_core import Decoder


def parse_args(program_info:Dict[str,str]) -> Tuple[Namespace, ArgumentParser]:
    parser = build_parser(program_info)
    args = parser.parse_args()
    # print(args)
    return args, parser

def build_state(args: Namespace, parser: ArgumentParser) -> Dict[str,str|int|None]:
    if not args.input_filename:
        raise ArgError("No input file in arguments", parser)
    if not args.output_filename:
        raise ArgError("No output file in arguments", parser)
    if not args.verbose:
        pass
        # print("Not verbose")
    dither_raw = str(args.dither).strip().lower() if getattr(args, "dither", None) is not None else "none"
    if dither_raw in {"true", "1", "yes", "on"}:
        dither = True
    elif dither_raw in {"none", "false", "0", "no", "off"}:
        dither = False
    else:
        dither = None

    return { "input" : args.input_filename,
              "output" : args.output_filename,
              "verbose": True if getattr(args, "verbose", None) is not None else False,
              "transform": args.transform if getattr(args, "transform", None) is not None else "dot",
              "tile_size": int(args.tile_size) if getattr(args, "tile_size", None) is not None else 10,
              "tile_method": args.tile_method if getattr(args, "tile_method", None) is not None else "mean",
              "energy_method": args.energy_method if getattr(args, "energy_method", None) is not None else "e",
              "n_colours": int(args.n_colours) if getattr(args, "n_colours", None) is not None else 32,
              "flatten_passes": int(args.flatten_passes) if getattr(args, "flatten_passes", None) is not None else 1,
              "median_size": int(args.flatten_ms) if getattr(args, "median_size", None) is not None else 3,
              "dither": dither,
              "dither_raw": args.dither if getattr(args, "dither", None) is not None else "none" }

def validate_state(init_state:Dict[str, str|int|bool|Any], parser: ArgumentParser) -> None:
    if not init_state.get("input"):
        raise ArgError(
            f"No input filename argument provided.", parser
        )
    if not init_state.get("output"):
        raise ArgError(
            f"No output filename argument provided.", parser
        )
    if init_state.get("transform") not in {"dot", "energy", "pixel", "pixel-dot"}:
        raise ArgError(
            f"Output type is not valid. Expected one of \"dot\", \"energy\", \"pixel\" or \"pixel-dot\" for --transform, got {init_state.get("output_type")}",
            parser
        )
    if not init_state.get("tile_size") or not isinstance(init_state.get("tile_size"), int) or init_state.get("tile_size") <= 0:
        raise ArgError(
            f"Invalid tile size argument provided. Must be positive integer. Got {init_state.get("tile_size")}",
            parser
        )
    if not init_state.get("tile_method") or not isinstance(init_state.get("tile_method"), str) or (init_state.get("tile_method") != "mean" and init_state.get("tile_method") != "dot"):
        raise ArgError(
            f"Invalid tile method argument provided. Must be \"mean\" or \"dot\"", parser
        )
    if init_state.get("energy_method") not in {"e", "rms", "mean", "var", "dot"}:
        raise ArgError(
            f"Invalid energy_method argument provided. Must be one of \"e\", \"rms\", \"mean\", \"var\", or \"dot\". Got {init_state.get('energy_method')}",
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
    if not isinstance(init_state.get("dither"), bool):
        raise ArgError(
            f"Invalid dithering argument. Accepts true/false/none (also yes/no/on/off/1/0). Got {init_state.get('dither_raw')}",
            parser
        )
    if init_state.get("verbose") == True:
        print("Validated input successfully. This does not mean filenames have been resolved.")
    return

def build_parser(program_info:Dict[str,str]) -> ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=program_info.get("program_name", None),
        description=program_info.get("description", None),
        epilog=program_info.get("epilog", None))
    parser.add_argument("--input-filename", "-if", help="The path to the input filename.")
    parser.add_argument("--output-filename", "-of", help="The path to the output filename.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output verbose logging to the terminal.")
    parser.add_argument("--transform", "-t", help="The type of image transformation to use. Current options are dot, energy and pixel.")
    parser.add_argument("--tile-size", "-ts", default="10", help="How many pixels (x,y) per tile? Defaults to 10.")
    parser.add_argument("--tile-method", "-tm", default="mean", help="The mathematical method for generating tiles. Should be mean or dot.")
    parser.add_argument("--energy-method", "-em", default="e", help="Energy metric for --transform energy and pixel-dot prepass. One of: e, rms, mean, var, dot.")
    parser.add_argument("--n-colours", "-c", default="32", help="How many colours should the colour palate be constrained to? Defaults to 32."),
    parser.add_argument("--flatten-passes", "-p", default="6", help="How many passes should the flattener make?")
    parser.add_argument("--flatten-ms", "-m", default="3", help="How many adjacent tiles should the flattener look at? Defaults to 3.")
    parser.add_argument("--dither", "-d", default="none", help="Enable dithering on the output image. Accepts true/false/none (also yes/no/on/off/1/0).")
    parser.add_argument("--version", action="version", version="ImageRetrofier 0.2.0")
    return parser

def main(program_info:Dict[str,str]) -> None:
    args, parser = parse_args(program_info)
    init_state = build_state(args, parser)
    validate_state(init_state, parser)
    decoder = Decoder(init_state.get("input"), init_state)
    arr = decoder.decode_image_to_rows()
    # sanity check
    # decoder.untile_image_rgb(tiled)
    untiled = None
    pix = None
    if init_state.get("transform") == "dot":
        tiled = decoder.tile_image_rgb(arr, int(init_state.get("tile_size")), "crop")
        dot_filled = decoder.tile_dot_fill(tiled)
        untiled = decoder.untile_image_rgb(dot_filled)
        decoder.save_rgb_image_per_channel(init_state.get("output"), untiled)
        return
    elif init_state.get("transform") == "energy":
        tiled = decoder.tile_image_rgb(arr, int(init_state.get("tile_size")), "crop")
        energy_filled = decoder.tile_channel_energy_fill(
            tiled,
            method=init_state.get("energy_method")
        )
        untiled = decoder.untile_image_rgb(energy_filled)
    elif init_state.get("transform") == "pixel" or init_state.get("transform") == "pixel-dot":
        if init_state.get("transform") == "pixel-dot":
            tiled = decoder.tile_image_rgb(arr, int(init_state.get("tile_size")), "crop")
            energy_method = init_state.get("energy_method")
            if energy_method in {"mean", "dot"}:
                energy_filled = decoder.tile_channel_energy_fill_divconq_rows(tiled, mode=energy_method)
                arr = decoder.untile_image_rgb(energy_filled)
            else:
                energy_filled = decoder.tile_channel_energy_fill(
                    tiled,
                    method=energy_method
                )
                arr = decoder.untile_image_rgb(energy_filled)
        flat = decoder.edge_preserving_flatten(
            arr,
            median_size=int(init_state.get("median_size")),
            passes=int(init_state.get("flatten_passes"))
        )
        pix = decoder.pixel_art_dominant_tile_quantize(
            flat,
            tile_size=int(init_state.get("tile_size")),
            n_colours=int(init_state.get("n_colours")),
            dither=init_state.get("dither"),
            mode="crop")
        decoder.save_rgb_image_per_channel(init_state.get("output"), pix)
    assert pix is not None or untiled is not None
    return

if __name__ == '__main__':
    prog_info = {"program_name" : "ImageRetrofier",
    "description" : "Converts images to ASCII or pixel art.",
    "epilog": ""}
    try:
        main(prog_info)
    except AssertionError as e:
        raise AssertionError(f"Neither untiled nor pix are not None; program ending anomalously, no image generated.", e)

# ImageRetrofier

## Example commands
* `python3 /src/main.py --input-filename "/home/dorje/Public/retrofier/some-image.jpg" --output-filename "/home/dorje/Public/retrofier/output.jpg" --tile-size 4 --transform pixel --n-colours 16 --flatten-passes 6 --flatten-ms 5 --dither none --crt true --verbose`

Breaking this down:

* `python3 src/main.py` - Invoke your python 3 interpreter to run the file at `src/main.py`. **Note:** Your command line shell must have the ImageRetrofier directory as your current working directory.
* `--input-filename /home/dorje/Public/retrofier/some-image.jpg` - Tell your computer which file path to use as the input image. Your path will likely start with something like C:\ if you are on Windows; this is a Linux example.
* `--output-filename /home/dorje/Public/retrofier/output-image.jpg` - Tell your computer which file path to save the output to.
* `--tile-size 4` - The tile size for pixels, a value of 4 will use a tile size of 4x4 pixels in the input for one tiled pixel in the output.
* `--transform pixel` - The type of image transformation to use. Currently, I recommend sticking with pixel (pixel art)
* `--energy-method e` - For `--transform energy` and the `pixel-dot` prepass, choose the per-channel tile metric: `e`, `rms`, `mean`, `var`, or `dot`. Default is `e`.
* `--dither none` - Dithering toggle for quantization (`--transform pixel` and `pixel-dot`). Accepts `true/false/none` and aliases `yes/no/on/off/1/0`. Default is `none` (off).
* `--crt true` - Apply CRT-style post-processing after transform generation. `true` aliases medium strength. Also accepts `false/none/off` (disabled), presets `low|med|high`, and `low-nb|med-nb|high-nb` (same strengths without border fill or curvature warp).
* `--n-colours 16` - The number of colours to use in the final palate. I recommend 8, 16, or 32.
* `--flatten-pases 6` - How aggressively the image is flattened. I recommend between 3 and 10 passes.
* `--flatten-ms 5` - Must be an odd integer of at least 3. This defines how many adjacent tiles the flattening operation is looking at. Normally I recommend 3 or 5.
* `--verbose` - The amount of command line output you want to see (WIP)

## Dependencies

* Pillow - `pip3 install pillow`

Note that this was written and tested on the Python 3.13 interpreter.

## Changelog

* Added version argument.
* Corrected png input file processing crash issue.

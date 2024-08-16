import datetime
import os.path
import sys

import click

from . import image_fetching
from . import vmt
from . import vtf

#TODO: Refactor to eliminate repeated code
@click.group()
def cli():
    pass

@cli.command(help = "Set TF2 directory.")
@click.argument("tf2_dir", type = click.Path(exists = True))
def set_directory(tf2_dir):
    with open("TF2_DIRECTORY.txt", "wt") as file:
        file.write(tf2_dir)

#TODO: ensure TF2_DIRECTORY.txt is always in the same folder as the script for neatness (default working directory not always proper with vscode)
def get_tf2_directory() -> str | None:
    try: 
        with open ("TF2_DIRECTORY.txt", "rt") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError("TF2 directory not configured! Please run pyspray set-directory before using the script!")


source_option = click.option("-s", "--source", required = True, type = str, help = "The filepath or url of the image/animation. Filepaths are checked before attempting url retrieval.")
format_option = click.option("-f", "--format", type = click.Choice([format.name for format in vtf.ImageFormats if format.implemented], case_sensitive = False), default = "DXT5", help = "The format in which the spray is encoded.")
aspect_ratio_flag = click.option("--preserve_aspect_ratio/--no_preserve_aspect_ratio", is_flag = True, default = True, help = "Whether the aspect ratio of the image should be preserved. If true, the image will be padded to square using either transparent or black padding, determined by the image format's alpha support.")
name_option = click.option("-n", "--name", type = str, help = "The name of the spray.")

@cli.command(help = "Create an animated spray.")
@source_option
@format_option
@aspect_ratio_flag
@name_option
@click.option("-start", "--start_time", type = float, default = 0, help = "The time in seconds at which the animation starts.")
@click.option("-end", "--end_time", type = float, default = sys.float_info.max, help = "The time in seconds at which the animation ends.")
@click.option("-spf", "--seconds_per_frame", type = float, default = 0.2, help = "The number of seconds in between each frame of the animation. The default value is the same framerate as natively possessed by sprays, so any change will speed up or slow down the animation.")
def anim(source, format, preserve_aspect_ratio, name, start_time, end_time, seconds_per_frame):
    tf2_directory = get_tf2_directory()
    images = image_fetching.get_images(source, seconds_per_frame, start_time, end_time)
    flags =     vtf.TextureFlags.CLAMPS |\
                vtf.TextureFlags.CLAMPT |\
                vtf.TextureFlags.NOLOD |\
                vtf.TextureFlags.EIGHTBITALPHA |\
                vtf.TextureFlags.NOMIP 
    animated_spray = vtf.AnimatedSpray(images, vtf.ImageFormats.from_name(format), flags, preserve_aspect_ratio)
    spray_path = os.path.join(tf2_directory, "tf", "materials", "vgui", "logos", f"{name}.vtf")
    animated_spray.save(spray_path)
    vmt.write_vmt_files(name, tf2_directory)

@cli.command(help = "Create a spray that changes with distance.")
@click.option("-s", "--source", multiple = True, required = True, type = str, help = "The filepath or url of the image. Filepaths are checked before attempting url retrieval. The first source will be shown from furthest away, and the last source will be shown when you get closest to the spray.")
@format_option
@aspect_ratio_flag
@name_option
def fade(source, format, preserve_aspect_ratio, name):
    tf2_directory = get_tf2_directory()
    images = [image_fetching.get_images(path)[0] for path in source]
    name = name or datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    flags =     vtf.TextureFlags.CLAMPS |\
                vtf.TextureFlags.CLAMPT |\
                vtf.TextureFlags.NOLOD |\
                vtf.TextureFlags.EIGHTBITALPHA
    fade_spray = vtf.FadeSpray(images, vtf.ImageFormats.from_name(format), flags, preserve_aspect_ratio)
    spray_path = os.path.join(tf2_directory, "tf", "materials", "vgui", "logos", f"{name}.vtf")
    fade_spray.save(spray_path)
    vmt.write_vmt_files(name, tf2_directory)

@cli.command(help = "Create a spray using a static image.")
@source_option
@format_option
@aspect_ratio_flag
@name_option
def static(source, format, preserve_aspect_ratio, name):
    tf2_directory = get_tf2_directory()
    images = image_fetching.get_images(source)
    first_frame = images[0]
    name = name or datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    flags =     vtf.TextureFlags.CLAMPS |\
                vtf.TextureFlags.CLAMPT |\
                vtf.TextureFlags.NOLOD |\
                vtf.TextureFlags.EIGHTBITALPHA |\
                vtf.TextureFlags.NOMIP 
    static_spray = vtf.AnimatedSpray([first_frame], vtf.ImageFormats.from_name(format), flags, preserve_aspect_ratio)
    spray_path = os.path.join(tf2_directory, "tf", "materials", "vgui", "logos", f"{name}.vtf")
    static_spray.save(spray_path)
    vmt.write_vmt_files(name, tf2_directory)
    
if __name__ == "__main__":
    cli()
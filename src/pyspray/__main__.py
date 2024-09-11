import datetime
import os.path
import platform
import sys

import click

from . import image_fetching
from . import vmt
from . import vtf

@click.group()
def cli():
    pass

def get_module_directory() -> str:
    module_path = os.path.abspath(__file__)
    module_dir = os.path.dirname(module_path)
    return module_dir
@cli.command(help = "Set TF2 directory.")
@click.argument("tf2_directory", type = click.Path(exists = True, dir_okay = True, file_okay = False))
def set_directory(tf2_directory):
    #Check to make sure this actually the Team Fortress 2 directory by checking for the existance of the executable
    executable_name = "tf_win64.exe" if platform.system() == "Windows" else "tf_linux64"
    executable_path = os.path.join(tf2_directory, executable_name) 
    if not os.path.exists(executable_path):
        raise click.BadParameter("Team Fortress 2 executable not detected, this is not the Team Fortress 2 directory!")
    module_dir = get_module_directory()
    tf2_directory_file = os.path.join(module_dir, "TF2_DIRECTORY.txt")
    with open(tf2_directory_file, "wt") as file:
        file.write(tf2_directory)

def get_tf2_directory() -> str:
    try: 
        module_dir = get_module_directory()
        tf2_dir_file = os.path.join(module_dir, "TF2_DIRECTORY.txt")
        with open (tf2_dir_file, "rt") as file:
            return file.read()
    except FileNotFoundError:
        raise click.UsageError("TF2 directory not configured! Please run pyspray set-directory before generating any sprays!")

def ensure_spray_directories(tf2_directory: str):
    ui_dir = os.path.join(tf2_directory, "tf", "materials", "vgui", "logos", "ui")
    os.makedirs(ui_dir, exist_ok = True)


source_argument = click.argument("source", type = str)
format_option = click.option("-f", "--format", type = click.Choice([format.name for format in vtf.ImageFormats if format.implemented], case_sensitive = False), default = "DXT5", help = "The format in which the spray is encoded.")
aspect_ratio_flag = click.option("--preserve_aspect_ratio/--no_preserve_aspect_ratio", is_flag = True, default = True, help = "Whether the aspect ratio of the image should be preserved. If true, the image will be padded to square using either transparent or black padding, determined by the image format's alpha support.")
dry_run_flag = click.option("--dry/--no_dry", is_flag = True, default = False, help = "If enabled, spray files will not be generated. Best for previewing how a spray might look with specifc options.")
show_flag = click.option("--show/--no_show", is_flag = True, default = False, help = "If enabled, a preview of the spray will be shown. Best used if you want to preview a spray while also generating files.")
name_option = click.option("-n", "--name", type = str, help = "The name of the spray.", default = lambda : datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

@cli.command(help = "Create an animated spray. Argument is a filepath or a url. Local filepaths are checked before attempting url retrieval.")
@source_argument
@format_option
@aspect_ratio_flag
@dry_run_flag
@show_flag
@name_option
@click.option("-start", "--start_time", type = float, default = 0, help = "The time in seconds at which the animation starts.")
@click.option("-end", "--end_time", type = float, default = sys.float_info.max, help = "The time in seconds at which the animation ends.")
@click.option("-spf", "--seconds_per_frame", type = float, default = 0.2, help = "The number of seconds in between each frame of the animation. The default value is the same framerate as natively possessed by sprays, so any change will speed up or slow down the animation.")
def anim(source, format, preserve_aspect_ratio, dry, show, name, start_time, end_time, seconds_per_frame):
    tf2_directory = get_tf2_directory()
    ensure_spray_directories(tf2_directory)
    images = image_fetching.get_images(source, seconds_per_frame, start_time, end_time)
    flags =     vtf.TextureFlags.CLAMPS |\
                vtf.TextureFlags.CLAMPT |\
                vtf.TextureFlags.NOLOD |\
                vtf.TextureFlags.EIGHTBITALPHA |\
                vtf.TextureFlags.NOMIP 

    animated_spray = vtf.AnimatedSpray(images, vtf.ImageFormats.from_name(format), flags, preserve_aspect_ratio)
    if show:
        preview = vtf.AnimatedSpray.frombytes(animated_spray.tobytes())
        preview.show()
    if not dry: 
        spray_path = os.path.join(tf2_directory, "tf", "materials", "vgui", "logos", f"{name}.vtf")
        animated_spray.save(spray_path)
        vmt.write_vmt_files(name, tf2_directory)

@cli.command(help = "Create a spray that changes with distance. Arguments are filepaths or urls. Filepaths are checked before attempting url retrieval. The first source will be shown from furthest away, and the last source will be shown when you get closest to the spray." )
@click.argument("source", nargs = -1, type = str)
@format_option
@aspect_ratio_flag
@dry_run_flag
@show_flag
@name_option
def fade(source, format, preserve_aspect_ratio, dry, show, name):
    tf2_directory = get_tf2_directory()
    ensure_spray_directories(tf2_directory)
    images = [image_fetching.get_images(path)[0] for path in source]
    flags =     vtf.TextureFlags.CLAMPS |\
                vtf.TextureFlags.CLAMPT |\
                vtf.TextureFlags.NOLOD |\
                vtf.TextureFlags.EIGHTBITALPHA
    
    fade_spray = vtf.FadeSpray(images, vtf.ImageFormats.from_name(format), flags, preserve_aspect_ratio)
    if show:
        preview = vtf.FadeSpray.frombytes(fade_spray.tobytes())
        preview.show()
    if not dry:    
        spray_path = os.path.join(tf2_directory, "tf", "materials", "vgui", "logos", f"{name}.vtf")
        fade_spray.save(spray_path)
        vmt.write_vmt_files(name, tf2_directory)

@cli.command(help = "Create a spray using a static image. Argument is a filepath or a url. Local filepaths are checked before attempting url retrieval.")
@source_argument
@format_option
@aspect_ratio_flag
@dry_run_flag
@show_flag
@name_option
def static(source, format, preserve_aspect_ratio, dry, show, name):
    tf2_directory = get_tf2_directory()
    ensure_spray_directories(tf2_directory)
    images = image_fetching.get_images(source)
    first_frame = images[0]
    flags =     vtf.TextureFlags.CLAMPS |\
                vtf.TextureFlags.CLAMPT |\
                vtf.TextureFlags.NOLOD |\
                vtf.TextureFlags.EIGHTBITALPHA |\
                vtf.TextureFlags.NOMIP 
    
    static_spray = vtf.AnimatedSpray([first_frame], vtf.ImageFormats.from_name(format), flags, preserve_aspect_ratio)
    if show:
        preview = vtf.AnimatedSpray.frombytes(static_spray.tobytes())
        preview.show()
    if not dry:
        spray_path = os.path.join(tf2_directory, "tf", "materials", "vgui", "logos", f"{name}.vtf")
        static_spray.save(spray_path)
        vmt.write_vmt_files(name, tf2_directory)
if __name__ == "__main__":
    cli()
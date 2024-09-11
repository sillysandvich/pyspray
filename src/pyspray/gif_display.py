from PIL import Image, ImageShow

NativeViewer = type(ImageShow._viewers[0])
class GifViewer(NativeViewer):
    format = "GIF"
gif_viewer = GifViewer()
def show_gif(frame_list: list[Image.Image], frame_duration: int) -> None:
    filename = frame_list[0]._dump(format = "gif", save_all = True, append_images = frame_list[1:], loop = 0, duration = frame_duration)
    gif_viewer.show_file(filename)
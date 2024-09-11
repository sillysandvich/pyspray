from dataclasses import dataclass
import enum
import functools
import io
import math
import struct
from typing import ClassVar

import numpy
from PIL import Image
from quicktex import dds, s3tc

from .gif_display import show_gif

class ImageFormats(enum.Enum):
    def __new__(cls, value: int, bits_per_pixel: int, block_side_length: int, implemented: bool):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.bits_per_pixel = bits_per_pixel
        obj.block_side_length = block_side_length
        obj.implemented = implemented
        return obj
    @classmethod
    def from_name(cls, name):
        name = name.upper()
        for member in cls.__members__.values():
            if member.name == name:
                return member
        raise ValueError(f"No enum member with name {name}")
    
    
    NONE = -1, None, None, False
    RGBA8888 = 0, 32, 1, True
    ABGR8888 = 1, 32, 1, True
    RGB888 = 2, 24, 1, True
    BGR888 = 3, 24, 1, True
    RGB565 = 4, 16, 1, False
    I8 = 5, 8, 1, True
    IA88 = 6, 16, 1, True
    P8 = 7, 8, 1, False
    A8 = 8, 8, 1, True
    RGB888_BLUESCREEN = 9, 24, 1, True
    BGR888_BLUESCREEN = 10, 24, 1, True
    ARGB8888 = 11, 32, 1, True
    BGRA8888 = 12, 32, 1, True
    DXT1 = 13, 4, 4, True
    DXT3 = 14, 8, 4, True
    DXT5 = 15, 8, 4, True
    BGRX8888 = 16, 32, 1, True
    BGR565 = 17, 16, 1, True
    BGRX5551 = 18, 16, 1, True
    BGRA4444 = 19, 16, 1, True
    DXT1_ONEBITALPHA = 20, 4, 4, False
    BGRA5551 = 21, 16, 1, True
    UV88 = 22, 16, 1, False
    UVWQ8888 = 23, 32, 1, False
    RGBA16161616F = 24, 64, 1, False
    RGBA16161616 = 25, 64, 1, False
    UVLX8888 = 26, 32, 1, False
class TextureFlags(enum.Flag):
    POINTSAMPLE = 0x00000001
    TRILINEAR = 0x00000002
    CLAMPS = 0x00000004
    CLAMPT = 0x00000008
    ANISOTROPIC = 0x00000010
    HINT_DXT5 = 0x00000020
    PWL_CORRECTED = 0x00000040
    NORMAL = 0x00000080
    NOMIP = 0x00000100
    NOLOD = 0x00000200
    ALL_MIPS = 0x00000400
    PROCEDURAL = 0x00000800

    ONEBITALPHA = 0x00001000
    EIGHTBITALPHA = 0x00002000

    ENVMAP = 0x00004000
    RENDERTARGET = 0x00008000
    DEPTHRENDERTARGET = 0x00010000
    NODEBUGOVERRIDE = 0x00020000
    SINGLECOPY = 0x00040000
    PRE_SRGB = 0x00080000

    UNUSED_00100000 = 0x00100000
    UNUSED_00200000 = 0x00200000
    UNUSED_00400000 = 0x00400000

    NODEPTHBUFFER = 0x00800000

    UNUSED_01000000 = 0x01000000

    CLAMPU = 0x02000000
    VERTEXTEXTURE = 0x04000000
    SSBUMP = 0x08000000            

    UNUSED_10000000 = 0x10000000

    BORDER = 0x20000000

    UNUSED_40000000 = 0x40000000
    UNUSED_80000000 = 0x80000000

from time import time

class VTFFile:
    @dataclass(kw_only=True)
    class VTFHeader:
        STRUCT_FORMAT_STRING: ClassVar[str] = "=4s2IIHHIHH4s3f4sfiBiBBx"
        VTF_HEADER_BYTES: ClassVar[int] = 64

        secret_message: bytes = b"YIFFY:3c"

        major_version: int = 7
        minor_version: int = 1
        header_size: int = VTF_HEADER_BYTES
        flags: TextureFlags
        high_res_width: int = 0
        high_res_height: int = 0
        high_res_format: ImageFormats = ImageFormats.DXT1
        low_res_width: int = 0
        low_res_height: int = 0
        low_res_format: ImageFormats = ImageFormats.DXT1
        num_frames: int = 1
        first_frame: int = 0
        num_mipmaps: int = 1
        reflectivity_vector: tuple[int, int, int] = (0, 0, 0)
        bumpmap_scale: int = 0
        def tobytes(self) -> bytes:
            return struct.pack(
            self.STRUCT_FORMAT_STRING,
            b"VTF\0", #signature
            self.major_version,
            self.minor_version,
            self.header_size,
            self.high_res_width, 
            self.high_res_height,
            self.flags.value,
            self.num_frames,
            self.first_frame,
            self.secret_message[0:4], #Normally padding, encoding secret message
            *self.reflectivity_vector, #unpack reflectivity vector
            self.secret_message[4:8], #Normally padding, encoding secret message
            self.bumpmap_scale,
            self.high_res_format.value,
            self.num_mipmaps,
            self.low_res_format.value,
            self.low_res_width,
            self.low_res_height,
        )
        @classmethod
        def frombytes(cls: type, bytes: bytes):
            header_object = object.__new__(cls)
            unpacked_data = struct.unpack(cls.STRUCT_FORMAT_STRING, bytes)
            header_object.signature = unpacked_data[0]
            header_object.major_version = unpacked_data[1]
            header_object.minor_version = unpacked_data[2]
            header_object.header_size = unpacked_data[3]
            header_object.high_res_width = unpacked_data[4]
            header_object.high_res_height = unpacked_data[5]
            header_object.flags = TextureFlags(unpacked_data[6])
            header_object.num_frames = unpacked_data[7]
            header_object.first_frame = unpacked_data[8]
            header_object.secret_message = unpacked_data[9] + unpacked_data[13]
            header_object.reflectivity_vector = unpacked_data[10:13]
            header_object.bumpmap_scale = unpacked_data[14]
            header_object.high_res_format = ImageFormats(unpacked_data[15])
            header_object.num_mipmaps = unpacked_data[16]
            header_object.low_res_format = ImageFormats(unpacked_data[17])
            header_object.low_res_width = unpacked_data[18]
            header_object.low_res_height = unpacked_data[19]  
            return header_object 

    MAX_BYTES = 512 * 1024
    def __init__(self, **kwargs) -> None:
        self.header = self.VTFHeader(**kwargs)
        self.high_res_data = numpy.empty((self.header.num_mipmaps, self.header.num_frames), dtype='object')
        self.low_res_data = Image.new("RGBA", (0,0))
    @staticmethod
    def reorder_channels(img: Image.Image, order: str, inverse: bool = False) -> Image.Image:
            channels = img.split()
            band_names = img.getbands()
            indices = [band_names.index(channel) for channel in order]
            #Undo the reordering specified by the order string, indices patterns are not necessarily their own inverse so this is necessary
            if inverse:
                new_indices = [None for _ in indices]
                for pos, index in enumerate(indices):
                    new_indices[index] = pos
                indices = new_indices
            ordered_channels = tuple(channels[index] for index in indices)
            reordered_image = Image.merge(img.mode, ordered_channels)
            for channel in ordered_channels:
                channel.close()
            return reordered_image
    @staticmethod
    def get_raw_image_data(img: Image.Image, format: ImageFormats) -> bytes:
        #NOTE: FIRST channel is stored in the LEAST significant bits
        def trim_channels_uint16(img: Image.Image, channel_depths: tuple[int, ...]) -> bytes:
            assert sum(channel_depths) == 16, "channel depths do not sum to 16 bits!"
            eight_bit_color_array = numpy.asarray(img)
            #Bitshift each channel such that all most significant bits fit within bit depth
            trimmed_array = numpy.right_shift(eight_bit_color_array, [8 - channel_depth for channel_depth in channel_depths]).astype(numpy.uint16)
            #Take bitwise or of each channel, shifted by appropriate amount
            trimmed_data = functools.reduce(lambda x, y: x | y, (trimmed_array[:, :, idx] << sum(channel_depths[:idx]) for idx in range(len(channel_depths))))
            return trimmed_data.tobytes()     
        reorder_channels = __class__.reorder_channels
        match format:
            case ImageFormats.RGBA8888: #Tested, works
                with img.convert("RGBA") as converted:
                    return converted.tobytes()
            case ImageFormats.ABGR8888: #Tested, works
                with img.convert("RGBA") as converted, reorder_channels(converted, "ABGR") as reordered:
                    return reordered.tobytes()
            case ImageFormats.RGB888: #Tested, works
                with img.convert("RGB") as converted:
                    return converted.tobytes()
            case ImageFormats.BGR888: #Tested, works
                with img.convert("RGB") as converted, reorder_channels(converted, "BGR") as reordered:
                    return reordered.tobytes()
            case ImageFormats.RGB565:
                with img.convert("RGB") as converted:
                    #Not working, might be error with source engine?
                    #return trim_channels_uint16(converted, (5, 6, 5))
                    raise ValueError("RGB565 does not work, use BGR565 instead")
            case ImageFormats.I8: #Tested, works
                with img.convert("L") as converted:
                    return converted.tobytes()
            case ImageFormats.IA88: #Tested, works
                with img.convert("L") as greyscale, img.convert("RGBA") as rgba:
                    greyscale_data = greyscale.tobytes()
                    channels = rgba.split() 
                    alpha_data = channels[-1].tobytes()
                    for channel in channels:
                        channel.close()
                    return b"".join(byte.to_bytes() for pixel in zip(greyscale_data, alpha_data) for byte in pixel)
            case ImageFormats.P8:
                #TODO: Not working, find docs for vtf paletted mode 
                #NOTE: Unsupported by VTFEdit? Might not be able to implement
                with img.convert("P") as converted:
                    palette_data = b"".join(value.to_bytes() for value in converted.getpalette())
                    color_data = converted.tobytes()
                    return palette_data + color_data
            case ImageFormats.A8: #Tested, works
                with img.convert("RGBA") as converted:
                    channels = converted.split()
                    alpha_data = channels[-1].tobytes()
                    for channel in channels:
                        channel.close()
                    return alpha_data
            case ImageFormats.RGB888_BLUESCREEN: #Tested, works
                with img.convert("RGB") as converted:
                    return converted.tobytes()
            case ImageFormats.BGR888_BLUESCREEN: #Tested, works
                with img.convert("RGB") as converted, reorder_channels(converted, "BGR") as reordered:
                    return reordered.tobytes()
            case ImageFormats.ARGB8888: #Tested, works
                with img.convert("RGBA") as converted, reorder_channels(converted, "ARGB") as reordered:
                    return reordered.tobytes()
            case ImageFormats.BGRA8888: #Tested, works
                with img.convert("RGBA") as converted, reorder_channels(converted, "BGRA") as reordered:
                    return reordered.tobytes()
            #TODO: Switch to library with support for DXT1a encoding
            case ImageFormats.DXT1: #Tested, works
                with img.convert("RGBA") as converted:
                    encoder = s3tc.bc1.BC1Encoder()
                    dds_file = dds.encode(converted, encoder, "DXT1", 1)
                    ret = dds_file.textures[0].tobytes()
            case ImageFormats.DXT3: #Tested, works
                with img.convert("RGBA") as converted:
                    encoder = s3tc.bc1.BC1Encoder()
                    dds_file = dds.encode(converted, encoder, "DXT1", 1)
                    color_data = dds_file.textures[0].tobytes()
                    #TODO: Switch to library with built-in BC2 encoding. Current solution is to extract and organize alpha data via bit manipulation
                    pixel_array = numpy.asarray(converted)  
                    
                    alpha_array = pixel_array[:, :, 3]
                    reordered_alpha_data = numpy.concatenate([alpha_array[y:y+4, x:x+4] 
                                                            for y in range(0, alpha_array.shape[0], 4) 
                                                                for x in range(0, alpha_array.shape[1], 4)]).ravel()
                    high_alpha_bits = (reordered_alpha_data[0::2] >> 4) << 4
                    low_alpha_bits = reordered_alpha_data[1::2] >> 4
                    alpha_data = (high_alpha_bits | low_alpha_bits).tobytes()

                    return b"".join(alpha_data[block:block + 8] + color_data[block:block + 8] for block in range(0, len(color_data), 8))
            case ImageFormats.DXT5: #Tested, works
                with img.convert("RGBA") as converted:
                    encoder = s3tc.bc3.BC3Encoder()
                    dds_file = dds.encode(converted, encoder, "DXT5", 1)
                    return dds_file.textures[0].tobytes()
            case ImageFormats.BGRX8888: #Tested, works
                #Alpha data ignored by game engine, assumed to be 255 universally
                with img.convert("RGBA") as converted, reorder_channels(converted, "BGRA") as reordered:
                    return reordered.tobytes()
            #NOTE: It looks like for some reason when dealing with 16 bit formats, the first channel is stored in the least significant bits?
            #NOTE: trim_channels_uint16 function accounts for this, but it is a weird implementation
            case ImageFormats.BGR565: #Tested, works
                with img.convert("RGB") as converted, reorder_channels(converted, "BGR") as reordered:
                    return trim_channels_uint16(reordered, (5, 6, 5))       
            case ImageFormats.BGRX5551: #Tested, works
                #Alpha data ignored by game engine, assumed to be 1 universally
                with img.convert("RGBA") as converted, reorder_channels(converted, "BGRA") as reordered:
                    return trim_channels_uint16(reordered, (5, 5, 5, 1))   
            case ImageFormats.BGRA4444: #Tested, works
                with img.convert("RGBA") as converted, reorder_channels(converted, "BGRA") as reordered:
                    return trim_channels_uint16(reordered, (4, 4, 4, 4))
            case ImageFormats.DXT1_ONEBITALPHA: #Broken in source
                raise ValueError("DXT1_ONEBITALPHA does not work, use DXT1 instead")
            case ImageFormats.BGRA5551: #Tested, works
                with img.convert("RGBA") as converted, reorder_channels(converted, "BGRA") as reordered:
                    return trim_channels_uint16(reordered, (5, 5, 5, 1))
            case _:
                raise ValueError(f"Unimplemented Data Format: {format}")
        return ret
    @staticmethod 
    def get_image_from_raw_data(image_bytes: bytes, format: ImageFormats, width: int, height: int) -> Image.Image:
        reorder_channels = __class__.reorder_channels
        def extract_image_array_from_uint16(image_bytes: bytes, channel_depths: tuple[int, ...], width: int, height: int) -> numpy.ndarray:
            assert sum(channel_depths) == 16, "channel depths do not sum to 16 bits!"
            image_data_uint16 = numpy.frombuffer(image_bytes, dtype = numpy.uint16).reshape(height, width)

            #TODO: Use different algorithm from simple zero-extension to reverse truncation such that values match those in-game as closely as possible
            #Band-aid fix for one-bit values for now
            channel_data_list = [image_data_uint16 >> sum(channel_depths[:idx]) << 8 - depth for idx, depth in enumerate(channel_depths)]
            return numpy.stack(channel_data_list, dtype = numpy.uint8, axis = 2)
    
        #Early return for empty images to prevent quicktex from getting fussy
        if width * height == 0:
            return Image.new("RGBA", (0, 0))
        match format:
            case ImageFormats.RGBA8888:
                return Image.frombytes("RGBA", (width, height), image_bytes)
            case ImageFormats.ABGR8888:
                with Image.frombytes("RGBA", (width, height), image_bytes) as img:
                    return reorder_channels(img, "ABGR", True)
            case ImageFormats.RGB888: 
                return Image.frombytes("RGB", (width, height), image_bytes)
            case ImageFormats.BGR888:
                with Image.frombytes("RGB", (width, height), image_bytes) as img:
                    return reorder_channels(img, "BGR", True)
            case ImageFormats.RGB565:
                image_array = extract_image_array_from_uint16(image_bytes, (5, 6, 5), width, height)
                return Image.fromarray(image_array, "RGB")
            case ImageFormats.I8:
                return Image.frombytes("L", (width, height), image_bytes)
            case ImageFormats.IA88:
                return Image.frombytes("LA", (width, height), image_bytes)
            case ImageFormats.A8:
                image_data = numpy.zeros((height, width, 4), dtype = numpy.uint8)
                image_data[:, :, 3] = numpy.frombuffer(image_bytes, dtype = numpy.uint8).reshape((height, width))
                return Image.fromarray(image_data, "RGBA")
            case ImageFormats.RGB888_BLUESCREEN | ImageFormats.BGR888_BLUESCREEN:
                color_data = numpy.frombuffer(image_bytes, dtype = numpy.uint8).reshape((height, width, 3))
                if format == ImageFormats.BGR888_BLUESCREEN:
                    #Reorder to RGB ordering
                    color_data = color_data.copy()
                    color_data[:, :, [0, 2]] = color_data[:, :, [2, 0]]
                is_blue = numpy.logical_and.reduce([color_data[:, :, 0] == 0, color_data[:, :, 1] == 0, color_data[:, :, 2] == 255])
                #Blue is displayed as transparent
                alpha_data = numpy.where(is_blue, 0, 255).astype(numpy.uint8)
                image_data = numpy.concatenate((color_data, alpha_data[:, :, numpy.newaxis]), axis = 2)
                return Image.fromarray(image_data, "RGBA")
            case ImageFormats.ARGB8888:
                with Image.frombytes("RGBA", (width, height), image_bytes) as img:
                    return reorder_channels(img, "ARGB", True)
            case ImageFormats.BGRA8888:
                with Image.frombytes("RGBA", (width, height), image_bytes) as img:
                    return reorder_channels(img, "BGRA", True)
            case ImageFormats.DXT1:
                texture = s3tc.bc1.BC1Texture.from_bytes(image_bytes, width, height)
                decoder = s3tc.bc1.BC1Decoder()
                texture = decoder.decode(texture)
                return Image.frombuffer('RGBA', texture.size, texture)
            case ImageFormats.DXT3:
                color_bytes = b''.join(image_bytes[index:index + 8] for index in range(8, len(image_bytes), 16))
                alpha_bytes = b''.join(image_bytes[index:index + 8] for index in range(0, len(image_bytes), 16))
                alpha_array = numpy.frombuffer(alpha_bytes, dtype = numpy.uint8)
                alpha_high_bit_array = alpha_array & 0b11110000
                alpha_low_bit_array = (alpha_array & 0b00001111) << 4
                alpha_array = numpy.empty(height * width, dtype = numpy.uint8)
                alpha_array[0::2] = alpha_high_bit_array
                alpha_array[1::2] = alpha_low_bit_array
                alpha_array = alpha_array.reshape(height, width)
                alpha_image = Image.fromarray(alpha_array, "L")

                color_texture = s3tc.bc1.BC1Texture.from_bytes(color_bytes, width, height)
                decoder = s3tc.bc1.BC1Decoder()
                texture = decoder.decode(color_texture)
                image = Image.frombuffer("RGBA", texture.size, texture)
                image.putalpha(alpha_image)
                return image
            case ImageFormats.DXT5:
                texture = s3tc.bc3.BC3Texture.from_bytes(image_bytes, width, height)
                decoder = s3tc.bc3.BC3Decoder()
                texture = decoder.decode(texture)
                return Image.frombuffer('RGBA', texture.size, texture)
            case ImageFormats.BGRX8888:
                with Image.frombytes("RGBX", (width, height), image_bytes) as img:
                    return reorder_channels(img, "BGRX", True)
            case ImageFormats.BGR565: 
                image_array = extract_image_array_from_uint16(image_bytes, (5, 6, 5), width, height)
                with Image.fromarray(image_array, "RGB") as img:
                    return reorder_channels(img, "BGR", True)      
            case ImageFormats.BGRX5551:
                image_array = extract_image_array_from_uint16(image_bytes, (5, 5, 5, 1), width, height)
                with Image.fromarray(image_array, "RGBX") as img:
                    return reorder_channels(img, "BGRX", True)  
            case ImageFormats.BGRA4444:
                image_array = extract_image_array_from_uint16(image_bytes, (4, 4, 4, 4), width, height)
                with Image.fromarray(image_array, "RGBA") as img:
                    return reorder_channels(img, "BGRA", True)  
            case ImageFormats.DXT1_ONEBITALPHA:
                raise ValueError("DXT1_ONEBITALPHA does not work, use DXT1 instead")
            case ImageFormats.BGRA5551:
                image_array = extract_image_array_from_uint16(image_bytes, (5, 5, 5, 1), width, height)
                #Band-aid fix, one-bit values must be either 0 or 255
                alpha_array = image_array[:, :, 3]
                image_array[:, :, 3][alpha_array > 0] = 255
                with Image.fromarray(image_array, "RGBA") as img:
                    return reorder_channels(img, "BGRA", True)  
            case _:
                raise ValueError(f"Unimplemented Data Format: {format}")
            

    @staticmethod
    def resize_image(image: Image.Image, side_length: int, preserve_aspect_ratio: bool):
        if preserve_aspect_ratio:
            if image.width > image.height:
                w, h = side_length, round(side_length * (image.height / image.width))
            else:
                w, h = round(side_length * (image.width / image.height)), side_length
            with image.resize((w, h)) as resized_image: 
                x_offset = (side_length - w) // 2
                y_offset = (side_length - h) // 2
                result = Image.new("RGBA", (side_length, side_length), (0, 0, 0, 0))
                result.paste(resized_image, (x_offset, y_offset))
        else:
            result = image.resize((side_length, side_length))
        return result
    
    def assert_full(self):
        assert None not in self.high_res_data, "All mipmaps and frames must be filled!"
    def tobytes(self) -> bytes:
        self.assert_full()
        header_bytes = [self.header.tobytes()]
        low_res_bytes = [self.get_raw_image_data(self.low_res_data, self.header.low_res_format)] if 0 not in self.low_res_data.size else []
        high_res_bytes = [self.get_raw_image_data(img, self.header.high_res_format) for img in self.high_res_data.ravel()]
        file_data = b''.join(header_bytes + low_res_bytes + high_res_bytes)
        return file_data
    #NOTE: only implemented for VTF 7.1 as of right now
    @classmethod
    def frombytes(cls: type, vtf_bytes: bytes) -> 'VTFFile':
        vtf_data = io.BytesIO(vtf_bytes)  
        vtf_object = object.__new__(cls)

        header_bytes = vtf_data.read(cls.VTFHeader.VTF_HEADER_BYTES)
        header = cls.VTFHeader.frombytes(header_bytes)

        vtf_object.header = header
        num_low_res_bytes = header.low_res_width * header.low_res_height * header.low_res_format.bits_per_pixel // 8
        low_res_bytes = vtf_data.read(num_low_res_bytes)
        vtf_object.low_res_data = cls.get_image_from_raw_data(low_res_bytes, header.low_res_format, header.low_res_width, header.low_res_height)

        vtf_object.high_res_data = numpy.empty((header.num_mipmaps, header.num_frames), dtype = 'object')
        for mipmap in range(header.num_mipmaps):
            resize_factor = 2 ** (header.num_mipmaps - mipmap - 1) #Smallest mipmaps are first
            mipmap_width = max(header.high_res_width // resize_factor,  1)
            mipmap_height = max(header.high_res_height // resize_factor, 1)

            bytes_per_frame = mipmap_width * mipmap_height * header.high_res_format.bits_per_pixel // 8
            for frame in range(header.num_frames):
                frame_bytes = vtf_data.read(bytes_per_frame)
                vtf_object.high_res_data[mipmap, frame] = cls.get_image_from_raw_data(frame_bytes, header.high_res_format, mipmap_width, mipmap_height)
        
        return vtf_object
    def save(self, path) -> None:
        with open(path, "wb") as file:
            file.write(self.tobytes())
    def calculate_max_high_res_bits(self) -> int:
        VTF_HEADER_BITS = self.header.VTF_HEADER_BYTES * 8
        MAX_VTF_BITS = self.MAX_BYTES * 8

        low_res_bits = self.header.low_res_height * self.header.low_res_width * self.header.low_res_format.bits_per_pixel
        max_high_res_bits = MAX_VTF_BITS - VTF_HEADER_BITS - low_res_bits
        return max_high_res_bits
        
    def normalize_mipmap_sizes(self) -> None:
        self.assert_full()
        #Iterate over mipmaps in order of decreasing size
        for idx, mipmap in enumerate(self.high_res_data[::-1]):
            resize_factor = 2 ** idx
            mipmap_width = max(self.header.high_res_width // resize_factor,  1)
            mipmap_height = max(self.header.high_res_height // resize_factor, 1)
            for idx, frame in enumerate(mipmap):
                mipmap[idx] = frame.resize((mipmap_width, mipmap_height))
    def show(self) -> None:
        raise NotImplementedError("VTFFile.show method only implemented on subclasses!")

class FadeSpray(VTFFile):
    SMALLEST_MIPMAP_SIDE_LENGTH = 32
    def calculate_num_mipmaps(self) -> int:
        MIPMAP_PIXEL_RATIO = 4
 
        max_high_res_bits = self.calculate_max_high_res_bits()
        max_high_res_pixels = max_high_res_bits // self.header.high_res_format.bits_per_pixel

        max_high_res_pixels //= self.header.num_frames

        #Mipmap pixel sum is defined by a geometric series, solve for maximum number of mipmaps that will still fit within 512 KB VTF file
        num_mipmaps = math.floor(math.log(1 - ((max_high_res_pixels * (1 - MIPMAP_PIXEL_RATIO)) / (self.SMALLEST_MIPMAP_SIDE_LENGTH ** 2)), MIPMAP_PIXEL_RATIO))
        return num_mipmaps
    def __init__(self, images: Image.Image, format: ImageFormats, flags: TextureFlags, preserve_aspect_ratio: bool):
        #Base VTFFile must be inited before calculating minimum mipmap sizes
        VTFFile.__init__(self, high_res_format = format, flags = flags)
        self.header.num_mipmaps = self.calculate_num_mipmaps()
        assert self.header.num_mipmaps >= len(images), "Too many images for faded spray!"
        self.header.high_res_width = self.header.high_res_height = self.SMALLEST_MIPMAP_SIDE_LENGTH * 2 ** (self.header.num_mipmaps - 1)
        #Resize high res data array, originally sized for one mipmap
        self.high_res_data = numpy.empty((self.header.num_mipmaps, self.header.num_frames), dtype='object')
        for idx, image in enumerate(reversed(images)):
            self.high_res_data[-(1 + idx), :] = [image for _ in range(self.header.num_frames)]
        #If mipmaps are activated then the game will attempt to read all mipmaps down the minumum. Smallest mipmap will be used for all smaller mipmaps to avoid reading garbage data
        for mipmap in self.high_res_data:
            for idx, frame in enumerate(mipmap):
                if frame is None:
                    mipmap[idx] = images[0]
        
        
        for idx, mipmap in enumerate(self.high_res_data):
            mipmap_side_length = self.SMALLEST_MIPMAP_SIDE_LENGTH * (2 ** idx)
            for idx2, frame in enumerate(mipmap):
                mipmap[idx2] = self.resize_image(frame, mipmap_side_length, preserve_aspect_ratio)
    def show(self) -> None:
        with Image.new("RGBA", (self.header.high_res_width * self.header.num_mipmaps, self.header.high_res_height)) as image:
            for idx, mipmap in enumerate(self.high_res_data[::-1]):
                #For faded animated images, this would need to be changed
                with mipmap[0].resize((self.header.high_res_width, self.header.high_res_height), resample = Image.NEAREST) as frame:
                    image.paste(frame, (self.header.high_res_width * idx, 0))
            image.show()

            


class AnimatedSpray(VTFFile):
    SECONDS_PER_FRAME = 0.2
    def calculate_frame_side_length(self) -> int:
        max_high_res_bits = self.calculate_max_high_res_bits()
        max_bits_per_frame = max_high_res_bits // self.header.num_frames
        max_pixels_per_frame = max_bits_per_frame // self.header.high_res_format.bits_per_pixel

        max_possible_side_length = math.floor(math.sqrt(max_pixels_per_frame))
        actual_side_length = (max_possible_side_length // self.header.high_res_format.bits_per_pixel) * self.header.high_res_format.bits_per_pixel
        return actual_side_length
        
    def __init__(self, images: list[Image.Image], format: ImageFormats, flags: TextureFlags, preserve_aspect_ratio: bool):
        VTFFile.__init__(self, high_res_format = format, num_frames = len(images), flags = flags)
        self.header.high_res_height = self.header.high_res_width = side_length = self.calculate_frame_side_length()
        for idx, image in enumerate(images):
            images[idx] = self.resize_image(image, side_length, preserve_aspect_ratio)
        self.high_res_data[0, :] = images
    def show(self) -> None:
        frame_list = self.high_res_data[0, :]
        show_gif(frame_list, frame_duration = self.SECONDS_PER_FRAME * 1000)
        


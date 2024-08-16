from os.path import join
#TODO: Implement more robust VMT framework for future improvements, for now hardcode strings
#Writes VMTs for spray
def write_vmt_files(file_name: str, tf2_directory: str) -> None:
    logo_dir = join(tf2_directory, "tf", "materials", "vgui", "logos")
    logo_vmt_file = join(logo_dir, f"{file_name}.vmt")
    with open(logo_vmt_file, "wt") as logo_vmt:
        logo_vmt.write(""""UnlitGeneric"
        {{
            "$basetexture"	"vgui/logos/{file_name}"
            "$translucent" "1"
            "$ignorez" "1"
            "$vertexcolor" "1"
            "$vertexalpha" "1"
        }}""".format(file_name = file_name))
    ui_vmt_file = join(logo_dir, "ui", f"{file_name}.vmt")
    with open(ui_vmt_file, "wt") as ui_vmt:
        ui_vmt.write(""""UnlitGeneric"
        {{
            // Original shader: BaseTimesVertexColorAlphaBlendNoOverbright
            "$translucent" 1
            "$basetexture" "VGUI/logos/{file_name}"
            "$vertexcolor" 1
            "$vertexalpha" 1
            "$no_fullbright" 1
            "$ignorez" 1
        }}""".format(file_name = file_name))
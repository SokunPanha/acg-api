import os
from dataclasses import dataclass


def _hex_to_ass(hex_color: str) -> str:
    """#RRGGBB → ASS &HAABBGGRR (opaque). Falls back to white on bad input."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


@dataclass
class PipelineConfig:
    work_dir: str
    session_id: str
    voice_ref: str = ""
    background: str = ""
    output_path: str = ""
    control_instruction: str = "Gentle and calm voice tone"
    cfg_value: float = 2.0
    denoise: bool = False
    normalize: bool = False
    include_subtitles: bool = True
    resolution: str = "16:9"   # "16:9" (landscape) or "9:16" (portrait)
    # subtitle styling
    subtitle_font_size: int = 24
    subtitle_color: str = "#FFFFFF"
    subtitle_position: str = "bottom"   # "bottom" | "middle" | "top"
    subtitle_outline: int = 2
    subtitle_font: str = ""             # libass FontName (matched from fonts_dir); "" = default
    fonts_dir: str = ""                 # dir libass scans for the chosen font file
    source: str = "single"              # storage bucket: "single" | "bulk"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (1080, 1920) if self.resolution == "9:16" else (1920, 1080)

    @property
    def subtitle_force_style(self) -> str:
        align = {"bottom": 2, "middle": 5, "top": 8}.get(self.subtitle_position, 2)
        font = f"FontName={self.subtitle_font}," if self.subtitle_font else ""
        return (
            f"{font}"
            f"FontSize={self.subtitle_font_size},"
            f"PrimaryColour={_hex_to_ass(self.subtitle_color)},"
            f"OutlineColour=&H00000000,"
            f"BorderStyle=1,Outline={self.subtitle_outline},Shadow=0,"
            f"Alignment={align},MarginV=40"
        )

    @property
    def assets_dir(self) -> str:
        return os.path.join(self.work_dir, "assets")

    @property
    def session_dir(self) -> str:
        return os.path.join(self.work_dir, "output", "sessions", self.source, self.session_id)

    @property
    def mp3_dir(self) -> str:
        return os.path.join(self.session_dir, "mp3")

    @property
    def srt_path(self) -> str:
        return os.path.join(self.session_dir, "subtitles.srt")

    @property
    def merged_audio(self) -> str:
        return os.path.join(self.session_dir, "merged.mp3")

    @property
    def concat_file(self) -> str:
        return os.path.join(self.session_dir, "concat.txt")

    @property
    def durations_path(self) -> str:
        return os.path.join(self.session_dir, "durations.json")

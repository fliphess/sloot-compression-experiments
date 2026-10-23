"""Paper capacity calculator: how many bytes fit on one page?

A module is one black/white cell (1 bit). Its size is limited by two things:
  printer  a module must be a whole number of printer dots, and lasers smear
           single dots, so we want at least 2-3 dots per module
  capture  the scanner/camera must see each module with >= ~3 pixels, or blur
           and misalignment make neighbouring modules bleed into each other

Usage: python3 lab/paper.py
"""
import math

MM_PER_INCH = 25.4
A4 = (210.0, 297.0)
MARGIN_MM = 10.0
STRUCTURE_OVERHEAD = 0.06  # finder patterns, sync tracks, page header
ECC_OVERHEAD = 0.25  # Reed-Solomon parity share; tunable after real print tests

# name, printer dpi, capture pixels per mm on the page, min dots per module, min pixels per module
PROFILES = [
    ("scan 600dpi, laser 600dpi", 600, 600 / MM_PER_INCH, 3, 3.0),
    ("scan 1200dpi, laser 1200dpi", 1200, 1200 / MM_PER_INCH, 4, 3.5),
    ("phone 12MP, whole page", 600, 4032 / A4[1], 3, 3.5),
    ("phone 48MP, whole page", 600, 8064 / A4[1], 3, 4.0),
    ("phone 12MP, 4 tiles/page", 600, 2 * 4032 / A4[1], 3, 3.5),
]

# bits per byte of source text after compression, from lab/baselines.py and lab/repair.py
TEXT_BPC = [("raw", 8.0), ("gzip", 2.85), ("xz", 2.33), ("v0 repair", 3.16), ("cmix (lit.)", 1.2)]


def capacity(printer_dpi, capture_px_per_mm, min_dots, min_px):
    dot_mm = MM_PER_INCH / printer_dpi
    dots = max(min_dots, math.ceil(min_px / capture_px_per_mm / dot_mm))
    module_mm = dots * dot_mm
    w, h = (side - 2 * MARGIN_MM for side in A4)
    modules = int(w / module_mm) * int(h / module_mm)
    net_bytes = modules * (1 - STRUCTURE_OVERHEAD) * (1 - ECC_OVERHEAD) / 8
    return module_mm, modules, net_bytes


def main():
    print(f"A4, {MARGIN_MM:.0f} mm margins, {STRUCTURE_OVERHEAD:.0%} structure, {ECC_OVERHEAD:.0%} ECC\n")
    header = f"{'profile':<30}{'module':>9}{'modules':>11}{'net/page':>11}"
    header += "".join(f"{name:>13}" for name, _ in TEXT_BPC)
    print(header)
    print(f"{'':<61}" + "".join(f"{'chars':>13}" for _ in TEXT_BPC))
    for name, dpi, px_mm, min_dots, min_px in PROFILES:
        module_mm, modules, net = capacity(dpi, px_mm, min_dots, min_px)
        row = f"{name:<30}{module_mm:>7.3f}mm{modules:>11,}{net / 1000:>9.0f}KB"
        row += "".join(f"{net * 8 / bpc / 1e6:>12.2f}M" for _, bpc in TEXT_BPC)
        print(row)
    print("\n'chars' = characters of English text per page at that compression")
    print("Reference: a QR code v40 holds 2,953 bytes; a novel is ~0.5M chars.")


if __name__ == "__main__":
    main()

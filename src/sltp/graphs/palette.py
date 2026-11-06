# -------- palette.py --------
# Chart colours: the first eight stay distinct next to each other and on white.
#   from sltp.graphs.palette import color_at

# -------- Palette --------
PALETTE: tuple[str, ...] = (
    "#4f46e5",  # indigo
    "#e11d48",  # rose
    "#0d9488",  # teal
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0284c7",  # sky
    "#65a30d",  # lime
    "#db2777",  # pink
)


# The colour at a position, cycling once the palette runs out.
def color_at(index: int) -> str:
    return PALETTE[index % len(PALETTE)]

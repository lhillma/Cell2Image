import sys
import matplotlib.colors as mcolors


def check_color_available(color: str) -> None:
    """
    Check if the color is available in matplotlib CSS4 colors and exit if it is not.
    """
    if color in mcolors.CSS4_COLORS:
        return

    print(f"Color {color} is not available in matplotlib CSS4 colors.")
    print("Available colors are:")
    print(list(mcolors.CSS4_COLORS.keys()))
    print()
    print(
        "Check the list of available colors at: https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors"
    )
    sys.exit(1)


from tfl_detection import test_find_tfl_lights2


if __name__ == "__main__":
    test_find_tfl_lights2("image/aachen_000012_000019_leftImg8bit.png", 
    "json/aachen_000012_000019_gtFine_polygons.json", 1)

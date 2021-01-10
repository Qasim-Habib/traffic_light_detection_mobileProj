import json

def get_resource( index ):
    with open("resources.json") as f:
        data = json.load(f)
        if 0 <= index < len( data ):
            return data[0]
    return load_default_resource()

def load_default_resource():
    return {
        "img": "image/aachen_000001_000019_leftImg8bit.png",
        "json": "json/aachen_000001_000019_gtFine_polygons.json"
    }
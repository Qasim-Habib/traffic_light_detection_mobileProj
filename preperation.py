

from part01 import tfl_detection as det
from part02.my_solution import part2 as attention
import json



def load_pls( pls_file_name ):
    with open(pls_file_name, 'r') as f:
        js = json.loads( f.read() )
        return js


def prepare_data():
    from part02.my_solution.part2 import crop_for_part04
    pkl_instance = load_pls( 'data.pls' )
    for img_path in pkl_instance["frames_pathes"]:
        pass
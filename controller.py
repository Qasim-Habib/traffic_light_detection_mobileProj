from tfl_manager import TFLManager
import json
import pickle
import numpy as np
from termcolor import colored



def load_pls( pls_file_name ):
    with open(pls_file_name, 'r') as f:
        js = json.loads( f.read() )
        return js

def get_EM( data, prev_frame_id, curr_frame_id ):
    EM = np.eye(4)
    for i in range(prev_frame_id, curr_frame_id):
        EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
    return EM



def adapt_frames_data( pkl_instance ):
    prev_frame_id_list = [] 
    curr_frame_id_list = []

    for i in range( len( pkl_instance["frames_id"] ) - 1 ):
        prev_frame_id_list.append( pkl_instance["frames_id"][ i ] )
        curr_frame_id_list.append( pkl_instance["frames_id"][ i + 1 ] )

    if len( prev_frame_id_list ) != len( curr_frame_id_list ):
        return None

    frames_data = []
    for i in range( len( prev_frame_id_list ) ):
        frames_data.append( ( prev_frame_id_list[i], curr_frame_id_list[i] ) )
    return frames_data


def get_pls_data( pls_file_name):
    pkl_instance = load_pls( pls_file_name )
    adapted_data = adapt_frames_data( pkl_instance )

    result = {}
    with open( pkl_instance["PKL"] , 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
        result["PP"] = data['principle_point']
        result["Focal Length"] =  data['flx']
        result["frames_id"] = pkl_instance["frames_id"]
        result["frames_pathes"] = pkl_instance["frames_pathes"]
        result["EMs"] = []
        for pairs_of_frames in adapted_data:
            EM = get_EM( data, int( pairs_of_frames[0] ), int( pairs_of_frames[1] ) )
            result["EMs"].append( [ pairs_of_frames, EM ] )

    return result


def run( tfl_manager, pls_data ):
    for i in range( len( pls_data["frames_pathes"] ) - 1 ):
        prev_frame = ( pls_data["frames_pathes"][ i ], pls_data["frames_id"][ i ] )
        curr_frame = ( pls_data["frames_pathes"][ i + 1 ], pls_data["frames_id"][ i + 1 ] )
        print( colored( "Prev Frame: ( " + prev_frame[0] + ", " + str( prev_frame[1] ) + " )", 'red' ) )
        print( colored( "Current Frame: ( " + curr_frame[0] + ", " + str( curr_frame[1] ) + " )", 'green' ) )
        EM = pls_data["EMs"][i][1]
        tfl_manager.run( EM, prev_frame )
    


# def make_bin_file( pls_file_name ):
#     from part02.my_solution.part2 import crop_for_part04
#     pkl_instance = load_pls( pls_file_name )
#     crop_for_part04( pkl_instance["frames_pathes"] )


if __name__ == "__main__":
    # make_bin_file( "data.pls" )
    pls_data = get_pls_data( "data.pls" ) 
    tfl_manager = TFLManager( PP = pls_data["PP"], FL = pls_data["Focal Length"] )
    run( tfl_manager, pls_data )

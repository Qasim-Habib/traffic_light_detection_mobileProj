from part02.my_solution import part2 as attention
from part01 import tfl_detection as det
from nn import get_prediction
from part03.SFM_standAlone import FrameContainer
import part03.SFM as sfm
from visualizaton import visualize
import numpy as np



class TFLManager:

    def __init__(self, PP = None, FL = None, pls_file_name = "data.pls"):
        self.PP = PP
        self.FL = FL
        self.pls_file_name = pls_file_name
        self.frames = { }
        self.prev = None


    def make_prediction( self, frame_path, frame_id ):
        src_image = attention.load_image( frame_path )
        # integration with part 01 ( Dettection )
        points_from_detect = det.tfl_detect( src_image )
        # integration with part 02 ( Attention )
        attention.crop_for_part04( str( frame_id ), src_image, points_from_detect )
        prediction = get_prediction(frame_id)
        tfls = []
        for i, pred in enumerate( prediction ):
            if pred == 1:
                tfls.append( list( points_from_detect[i] ) )
        # self.frames[ frame_id ] = ( Frame( frame_id, frame_path, tfls ) )
        fc = FrameContainer(frame_path)
        fc.traffic_light = np.array( tfls )
        self.frames[ frame_id ] = fc
        

    
    # def make_predictions(self, frame_path, frame_id ):
    #     if frame_id not in self.frames.keys():
    #         self.make_prediction( frame_path, frame_id )


    # prev_frame => ( prev_path, prev_id )
    # curr_frame => ( curr_path, curr_id ) ]
    # def run(self, EM, prev_frame, curr_frame ):
    def run(self, EM, curr_frame ):
        # integration with part 01 and part 02( Dettection and Attention)
        self.make_prediction( curr_frame[0], curr_frame[1] )
        if self.prev is None:
            self.prev = curr_frame
            return
        prev_fcont = self.frames[ self.prev[1] ]
        curr_fcont = self.frames[ curr_frame[1] ]
        curr_fcont.EM = EM
        sfm.calc_TFL_dist( prev_fcont, curr_fcont, self.FL, self.PP )
        visualize( self.prev[1], curr_frame[1], prev_fcont, curr_fcont, self.FL, self.PP )
        self.prev = curr_frame

        
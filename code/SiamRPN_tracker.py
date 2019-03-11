from .net import SiamRPNvot#, SiamRPNBIG
from .run_SiamRPN import SiamRPN_init, SiamRPN_track
from .utils import get_axis_aligned_bbox, cxy_wh_2_rect

from os.path import realpath, dirname, join
import glob
import torch
import numpy as np

"""
This is a very light wrapper around the other functionality which encapsulates the state
"""
class SiamRPN_tracker(object):
    def __init__(self, image, ltwh_bbox, lost_conf=0.85, found_conf=0.95, expand_rate=20):
        """
        params
        ---------- 
        image : np.ndarray
            This is the first frame in the tracking sequence 
        ltwh_bbox : ArrayLike
            this is the location of the box in the first frame, given as [left, top, width, height]
        """
        # constants
        self.lost_conf = lost_conf
        self.found_conf = found_conf
        self.expand_rate = expand_rate
        self.padding = 0


        # initialize the network
        self.net = SiamRPNvot() # TODO determine the difference between *vot and *big
        self.net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
        self.net.eval().cuda()


        # manipulate the bounding box into the useful form
        [cx, cy, w, h] = [ltwh_bbox[0] + ltwh_bbox[2] / 2 , ltwh_bbox[1] + ltwh_bbox[3] / 2, ltwh_bbox[2], ltwh_bbox[3]]
        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

        # Initialize the actual tracker, which is now stored in the state dict
        self.state = SiamRPN_init(image, target_pos, target_sz, self.net)

    def predict(self, image):
        """
        image : np.ndarray
            The frame which you wish to search for the image
        """
        #update the state
        #p = state['p']
        #net = state['net']
        #avg_chans = state['avg_chans']
        #window = state['window']
        #target_pos = state['target_pos']
        #target_sz = state['target_sz']

        #wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        #hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        #s_z = np.sqrt(wc_z * hc_z)
        #scale_z = p.exemplar_size / s_z
        #d_search = (p.instance_size - p.exemplar_size) / 2
        #pad = d_search / scale_z

        new_state, crop_region = SiamRPN_track(self.state, image, self.padding)
        score = new_state["score"]
        if score < self.lost_conf:
            self.padding += self.expand_rate 
        elif score > self.found_conf:
            self.padding = 0

        ltwh = cxy_wh_2_rect(new_state['target_pos'], new_state['target_sz'])
        ltwh = [int(x) for x in ltwh]
        self.state = new_state

        return ltwh, score, crop_region

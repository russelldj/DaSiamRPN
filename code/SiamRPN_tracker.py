from .net import SiamRPNvot#, SiamRPNBIG
from .run_SiamRPN import SiamRPN_init, SiamRPN_track
from .utils import get_axis_aligned_bbox, cxy_wh_2_rect

from os.path import realpath, dirname, join
import glob
import torch
import numpy as np
import pdb
import copy

"""
This is a very light wrapper around the other functionality which encapsulates the state
"""
class SiamRPN_tracker(object):
    def __init__(self, image, ltwh_bbox, lost_conf=0.80, found_conf=0.95, expand_rate=0):
        """
        params
        ---------- 
        image : np.ndarray
            This is the first frame in the tracking sequence 
        ltwh_bbox : ArrayLike
            this is the location of the box in the first frame, given as [left, top, width, height]
        lost_conf : float
            This is taken from page 10 of the DaSiamRPN paper
        found_conf : float
            This is also taken from page 10 of the DaSiamRPN paper
        expand_rate : int
            the number of pixels to grow when lost 
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
        self.conf = 1.0 # completely sure when we start

    def getConf(self):
        return self.state["conf"]

    def getLostConf(self):
        return self.lost_conf

    def getLocation(self):
        location = self.state["target_pos"]
        return location

    def isLost(self):
        return self.conf < self.lost_conf

    def setSearchRegion(self, search_region):
        """
        search_region : ArrayLike[int]
            The [l, t, w, h] bounding box to search
        """
        # you have to set the self.state["target_pos"] = array(center)
        # and self.state["target_size"] = array([x, y])
    
        centered = self.xywh_to_centered(search_region)
        print("setting the search region to {}".format(centered))
        self.state["target_pos"] = np.asarray(centered[0:2])
        self.state["target_sz"] = np.asarray(centered[2:])

    def setSearchLocation(self, search_location):
        """
        search_region : ArrayLike[int]
            The x, y location to search around (centered)
        """
        assert len(search_location) == 2
        self.state["target_pos"] = np.asarray(search_location)

    def xywh_to_centered(self, xywh):
        """
        xywh : ArrayLike[Number]
        """
        centered = copy.copy(xywh)
        centered[0] += xywh[2] // 2 # integer division in case of int
        centered[1] += xywh[3] // 2 # integer division in case of int
        return centered

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
        p = new_state['p']
        print(p.exemplar_size)
        print(p.context_amount)
        print(p.instance_size)
        score = new_state["score"]
        if score < self.lost_conf:
            self.padding = self.expand_rate# this represents that the paper states that it expands in one timestep to the max size
        elif score > self.found_conf:
            self.padding = 0

        ltwh = cxy_wh_2_rect(new_state['target_pos'], new_state['target_sz'])
        ltwh = [int(x) for x in ltwh]

        self.state = new_state
        self.conf = score

        return ltwh, score, crop_region

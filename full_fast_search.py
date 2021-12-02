import numpy as np
import cv2
import sys

""" Base For Search Algorithm """
class BaseSearch(object):
    def __init__(self, refer_frame, curr_frame, block_mode, p):
        self.refer_frame = refer_frame
        self.curr_frame = curr_frame
        self.block_h ,self.block_w = block_mode
        self.p = p

        # Coordinates of marcoblock are in focus
        self.x = 0  
        self.y = 0

    def estimate_motion(self):
        h, w = self.curr_frame.shape
        h_blocks, w_blocks =  h // self.block_h, w // self.block_w

        residual_frame = np.zeros((h, w), dtype = self.curr_frame.dtype)
        match_points = []

        for y in range(0, int(h_blocks*self.block_h), self.block_h):
            row = []
            for x in range(0, int(w_blocks*self.block_w), self.block_w):
                self.x = x
                self.y = y
                
                match_point, residual_block = self.find_match_block()
                row.append(match_point)
                residual_frame[y:y+self.block_h, x:x+self.block_w] = residual_block
            match_points.append(row)

        return np.asarray(match_points), residual_frame
    
    def get_refer_block(self, rx, ry):
        h, w = self.refer_frame.shape
        rx, ry = max(0, rx), max(0, ry)
        rx, ry = min(w-self.block_w, rx), min(h-self.block_h, ry)
        refer_block = self.refer_frame[
            ry : ry + self.block_h, 
            rx : rx + self.block_w
        ]
        return refer_block, rx, ry

    def find_match_block(self):
        raise NotImplementedError
    
    def get_MAD(self, curr_block, refer_block):
        return np.sum(np.abs(np.subtract(curr_block, refer_block)))/(self.block_w*self.block_h)


""" Full Search """
class FullSearch(BaseSearch):
    def find_match_block(self):
        curr_block = self.curr_frame[
            self.y : self.y + self.block_h,
            self.x : self.x + self.block_w,
        ]
        searching_range = range(-self.p, self.p+1)

        minMAD = sys.float_info.max
        match_point = None
        match_block = None

        for m in searching_range:
            for n in searching_range:
                rx, ry = self.x+n, self.y+m
                refer_block, rx, ry = self.get_refer_block(rx, ry)

                MAD = self.get_MAD(curr_block, refer_block)
                if MAD<minMAD:
                    minMAD=MAD
                    match_point = (rx, ry)
                    match_block = refer_block
        
        residual_block = curr_block - match_block
        return match_point, residual_block

""" Three Step Search """
class ThreeStepSearch(BaseSearch):
    def find_match_block(self):
        curr_block = self.curr_frame[
            self.y : self.y + self.block_h,
            self.x : self.x + self.block_w,
        ]
        step = 4

        minMAD = sys.float_info.max
        match_point = None
        match_block = None

        # Center point will be updated each loop
        cx = self.x + self.block_w // 2
        cy = self.y + self.block_h // 2 
        while step > 0:
            p1 = (cx-step, cy-step)
            p2 = (cx-step, cy)
            p3 = (cx-step, cy+step)

            p4 = (cx, cy-step)
            p5 = (cx, cy)
            p6 = (cx, cy+step)

            p7 = (cx+step, cy-step)
            p8 = (cx+step, cy)
            p9 = (cx+step, cy+step)
            points = [p1,p2,p3,p4,p5,p6,p7,p8,p9]

            for i in range(len(points)):
                px, py = points[i]  # center point
                rx, ry = px-self.block_w//2, py-self.block_h//2 # topleft point
                refer_block, rx, ry = self.get_refer_block(rx, ry)

                MAD = self.get_MAD(curr_block, refer_block)
                if MAD < minMAD:
                    minMAD = MAD
                    match_block = refer_block
                    match_point = (rx, ry)
                    cx, cy = points[i]

            step = step//2
        
        residual_block = curr_block - match_block
        return match_point, residual_block

####################################################################################
def extract_frames(input_file, start_pos, num_frames):
    results = []
    cap = cv2.VideoCapture(input_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if start_pos >= 0 and start_pos + num_frames <= total_frames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        
    while True:
        ret, frame = cap.read()
        if ret == True and num_frames != 0:
            results.append(frame)
            num_frames -= 1
            start_pos += 1
        else:
            break
    cap.release()
    return results

def decode(refer_frame, match_points, residual_frame, block_mode):
    h, w = refer_frame.shape
    block_h, block_w = block_mode
    h_blocks, w_blocks =  h // block_h, w // block_w
    curr_frame = np.zeros((h, w), dtype = refer_frame.dtype)

    for y in range(0, int(h_blocks*block_h), block_h):
        for x in range(0, int(w_blocks*block_w), block_w):
            rx, ry = match_points[y//block_h, x//block_w]
            match_block = refer_frame[ry:ry+block_h, rx:rx+block_w]

            residual_block = residual_frame[y:y+block_h, x:x+block_w]
            curr_frame[y:y+block_h, x:x+block_w] = match_block + residual_block

    return curr_frame

def YCrCb2BGR(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def BGR2YCrCb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

def preprocess(refer, current):
    if isinstance(refer, np.ndarray) and isinstance(current, np.ndarray):
        refer_frame = BGR2YCrCb(refer)[:, :, 0] # get luma (Y) channel
        curr_frame = BGR2YCrCb(current)[:, :, 0] # get luma (Y) channel
    else:
        raise ValueError
    return (refer_frame, curr_frame)

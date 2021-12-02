import numpy as np
from numpy.core.numeric import zeros_like
import cv2

frame = cv2.imread("peppers.png",0)

size =4 #chia ra làm các block 4x4
height = frame.shape[0]
width = frame.shape[1]
tmp_height=height+2
tmp_width=width+5
tmp = np.zeros((tmp_height,tmp_width), dtype=frame.dtype)
for row in range(height):
    for col in range(width):
        tmp[row+1,col+1] = frame[row,col] #tạo tmp chứa frame với các padding là 0

def mode0(frame, mode = "vertical"):
    tmp_out = np.zeros_like(tmp)
    for row in range(1, height+1):
        for col in range(1, width+1):
            if(row%size==0):
                tmp_out[row,col] = tmp[row-size,col]
            else:
                tmp_out[row,col] = tmp[row-row%size,col]
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode1(frame, mode = "horizontal"):
    tmp_out = np.zeros_like(tmp)
    for col in range(1, width+1):
        for row in range(1, height+1):
            if(col%size==0):
                tmp_out[row,col] = tmp[row,col-size]
            else:
                tmp_out[row,col] = tmp[row,col-col%size]
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode2(frame, mode ="DC"):
    tmp_out = np.zeros_like(tmp)
    for row in np.arange(1,stop=height-size+2,step=size):
        for col in np.arange(1,stop=width-size+2,step=size):
            x = np.full((4,4), (sum(tmp[row-1,col:col+size])+sum(tmp[row:row+4,col-1]))/8)
            tmp_out[row:row+size,col:col+size] = x
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode3(frame, mode = "down left"):
    tmp_out = np.zeros_like(tmp)
    for row in range(1,height+1):
        for col in range(1, width+1):
            if(row%4==0):
                tmp_out[row,col] = tmp[row-4,col+4]
            else:
                tmp_out[row,col] = tmp[row-row%4,col+row%4]
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode4(frame, mode = "down right"):
    tmp_out = np.zeros_like(tmp)
    for row in range(1,height+1):
        for col in range(1,width+1):
            if(row%4==1 or col%4==1):
                tmp_out[row,col]=tmp[row-1,col-1]
            else:
                tmp_out[row,col]=tmp_out[row-1,col-1]
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode5(frame, mode = "vertical right"):
    tmp_out = np.zeros_like(tmp)
    for row in np.arange(1,stop=height+1-size+1,step=size):
        for col in range(1,width+1):
            if(col % size == 1):
                    tmp_out[row:row+2,col] = np.full((2),tmp[row-1,col-1])
                    tmp_out[row+2:row+size,col] = np.full((2),tmp[row+1,col-1])
            else:
                tmp_out[row:row+2,col] = np.full((2),tmp[row-1,col-1])
                tmp_out[row+2:row+size,col] = np.full((2),tmp_out[row+1,col-1])
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode6(frame, mode ="horizontal down"):
    tmp_out = np.zeros_like(tmp)
    for col in np.arange(1,stop=width+1-size+1,step=size):
            for row in range(1,height+1):
                if(row % size == 1):
                    tmp_out[row,col:col+2] = np.full((2),tmp[row-1,col-1])
                    tmp_out[row,col+2:col+size] = np.full((2),tmp[row-1,col+1])
                else:
                    tmp_out[row,col:col+2] = np.full((2),tmp[row-1,col-1])
                    tmp_out[row,col+2:col+size] = np.full((2),tmp_out[row-1,col+1])
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode7(frame, mode = "vertical left"):
    tmp_out = np.zeros_like(tmp)
    for row in np.arange(1,stop=height+1-size+1,step=size):
        for col in range(1,width+1):
            tmp_out[row:row+2,col] = np.full((2),tmp[row-1,col+1])
            tmp_out[row+2:row+size,col] = np.full((2), tmp[row-1,col+2])
    output = tmp_out[1:height+1,1:width+1]
    return output

def mode8(fram, mode = "horizontal up"):
    tmp_out = np.zeros_like(tmp)
    for col in np.arange(1,stop=width+1-size+1,step=size):
            for row in range(1,height+1):
                tmp_out[row,col:col+2] = np.full((2),tmp[row,col-1])
                tmp_out[row,col+2:col+size] = np.full((2),tmp[row+1,col-1])
    output = tmp_out[1:height+1,1:width+1]
    return output

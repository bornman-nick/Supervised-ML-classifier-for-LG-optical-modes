# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:26 2018

@author: Nick
"""


import numpy as np
import cv2
from tkinter import filedialog
import tkinter

root = tkinter.Tk()

video_file_path =  filedialog.askopenfilename(initialdir = "C:\\", title = "Select file", filetypes = (("All","*.*"),("JPEG files","*.jpg"),("AVI files","*.avi")))
video_frames_path = filedialog.askdirectory(initialdir = "C:\\", title = "Select folder")
root.destroy()

capture = cv2.VideoCapture(video_file_path)

frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

print(frame_count)

for i in range(frame_count):
    ret, frame = capture.read()
    cv2.imwrite(video_frames_path + "/l=1,p=0,frame_" + str(i) + ".png", frame)
    if i % 1000 == 0:
        print(str(i) + " frames processed")

capture.release()
cv2.destroyAllWindows()


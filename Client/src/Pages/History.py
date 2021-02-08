import os
import time
import tkinter as tk
import tkinter.font as tkFont
from threading import Thread

import cv2
import PIL.Image
import src.Components.Style as Style
import src.Pages.Setting as Setting
from src.utils.utils import *


class App:
    def __init__(self, window):
        self.window = window
        self.imd_id = 0
        self.obj = []
        self.obj_img = []
        self.interrupt_process = False

    # run program
    def run(self):
        self.massage_1 = tk.Label(
            self.window, text='All Images in {}'.format(Setting.output_path), bg='#19191A', fg="white", font=Style.font(10))
        self.obj.append(self.massage_1)
        self.obj[len(self.obj)-1].place(x=10, y=53)
        sign = [['public/sign/double left.png', self.doubleLeft], ['public/sign/left.png', self.left], ['public/sign/stop.png',
                self.stopProcess], ['public/sign/right.png', self.right], ['public/sign/double right.png', self.doubleRight]]
        shift = 0
        for i in range(len(sign)):
            self.showImgBott(pic=sign[i][0], x=860+shift, y=1030,
                             size=30, color='#19191A', comm=sign[i][1])
            shift += 40
        self.interrupt_process = False
        self.showHistory()

    # หยุดการทำงานและทำลาย
    def DestroyComp(self):
        for obj in self.obj:
            obj.after(0, obj.destroy)
        self.obj = []
        self.imd_id = 0

    def stopProcess(self):
        self.interrupt_process = True

    # แสดงผลก่อนหน้า
    def left(self):
        self.interrupt_process = False
        if self.imd_id - 98*2 >= 0:
            self.imd_id -= 98*2
        else:
            self.imd_id = 0
        self.showHistory()

    # to first page
    def doubleLeft(self):
        self.interrupt_process = False
        self.imd_id = 0
        self.showHistory()

    # แสดงผลอีก
    def right(self):
        self.interrupt_process = False
        self.showHistory()

    # continue until end 
    def doubleRight(self):
        self.interrupt_process = False
        Thread(target=self.threadDoubleRight).start()

    def threadDoubleRight(self):
        self.interrupt_process = False
        t0 = time.time()
        t1 = torch_utils.time_synchronized()
        t2 = torch_utils.time_synchronized()
        while self.imd_id < len(self.filename) and not self.interrupt_process:
            if t2-t1 > 2:
                for obj in self.obj_img:
                    obj.after(2100, obj.destroy)
                t1 = torch_utils.time_synchronized()
                self.showHistory()
            t2 = torch_utils.time_synchronized()

    # แสดงประวัติ
    def showHistory(self):
        folders = Setting.output_path
        self.filename = []
        for dirpath, dirnames, files in os.walk(folders):
            for file_name in files:
                self.filename.append(dirpath+'/'+file_name)
        self.shift_y = 0
        for i in range(7):
            self.shift_x = 0
            for j in range(14):
                if self.imd_id < len(self.filename) and len(self.filename) is not 0:
                    self.showImg(
                        self.filename[self.imd_id], x=10+self.shift_x, y=80+self.shift_y, size=128)
                    self.imd_id += 1
                else:
                    self.showImg('public/Wallpaper/blank-profile.png',
                                 x=10+self.shift_x, y=80+self.shift_y, size=128)
                self.shift_x += 135
            self.shift_y += 135

    '''โหลดรูปและแสดง'''
    def showImg(self, pic, x, y, size):
        self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.resize(cv2.cvtColor(
                cv2.imread(pic), cv2.COLOR_BGR2RGB), (size, size))))
        self.img = tk.Label(self.window, image=self.photo, borderwidth=0)
        self.obj_img.append(self.img)
        self.obj_img[len(self.obj_img)-1].image = self.photo
        self.obj_img[len(self.obj_img)-1].place(x=x, y=y)

    def showImgBott(self, pic, x, y, size, color, comm):
        self.load = PIL.Image.open(pic)  
        self.load.thumbnail((size, size)) 
        self.render = PIL.ImageTk.PhotoImage(self.load)
        self.img = tk.Button(self.window, image=self.render,
                             bg=color, borderwidth=5, command=comm) 
        self.obj.append(self.img)
        self.obj[len(self.obj)-1].image = self.render
        self.obj[len(self.obj)-1].place(x=x, y=y)

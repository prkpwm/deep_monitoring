import os
import tkinter as tk
import tkinter.font as tkFont
from threading import Thread

import PIL.Image
import src.Components.Live as Live
import src.Components.Style as Style
import src.Pages.Face_Prediction as Face_Prediction
import src.Pages.History as History
import src.Pages.People_Tracking as People_Tracking
import src.Pages.Setting as Setting
import src.Pages.Statistics as Statistics

screen_width = 1920
screen_height = 1080


class App:

    def __init__(self, window):
        self.window = window

        ''' link menu '''
        self.faceDetection = Live.App(self.window, "detect")
        self.heat_map = Live.App(self.window, "heatmap")
        self.people_counting = Live.App(self.window, "peoplecounting")
        self.his = History.App(self.window)
        self.stat = Statistics.App(self.window)
        self.face_predict = Face_Prediction.App(self.window)
        self.Setting = Setting.App(self.window)
        self.People_Tracking = People_Tracking.App(self.window)

        self.window.after(0, lambda: self.showImgWithDelay(
            'public/Wallpaper/singha.jpg', x=0, y=0, size=1920, color='#000000'))  # เล่นโลโก้ หลักจาก 0 วินาที
        # แสดงเมนู หลักจาก 3 วินาที
        self.window.after(3000, lambda: self.run())
        # แสดง face detect หลักจาก 3 วินาที
        self.window.after(3000, lambda: self.faceDetection.init_preprocess())

    def setSize(self):
        global screen_width
        global screen_height
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

    def run(self):

        # กำหนด body background
        self.backg = tk.Label(self.window, width=1920,
                              height=1080, bg='#19191A')
        self.backg.place(x=0, y=0)

        # กำหนด header background
        self.header = tk.Label(self.window, width=1080, height=3, bg='#000000')
        self.header.place(x=0, y=0)

        self.header = tk.Label(self.window, text='Version: 1', bg='#000000',
                               fg="white", font=Style.font(18))  # Version Program

        self.header.place(x=1680, y=15)
        if os.name == 'nt':
            width_os = 14
            height_os = 2
        else:
            width_os = 15
            height_os = 2
        self.tk_Button = []
        self.tk_Button.append(tk.Button(self.window, text="Face Detection", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.Live))
        self.tk_Button.append(tk.Button(self.window, text="People Counting", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.peopleCounting))
        self.tk_Button.append(tk.Button(self.window, text="Heat Map", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.heatMap))
        self.tk_Button.append(tk.Button(self.window, text="Face Analytics", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.facePrediction))
        self.tk_Button.append(tk.Button(self.window, text="People Tracking", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.peopleTracking))
        self.tk_Button.append(tk.Button(self.window, text="History", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.History))
        self.tk_Button.append(tk.Button(self.window, text="Dashboard", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.Statistics))
        self.tk_Button.append(tk.Button(self.window, text="Setting", width=width_os,  height=height_os,
                                        borderwidth=0, highlightthickness=0, bg='#19191A', fg="white", font=Style.font(10), command=self.setting))

        padding = 0
        for obj in self.tk_Button:
            obj.place(x=10+padding, y=15)
            padding += 120

        self.showImg(pic='public/Logos/logo.png', x=1620,
                     y=1, size=50, color='#000000')  # Logo
        '''
        self.showImgWithComm(pic='public/sign/del.png', x=1830, y=10,
                             size=30, color='#000000', comm=self.window.iconify)  # พับหน้าจอ
        '''
        self.showImgWithComm(pic='public/sign/close.png', x=1880, y=10,
                             size=30, color='#000000', comm=self.window.destroy)  # ปิดหน้าจอ

    '''link และ destroy'''

    def Live(self):
        self.showImg('public/Wallpaper/fill.png', x=10, y=53,
                     size=1920, color='#19191A')  # วาดพื้นหลัง
        # เล่นเสียงคลิก
        '''destroy'''
        self.stat.DestroyComp()
        self.his.DestroyComp()
        self.people_counting.DestroyComp()
        self.heat_map.DestroyComp()
        self.face_predict.DestroyComp()
        self.People_Tracking.DestroyComp()
        self.faceDetection.init_preprocess()  # link ไปยัง faceDetection

    def History(self):
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=53, size=1920, color='#19191A')
        self.People_Tracking.DestroyComp()
        self.stat.DestroyComp()
        self.faceDetection.DestroyComp()
        self.face_predict.DestroyComp()
        self.people_counting.DestroyComp()
        self.heat_map.DestroyComp()
        self.his.run()

    def Statistics(self):
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=53, size=1920, color='#19191A')
        self.People_Tracking.DestroyComp()
        self.his.DestroyComp()
        self.faceDetection.DestroyComp()
        self.face_predict.DestroyComp()
        self.people_counting.DestroyComp()
        self.heat_map.DestroyComp()
        self.stat.run()

    def facePrediction(self):
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=53, size=1920, color='#19191A')
        self.People_Tracking.DestroyComp()
        self.his.DestroyComp()
        self.faceDetection.DestroyComp()
        self.stat.DestroyComp()
        self.people_counting.DestroyComp()
        self.heat_map.DestroyComp()
        self.face_predict.run()

    def heatMap(self):
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=53, size=1920, color='#19191A')
        self.People_Tracking.DestroyComp()
        self.his.DestroyComp()
        self.faceDetection.DestroyComp()
        self.stat.DestroyComp()
        self.face_predict.DestroyComp()
        self.people_counting.DestroyComp()
        self.heat_map.init_preprocess()

    def peopleCounting(self):
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=53, size=1920, color='#19191A')
        self.People_Tracking.DestroyComp()
        self.his.DestroyComp()
        self.faceDetection.DestroyComp()
        self.stat.DestroyComp()
        self.face_predict.DestroyComp()
        self.people_counting.init_preprocess()
        self.heat_map.DestroyComp()

    def peopleTracking(self):
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=53, size=1920, color='#19191A')
        self.his.DestroyComp()
        self.faceDetection.DestroyComp()
        self.stat.DestroyComp()
        self.face_predict.DestroyComp()
        self.people_counting.DestroyComp()
        self.People_Tracking.run()
        self.heat_map.DestroyComp()

    def setting(self):
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=53, size=1920, color='#19191A')
        self.People_Tracking.DestroyComp()
        self.his.DestroyComp()
        self.faceDetection.DestroyComp()
        self.stat.DestroyComp()
        self.face_predict.DestroyComp()
        self.people_counting.DestroyComp()
        self.heat_map.DestroyComp()
        self.Setting.run()

    '''โหลดรูป'''

    def showImg(self, pic, x, y, size, color):
        self.load = PIL.Image.open(pic)  # โหลด
        self.load.thumbnail((size, size))  # ปรับขนาด
        self.render = PIL.ImageTk.PhotoImage(
            self.load)  # เปลี่ยนเป็น PhotoImage
        self.img = tk.Label(self.window, image=self.render,
                            bg=color, borderwidth=0)  # ตั้งค่ารูปภาพ
        self.img.image = self.render  # ตั้งค่ารูปภาพ
        self.img.place(x=x, y=y)  # กำหนดตำแหน่ง

    def showImgWithDelay(self, pic, x, y, size, color):
        self.load = PIL.Image.open(pic)  # โหลด
        self.load.thumbnail((size, size))  # ปรับขนาด
        self.render = PIL.ImageTk.PhotoImage(
            self.load)  # เปลี่ยนเป็น PhotoImage
        self.img = tk.Label(self.window, image=self.render,
                            bg=color, borderwidth=0)  # ตั้งค่ารูปภาพ
        self.img.image = self.render  # ตั้งค่ารูปภาพ
        self.img.place(x=x, y=y)  # กำหนดตำแหน่ง
        # แสดงรูปภาพ หลักจาก 3.7 วินาที ให้ทำลาย
        self.img.after(3700, self.img.destroy)

    def showImgWithComm(self, pic, x, y, size, color, comm):
        self.load = PIL.Image.open(pic)  # โหลด
        self.load.thumbnail((size, size))  # ปรับขนาด
        self.render = PIL.ImageTk.PhotoImage(
            self.load)  # เปลี่ยนเป็น PhotoImage
        self.img = tk.Button(self.window, image=self.render,
                             bg=color, borderwidth=0, command=comm, highlightthickness=0)  # ตั้งค่าปุ่ม
        self.img.image = self.render  # ตั้งค่ารูปภาพ
        self.img.place(x=x, y=y)  # กำหนดตำแหน่ง

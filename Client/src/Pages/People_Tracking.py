import glob
import os
import pickle
import time
import tkinter
import tkinter.font as tkFont
from datetime import date, timedelta
from threading import Thread
from tkinter import filedialog

import cv2
import dlib
import imutils
import numpy as np
import PIL.Image
import PIL.ImageTk
import src.Components.Style as Style
import src.Pages.Setting as Setting
import torch.backends.cudnn as cudnn
from imutils import face_utils
from PIL import Image, ImageFilter
from src.Components.Live import EntryWithPlaceholder
from src.Pages.Face_Detection import Detection
from src.utils import google_utils
from src.utils.datasets import *
from src.utils.utils import *
from torchvision import transforms

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './models/shape_predictor_68_face_landmarks.dat')
face_recognition = dlib.face_recognition_model_v1(
    './models/dlib_face_recognition_resnet_model_v1.dat')
res_list = []

class App:
    def __init__(self, window):
        self.window = window
        self.obj = []
        self.running = True
        self.filename = []

    #เริ่มโปรแกรม
    def run(self):
        self.massage_1 = tkinter.Label(
            self.window, text='Face Recognition', bg='#19191A', fg="white", font=Style.font(12))
        self.massage_1.place(x=110, y=83)
        self.button1 = tkinter.Button(
            self.window, text="Load Image", bg='#19191A', fg="white", width=30, command=self.loadfile)
        self.button1.place(x=130, y=130)
        self.running = True
    
    #รับรูป
    def loadfile(self):
        self.window.filename = filedialog.askopenfilenames(
            initialdir="/backup", title="Select Image", filetypes=(("jpg", "*.jpg"), ("all", "*.*")))
        for obj in self.obj:
            obj.destroy()
        if len(self.window.filename) > 0:
            self.massage_2 = tkinter.Label(
                self.window, text='Image', bg='#19191A', fg="white", font=Style.font(12))
            self.massage_2.place(x=110, y=200)
            self.shift_y = 0
            self.imd_id = 0
            loopcon = True
            for i in range(4):
                self.shift_x = 0
                for j in range(4):
                    if self.imd_id < len(self.window.filename):
                        self.showImg(
                            '{}'.format(self.window.filename[self.imd_id]), x=110+self.shift_x, y=240+self.shift_y, size=128)
                        self.imd_id += 1
                        self.shift_x += 135
                    else:
                        loopcon = False
                if not loopcon:
                    break
                self.shift_y += 135

            self.pos_y = 815 if len(self.window.filename) > 16 else self.shift_y+405 if len(
                self.window.filename) % 4 != 0 else self.shift_y+270
            self.filterRule()
    #กรอกการค้นหา
    def filterSetting(self):
        sex = 'male' if repr(str(self.var.get())) == repr(str(1)) else 'female'
        age = int(self.e1.get())
        date = self.e2.get()
        self.obj.append(tkinter.Button(
            self.window, text="Stop", bg='#19191A', fg="white", width=30, command=self.stopRunning))
        self.obj[len(self.obj)-1].place(x=130, y=self.pos_y+90)
        folders = 'prediction'
        filename = []
        for dirpath, dirnames, files in os.walk(folders):
            for file_name in files:
                filename.append(dirpath+"/"+file_name)
        information = []
        for file in filename:
            file_data = open(file, "r")
            data = file_data.readlines()
            for line in data:
                if line != "\n":
                    if line.endswith('\n'):
                        line = line[:-1]
                    information.append(line)
        infoList = []
        for i in range(len(information)):
            information[i] = information[i].replace('[', "")
            information[i] = information[i].replace(']', "")
            information[i] = information[i].replace('\'', "")
            information[i] = information[i].replace(' ', "")
            infoList.append(information[i].split(','))
        self.dateCheck(date)
        for info in infoList:
            if info[0].split(':')[1] == sex and abs(int(info[1].split(':')[1])-age) <= 20:
                dt = info[3].split('/')[-5:-2]
                str_dt = str(dt[0]+'/'+dt[1]+'/'+dt[2])
                if str_dt in self.date_arr:
                    inf = str(info[3])
                    inf = inf.replace(Setting.separated_path,Setting.predicted_path)
                    self.filename.append(inf)
    #ช่วงวัน
    def daterange(self, date1, date2):
        for n in range(int((date2 - date1).days)+1):
            yield date1 + timedelta(n)
    #เช็ควัน
    def dateCheck(self, Date):
        date_filter = Date.split('-')
        start_date = date_filter[0].split('/')
        end_date = date_filter[1].split('/')
        start_dt = date(int(start_date[2]), int(
            start_date[1]), int(start_date[0]))
        end_dt = date(int(end_date[2]), int(end_date[1]), int(end_date[0]))
        self.date_arr = []
        for dt in self.daterange(start_dt, end_dt):
            self.date_arr.append(dt.strftime("%Y/%m/%d"))
    #รับข้อมูลกรอง
    def filterRule(self):
        self.var = tkinter.IntVar()
        self.obj.append(tkinter.Label(
            self.window, text="Filter : ", bg='#19191A', fg='#ffffff'))
        self.obj[len(self.obj)-1].place(x=110, y=self.pos_y-20)
        self.obj.append(tkinter.Label(
            self.window, text="Sex", bg='#19191A', fg='#ffffff'))
        self.obj[len(self.obj)-1].place(x=130, y=self.pos_y)
        self.obj.append(tkinter.Radiobutton(self.window, text="Male", variable=self.var,
                                            value=1, bg='#19191A', selectcolor='#000000', fg='#ffffff'))
        self.obj[len(self.obj)-1].place(x=130, y=self.pos_y+20)
        self.obj.append(tkinter.Radiobutton(self.window, text="Female", variable=self.var,
                                            value=2, bg='#19191A', selectcolor='#000000', fg='#ffffff'))
        self.obj[len(self.obj)-1].place(x=130, y=self.pos_y+40)
        self.obj.append(tkinter.Label(
            self.window, text="Age", bg='#19191A', fg='#ffffff'))
        self.obj[len(self.obj)-1].place(x=210, y=self.pos_y)
        self.e1 = tkinter.Entry(self.window)  # สร้างกล่องรับข้อความ
        self.e1.place(x=220, y=self.pos_y+20,width=160)
        self.obj.append(tkinter.Label(
            self.window, text="Date", bg='#19191A', fg='#ffffff'))
        self.obj[len(self.obj)-1].place(x=210, y=self.pos_y+40)
        #self.e2 = tkinter.Entry(self.window)  # สร้างกล่องรับข้อความ
        self.e2 = EntryWithPlaceholder(self.window,"dd/mm/yyyy - dd/mm/yyyy")
        self.e2.place(x=220, y=self.pos_y+60,width=160)
        self.obj.append(tkinter.Button(
            self.window, text="Recognition", bg='#19191A', fg="white", width=30, command=self.prerecognition))
        self.obj[len(self.obj)-1].place(x=130, y=self.pos_y+90)
    #หยุดการค้นหา
    def stopRunning(self):
        self.running = False

    #ก่อนการค้นหา
    def prerecognition(self):
        self.filterSetting()
        Thread(target=self.recognitionProcess).start()
    #จดจำใบหน้า
    def recognitionProcess(self):
        for img in self.window.filename:
            img = cv2.imread(img)
            self.face_desc = []
            for i in range(1, 5):
                dets = detector(img, i)
                if len(dets) != 0:
                    for temp, bound in enumerate(dets):
                        shape = predictor(img, bound)
                        face_descripter = face_recognition.compute_face_descriptor(
                            img, shape, 1)
                        self.face_desc.append(face_descripter)
                        if not self.running:
                            break
                    break
        Thread(target=self.searching).start()
    #ค้นหาใบหน้า
    def searching(self):
        self.massage_2 = tkinter.Label(
            self.window, text='Face Searching', bg='#19191A', fg="white", font=Style.font(12))
        self.massage_2.place(x=710, y=83)
        shift_x = 0
        shift_y = 0
        counting = 0
        global res_list
        res_list = []
        k = 0
        for path in self.filename:
            img = cv2.imread(path)
            self.showText(str(int(k*100/len(self.filename)))+"%   ")
            if not self.running:
                break

            if self.recognitionFromBackup(img):
                self.showImg(path, x=720+shift_x, y=110+shift_y, size=128)
                res_list.append(path)
                shift_x += 135
                if shift_x >= 1000:
                    shift_y += 135
                    shift_x = 0
                if shift_y >= 900:
                    shift_y = 0

            k += 1
        self.showText("Done.")
        self.trackingResult()

    #ผลลัพธ์การค้นหา
    def trackingResult(self):
        track = Res(self.window)
        track.run()

    #จดจำใบหน้าจาก backup
    def recognitionFromBackup(self, img):
        for i in range(1, 3):
            try:
                dets = detector(img, i)
                if len(dets) != 0:
                    for k, d in enumerate(dets):
                        if not self.running:
                            break
                        shape = predictor(img, d)
                        face_descriptor = face_recognition.compute_face_descriptor(
                            img, shape, 1)
                        for FACE_DESC in self.face_desc:
                            if np.linalg.norm(np.array(FACE_DESC) - np.array(face_descriptor)) < 0.4:
                                return True
                    break
            except:
                pass
            
        return False
    #แสดงรูป
    def showImg(self, pic, x, y, size):
        self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.resize(cv2.cvtColor(
                cv2.imread(pic), cv2.COLOR_BGR2RGB), (size, size))))
        self.img = tkinter.Label(self.window, image=self.photo, borderwidth=0)
        self.obj.append(self.img)
        self.obj[len(self.obj)-1].image = self.photo
        self.obj[len(self.obj)-1].place(x=x, y=y)
    # หยุดการทำงานและทำลาย
    def DestroyComp(self):
        for obj in self.obj:
            obj.after(0, obj.destroy)
        self.obj = []
        self.running = False
    #แสดงข้อความ
    def showText(self, text):
        self.text = tkinter.Label(self.window, text=text, borderwidth=0, justify=tkinter.LEFT, width=10,
                                  bg='#19191A', fg='#ffffff', font=Style.font(11))
        self.obj.append(self.text)
        self.obj[len(self.obj)-1].place(x=110, y=self.pos_y+90+40)


class Res:
    def __init__(self, window):
        self.window = window
        self.run()

    # run program
    def run(self):
        self.load = PIL.Image.open('public/sign/print_settings (1).png')
        self.load.thumbnail((40, 40))
        self.render = PIL.ImageTk.PhotoImage(
            self.load)
        self.button1 = tkinter.Button(self.window, image=self.render, bg='#19191A',
                                      fg="white", font=Style.font(12), command=self.createNewWindow, borderwidth=0, highlightthickness=0)
        self.button1.image = self.render
        self.button1.place(x=100, y=1000)

    # export
    def exportTxT(self):
        file = filedialog.asksaveasfile(
            initialdir="./", mode='w', defaultextension=".txt", filetypes=(("txt", "*.txt"), ("all", "*.*")))
        if file is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        for path in res_list:
            file.write(path)
        file.close()
    #แสดงผลลัพธ์เรียงจากสถานที่ๆพบบ่อย
    def sortByCount(self):
        res = []
        for i in range(len(res_list)):
            arr = res_list[i].split("/")[-2]
            res.append(arr)
        ans = []
        for i in range(len(res)):
            ans.append([res[i], res.count(res[i])])
        ans = sorted(ans, key=itemgetter(1), reverse=True)
        a = []
        for i in range(len(ans)):
            if i == 0:
                a.append([ans[i][0], ans[i][1]])
            elif ans[i][0] != ans[i-1][0]:  # self.cstr(ans[i],ans[i-1]):
                a.append([ans[i][0], ans[i][1]])
        total = 0
        for i in range(len(a)):
            total += a[i][1]
        self.mylist.delete(0, tkinter.END)
        for i in range(len(a)):
            self.mylist.insert(tkinter.END, a[i][0] + " | " + str(a[i][1]))
        self.mylist.insert(tkinter.END, "Total : " + str(total))
    #แสดงผลลัพธ์เรียงจากชื่อ
    def sortByName(self):
        res = []
        for i in range(len(res_list)):
            n = res_list[i].split("/")[-2]
            d = res_list[i].split("/")[-5:-2]
            t = res_list[i].split("/")[-1]
            res.append([n, d, t])
        ans = []
        for i in range(len(res)):
            ans.append(res[i])

        ans = sorted(ans, key=itemgetter(0), reverse=True)
        self.mylist.delete(0, tkinter.END)
        arr = []
        for i in range(len(ans)):
            tt = ans[i][2].split("_")[0]
            hh = tt.split("-")[0]
            mm = tt.split("-")[1]
            ss = tt.split("-")[2]
            arr.append([ans[i][0], ans[i][1], [hh, mm, ss]])

        for i in range(len(arr)):
            ddmmyy = arr[i][1][0]+"/"+arr[i][1][1]+"/"+arr[i][1][2]
            hhmmss = arr[i][2][0]+":"+arr[i][2][1]+":"+arr[i][2][2]
            self.mylist.insert(tkinter.END, str(
                arr[i][0]) + " | " + ddmmyy + " | " + hhmmss)
    #แสดงผลลัพธ์เรียงจากวัน
    def sortByDate(self):
        res = []
        for i in range(len(res_list)):
            n = res_list[i].split("/")[-2]
            d = res_list[i].split("/")[-5:-2]
            t = res_list[i].split("/")[-1]
            res.append([n, d, t])
        ans = []
        for i in range(len(res)):
            ans.append(res[i])

        ans = sorted(ans, key=itemgetter(0), reverse=True)
        self.mylist.delete(0, tkinter.END)
        arr = []
        for i in range(len(ans)):
            tt = ans[i][2].split("_")[0]
            hh = tt.split("-")[0]
            mm = tt.split("-")[1]
            ss = tt.split("-")[2]
            arr.append([ans[i][0], ans[i][1], [hh, mm, ss]])

        arr = self.sortDate(arr)
        self.mylist.delete(0, tkinter.END)
        for i in range(len(arr)):
            ddmmyy = arr[i][1][0]+"/"+arr[i][1][1]+"/"+arr[i][1][2]
            hhmmss = arr[i][2][0]+":"+arr[i][2][1]+":"+arr[i][2][2]
            self.mylist.insert(tkinter.END, str(
                arr[i][0]) + " | " + ddmmyy + " | " + hhmmss)
    #เรียงลำดับวัน
    def sortDate(self, arr):
        return sorted(arr, key=lambda x: (x[1][0], x[1][1], x[1][2], x[2][0], x[2][1], x[2][2]))
    #แสดงผลลัพธ์เฉยๆ
    def allList(self):
        self.mylist.delete(0, tkinter.END)
        for i in range(len(res_list)):
            self.mylist.insert(i, res_list[i])
            i += 1

    #เทปผลลัพธ์
    def createNewWindow(self):
        self.newWindow = tkinter.Toplevel(
            self.window, bg='#19191A')
        self.newWindow.geometry('770x340+140+700')
        self.newWindow.overrideredirect(1)
        self.scrollbar = tkinter.Scrollbar(self.newWindow)
        self.scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        if os.name == 'nt':
            height_os = 19
            width_os = 19
        else:
            height_os = 22
            width_os = 15
        self.mylist = tkinter.Listbox(
            self.newWindow, width=130, height=height_os, yscrollcommand=self.scrollbar.set, selectmode=tkinter.EXTENDED)

        i = 0
        for j in range(len(res_list)):
            self.mylist.insert(i, res_list[j])
            i += 1

        self.mylist.pack()
        obj = []
        self.scrollbar.config(command=self.mylist.yview)
        obj.append(tkinter.Button(self.newWindow, text="ALL",
                                  font=Style.font(8), bg='#19191A', fg="white", width=width_os, borderwidth=4, command=self.allList))
        obj.append(tkinter.Button(self.newWindow, text="sort by count",
                                  font=Style.font(8), bg='#19191A', fg="white", width=width_os, borderwidth=4, command=self.sortByCount))
        obj.append(tkinter.Button(self.newWindow, text="sort by name",
                                  font=Style.font(8), bg='#19191A', fg="white", width=width_os, borderwidth=4, command=self.sortByName))
        obj.append(tkinter.Button(self.newWindow, text="sort by date",
                                  font=Style.font(8), bg='#19191A', fg="white", width=width_os, borderwidth=4, command=self.sortByDate))
        obj.append(tkinter.Button(self.newWindow, text="export",
                                  font=Style.font(8), bg='#19191A', fg="white", width=width_os, borderwidth=4, command=self.exportTxT))
        obj.append(tkinter.Button(self.newWindow, text="Close",
                                  font=Style.font(8), bg='#19191A', fg="white", width=width_os, borderwidth=4, command=self.newWindow.destroy))
        for Obj in obj:
            Obj.pack(side="left")

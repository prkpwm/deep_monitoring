import os
import shutil
import tkinter as tk
import tkinter.font as tkFont
from datetime import date, datetime, timedelta
from threading import Thread
from tkinter import filedialog

import cv2
import PIL.Image
import src.Components.Style as Style
from imutils import face_utils

output_path = "images/backup/"
separated_path = "images/separated/"
predicted_path = "images/predicted/"
del_d = 90
setting_config = []


class App:
    def __init__(self, window):
        self.window = window
        self.obj = []
        self.readFile()

    #อ่านไฟล์ setting
    def readFile(self):
        file_setting = open("setting.txt", "r")
        setting = file_setting.readlines()
        global setting_config
        for line in setting:
            if line != "\n":
                if line.endswith('\n'):
                    line = line[:-1]
                setting_config.append(line)
        global output_path
        global separated_path
        global predicted_path
        global del_d
        output_path = setting_config[0]
        separated_path = setting_config[1]
        predicted_path = setting_config[2]
        del_d = int(setting_config[3])
        self.delBackup()

    # run program
    def run(self):
        self.massage_1 = tk.Label(
            self.window, text='Setting', bg='#19191A', fg="white", font=Style.font(22))
        self.obj.append(self.massage_1)
        self.obj[len(self.obj)-1].place(x=20, y=80)
        self.massage_2 = tk.Label(
            self.window, text='Output Path', bg='#19191A', fg="white", font=Style.font(16))
        self.obj.append(self.massage_2)
        self.obj[len(self.obj)-1].place(x=50, y=150)
        self.e1 = tk.Entry(self.window, font=Style.font(16))
        self.e1.insert(tk.END, str(output_path))
        self.e1.place(x=230, y=150, width=600, height=35)
        self.button6 = tk.Button(self.window, text="Search", font=Style.font(10),
                                 bg='#19191A', fg="white", width=15, borderwidth=4, command=lambda: self.search(1))
        self.button6.place(x=850, y=155)

        self.massage_2 = tk.Label(
            self.window, text='Separate Path', bg='#19191A', fg="white", font=Style.font(16))
        self.obj.append(self.massage_2)
        self.obj[len(self.obj)-1].place(x=50, y=220)
        self.e2 = tk.Entry(self.window, font=Style.font(16))
        self.e2.insert(tk.END, str(separated_path))
        self.e2.place(x=230, y=220, width=600, height=35)
        self.button7 = tk.Button(self.window, text="Search", font=Style.font(10),
                                 bg='#19191A', fg="white", width=15, borderwidth=4, command=lambda: self.search(2))
        self.button7.place(x=850, y=220)

        self.massage_3 = tk.Label(
            self.window, text='Prediction Path', bg='#19191A', fg="white", font=Style.font(16))
        self.obj.append(self.massage_3)
        self.obj[len(self.obj)-1].place(x=50, y=290)
        self.e3 = tk.Entry(self.window, font=Style.font(16))
        self.e3.insert(tk.END, str(predicted_path))
        self.e3.place(x=230, y=290, width=600, height=35)
        self.button7 = tk.Button(self.window, text="Search", font=Style.font(10),
                                 bg='#19191A', fg="white", width=15, borderwidth=4, command=lambda: self.search(3))
        self.button7.place(x=850, y=290)

        self.massage_4 = tk.Label(
            self.window, text='Time for Backup', bg='#19191A', fg="white", font=Style.font(16))
        self.obj.append(self.massage_4)
        self.obj[len(self.obj)-1].place(x=50, y=360)
        self.e4 = tk.Entry(self.window, font=Style.font(16))
        self.e4.insert(tk.END, str(del_d))
        self.e4.place(x=230, y=360, width=600, height=35)
        self.button8 = tk.Button(self.window, text="Enter", font=Style.font(10),
                                 bg='#19191A', fg="white", width=15, borderwidth=4, command=self.saveDate)
        self.button8.place(x=700, y=440)
    #ช่วงวัน
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)
    #ลบข้อมูล
    def delBackup(self):
        self.filename = []
        for dirpath, dirnames, files in os.walk(predicted_path):
            if not os.listdir(dirpath) :
                os.rmdir(dirpath)
            for file_name in files:
                rename = dirpath+"/"+file_name
                rename = rename.replace("\\", "/")
                self.filename.append(rename)
        date_get = []
        for d in self.filename:
            d = d.split("/")[2:5]
            fd = d[0]+"/"+d[1]+"/"+ d[2]
            if fd not in date_get:
                date_get.append(fd)
        date_format = "%Y/%m/%d"
        today = str(date.today())
        today = today.replace("-", "/")
        res = []
        for d in date_get:
            a= datetime.strptime(d, date_format)
            b= datetime.strptime(today, date_format)
            delta = b - a
            if delta.days > del_d:
                shutil.rmtree(predicted_path+d)
    #ค้นหา path
    def search(self,id):
        folder_selected = filedialog.askdirectory()
        if id == 1:
            self.e1.delete(0, tk.END)
            self.e1.insert(tk.END, str(folder_selected))
            self.savePath()
        elif id == 2:
            self.e2.delete(0, tk.END)
            self.e2.insert(tk.END, str(folder_selected))
            self.saveSeperate()
        else:
            self.e3.delete(0, tk.END)
            self.e3.insert(tk.END, str(folder_selected))
            self.savePredict()
    #save ไฟล์ setting
    def saveFile(self):
        file_setting = open("setting.txt", "w")
        for massage in setting_config:
            file_setting.write("%s\n" % massage)
        file_setting.close()
    #บันทึกวัน ที่ต้องการลบ
    def saveDate(self):
        global setting_config
        setting_config[3] = self.e4.get()
        self.saveFile()

   # บันทึก path สำหรับเก็บ ข้อมูลที่ภาพใบหน้าแล้ว
    def savePath(self):
        global output_path
        output_path = self.e1.get()
        output_path += "" if self.e1.get().endswith("/") else "/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        global setting_config
        setting_config[0] = output_path
        self.saveFile()
    # บันทึก path สำหรับเก็บ ข้อมูลที่แยกภาพใบหน้าแล้ว
    def saveSeperate(self):
        global separated_path
        separated_path = self.e2.get()
        separated_path += "" if self.e2.get().endswith("/") else "/"
        if not os.path.exists(separated_path):
            os.makedirs(separated_path)
        global setting_config
        setting_config[1] = separated_path
        self.saveFile()
     # บันทึก path สำหรับเก็บ ข้อมูลที่ทำนายใบหน้าแล้ว
    def savePredict(self):
        global predicted_path
        predicted_path = self.e3.get()
        predicted_path += "" if self.e3.get().endswith("/") else "/"
        if not os.path.exists(predicted_path):
            os.makedirs(predicted_path)
        global setting_config
        setting_config[2] = predicted_path
        self.saveFile()

    # หยุดการทำงานและทำลาย
    def DestroyComp(self):
        for obj in self.obj:
            obj.after(0, obj.destroy)
        self.obj = []

import os
import threading
import tkinter as tk
import tkinter.font as tkFont
from multiprocessing import Process
from tkinter import filedialog, ttk

import cv2
import PIL.Image
import src.Components.Style as Style
import src.Pages.Face_Detection as Face_Detection
import src.Pages.Heat_Map as Heat_Map
import src.Pages.People_Counting as People_Counting

num_dis = 4
entry_list = []


class App:
    def __init__(self, window, name):
        self.window = window
        self.name = name
        self.num = 4
        self.obj = []
        self.size_pack = [[1, 1595, 900, 0, 0, 1], [4, 798, 453, 798, 453, 2], [9, 533, 303, 533, 303, 3], [
            16, 397, 227, 397, 227, 4], [100, 317, 180, 317, 180, 5]]  # จำนวนหน้าจอต่ออื่นๆ
        self.init_preprocess()

    '''ปุ่มเลือกจำนวนหน้าจอ'''
    def build_btn_dis_select(self):
        self.obj.append(tk.Button(self.window, text='1', bg='#19191A', fg="white",
                                  font=Style.font(12), width=3, borderwidth=4, command=self.select1))
        self.obj[len(self.obj)-1].place(x=10, y=1000)

        self.obj.append(tk.Button(self.window, text='4', bg='#19191A', fg="white",
                                  font=Style.font(12), width=3, borderwidth=4, command=self.select4))
        self.obj[len(self.obj)-1].place(x=60, y=1000)

        self.obj.append(tk.Button(self.window, text='9', bg='#19191A', fg="white",
                                  font=Style.font(12), width=3, borderwidth=4, command=self.select9))
        self.obj[len(self.obj)-1].place(x=110, y=1000)

        self.obj.append(tk.Button(self.window, text='16', bg='#19191A', fg="white",
                                  font=Style.font(12), width=3, borderwidth=4, command=self.select16))
        self.obj[len(self.obj)-1].place(x=160, y=1000)

        self.obj.append(tk.Button(self.window, text='25', bg='#19191A', fg="white",
                                  font=Style.font(12), width=3, borderwidth=4, command=self.select25))
        self.obj[len(self.obj)-1].place(x=210, y=1000)

    # run program
    def init_preprocess(self):
        self.show_component()
        self.build_btn_dis_select()
        self.lb = ListBox(self.window)  # listbox
        self.lb.run()

    # run program
    def show_component(self):
        '''แสดงข้อความ'''
        self.massage_1 = tk.Label(
            self.window, text='Face Detection', bg='#19191A', fg="white", font=Style.font(10))
        self.massage_1.place(x=1630, y=53)
        self.massage_2 = tk.Label(
            self.window, text='Face Detection' if self.name == "detect" else 'Heat map' if self.name == "heatmap" else 'People Counting', bg='#19191A', fg="white", font=Style.font(10))
        self.massage_2.place(x=10, y=53)

        '''ปุ่มเล่นและปิด'''
        self.showImgWithComm(pic='public/sign/play.png', x=1542, y=55,
                             size=20, color='#19191A', comm=self.startDetection)
        self.showImgWithComm(pic='public/sign/stop.png', x=1570, y=55,
                             size=20, color='#19191A', comm=self.stopMultipleDisplay)
        self.shift_y = 0
        self.imd_id = 0

        '''วาดพื้นหลัง'''
        for i in range(7):
            self.shift_x = 0
            for j in range(2):
                self.showImg('public/Wallpaper/blank-profile-2.png', x=1630 +
                             self.shift_x, y=80+self.shift_y, width=128, height=128)  # แสดงภาพที่ยังไม่ได้ทำการ detect
                self.imd_id += 1
                self.shift_x += 145
            self.shift_y += 140

        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=80, width=1617, height=920)  # วาดพื้นหลัง

        shift_y = 0
        for f in range(5):
            if self.num <= self.size_pack[f][0]:
                for i in range(self.size_pack[f][5]):
                    shift_x = 0
                    for j in range(self.size_pack[f][5]):
                        self.showImg('public/Wallpaper/wall3.png', x=10+shift_x,
                                     y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2])  # วาดพื้นหลัง
                        shift_x += self.size_pack[f][3]
                    shift_y += self.size_pack[f][4]
                break

   # หยุดการทำงานและทำลาย
    def DestroyComp(self):
        for obj in self.obj:
            obj.after(0, obj.destroy)
        self.obj = []

    '''เลื่อกจำนวนหน้าจอ'''

    def select1(self):
        self.num = 1
        self.show_component()

    def select4(self):
        self.num = 4
        self.show_component()

    def select9(self):
        self.num = 9
        self.show_component()

    def select16(self):
        self.num = 16
        self.show_component()

    def select25(self):
        self.num = 25
        self.show_component()

    # เล่น vdo
    def startDetection(self):
        try:
            for obj in self.display_obj:
                obj.stop()
        except:
            pass
        self.showImg('public/Wallpaper/fill.png', x=10,
                     y=80, width=1617, height=920)  # วาดพื้นหลัง
        shift_y = 0
        k = 0
        self.display_obj = []
        # วาดพื้นหลัง
        for f in range(5):
            if self.num <= self.size_pack[f][0]:
                for i in range(self.size_pack[f][5]):
                    shift_x = 0
                    for j in range(self.size_pack[f][5]):
                        self.showImg('public/Wallpaper/wall3.png', x=10 +
                                     shift_x, y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2])
                        shift_x += self.size_pack[f][3]
                    shift_y += self.size_pack[f][4]
                break
        shift_y = 0
        # เล่น vdo
        global num_dis
        num_dis = self.num
        for f in range(5):
            if self.num <= self.size_pack[f][0]:
                for i in range(self.size_pack[f][5]):
                    shift_x = 0
                    for j in range(self.size_pack[f][5]):
                        if k >= self.num:
                            break
                        else:
                            arr = entry_list[k].split("|")

                            if len(arr) == 1:
                                if self.name == "detect":

                                    self.vdo = Face_Detection.Detection(
                                        self.window, vdo='{}'.format(entry_list[k]), x=10+shift_x, y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2], name="CCTV")
                                    self.display_obj.append(self.vdo)
                                    self.display_obj[len(
                                        self.display_obj)-1].run()
                                
                                elif self.name == "heatmap":
                                    self.vdo = Heat_Map.Heat_Map(
                                        self.window, vdo='{}'.format(entry_list[k]), x=10+shift_x, y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2])
                                    self.display_obj.append(self.vdo)
                                    self.display_obj[len(
                                        self.display_obj)-1].run()
                                elif self.name == "peoplecounting":
                                    self.vdo = People_Counting.PeopleCount(
                                        self.window, vdo='{}'.format(entry_list[k]), x=10+shift_x, y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2], coord=[0, 260, 1920, 260])
                                    self.display_obj.append(self.vdo)
                                    self.display_obj[len(
                                        self.display_obj)-1].run()
                            else:
                                path = arr[0].replace(" ", "")
                                name = arr[2].split("=")[1].replace(" ", "")
                                if arr[1].split("=")[1].replace(" ", "") == "0":
                                    self.vdo = Face_Detection.Detection(
                                        self.window, vdo=path, x=10+shift_x, y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2], name=name , confident=int(arr[3].split("=")[1]),crop_time=float(arr[4].split("=")[1]),tracking=True if arr[5].split("=")[1]== " on" else False)
                                    self.display_obj.append(self.vdo)
                                    self.display_obj[len(
                                        self.display_obj)-1].run()
                                elif arr[1].split("=")[1].replace(" ", "") == "1":
                                    line = int(arr[3].split("=")[
                                               1].replace(" ", ""))
                                    axis = arr[3].split(
                                        "=")[0].replace(" ", "") == "x"
                                    start_t = arr[4].split(
                                        "=")[1].replace(" ", "").split("-")[0]
                                    end_t = arr[4].split(
                                        "=")[1].replace(" ", "").split("-")[1]
                                    d = arr[5].split(
                                        "=")[1].replace(" ", "")
                                    self.vdo = People_Counting.PeopleCount(
                                        self.window, vdo=path, x=10+shift_x, y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2], name=name, coord=[line if axis else 0, line if not axis else 0, line if axis else 1920, line if not axis else 1080], start_t=int(start_t), end_t=int(end_t), direction=d, confident=int(arr[6].split("=")[1]),crop_time=float(arr[7].split("=")[1]),tracking=True if arr[8].split("=")[1]== 'on' else False)
                                    self.display_obj.append(self.vdo)
                                    self.display_obj[len(
                                        self.display_obj)-1].run()
                                elif arr[1].split("=")[1].replace(" ", "") == "2":
                                    time = int(''.join(i for i in arr[3].split("=")[
                                               1].replace(" ", "") if i.isdigit()))
                                    list_t = [['s', 1], ['m', 60],
                                              ['h', 3600], ['d', 86400]]
                                    char_t = arr[3].split(
                                        "=")[1].replace(" ", "")
                                    for i in range(4):
                                        if char_t.endswith(list_t[i][0]):
                                            time *= list_t[i][1]
                                            break

                                    self.vdo = Heat_Map.Heat_Map(
                                        self.window, vdo=path, x=10+shift_x, y=80+shift_y, width=self.size_pack[f][1], height=self.size_pack[f][2], name=name, save_t=time, confident=int(arr[4].split("=")[1]),crop_time=float(arr[5].split("=")[1]),tracking=True if arr[5].split("=")[6]== 'on' else False)
                                    self.display_obj.append(self.vdo)
                                    self.display_obj[len(
                                        self.display_obj)-1].run()
                        k += 1
                        shift_x += self.size_pack[f][3]
                    shift_y += self.size_pack[f][4]
                break

    # หยุดการทำงาน
    def stopMultipleDisplay(self):
        for obj in self.display_obj:
            obj.stop()



    '''โหลดรูป'''
    def showImg(self, pic, x, y, width, height):
        self.render = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.resize(cv2.cvtColor(
                cv2.imread(pic), cv2.COLOR_BGR2RGB), (width, height))))
        self.obj.append(
            tk.Label(self.window, image=self.render, borderwidth=0))  # ตั้งค่ารูปภาพ
        self.obj[len(self.obj)-1].image = self.render  # ตั้งค่ารูปภาพ
        self.obj[len(self.obj)-1].place(x=x, y=y)  # กำหนดตำแหน่ง

    def showImgWithComm(self, pic, x, y, size, color, comm):
        self.load = PIL.Image.open(pic)  # โหลดรูป
        self.load.thumbnail((size, size))  # ปรับขนาด
        self.render = PIL.ImageTk.PhotoImage(
            self.load)  # เปลี่ยนเป็น PhotoImage
        self.img = tk.Button(self.window, image=self.render,
                             bg=color, borderwidth=0, command=comm, highlightthickness=0)    # ตั้งค่าปุ่ม
        self.img.image = self.render  # ตั้งค่ารูปภาพ
        self.img.place(x=x, y=y)  # กำหนดตำแหน่ง

    # หยุดการทำงาน
    def Destroy(self):
        for obj in self.obj:
            obj.after(0, obj.destroy)
        self.obj = []


class ListBox:
    def __init__(self, window):
        self.window = window
        self.run()
        self.obj = []

    def showImg(self, pic, x, y, width, height):
        self.render = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.resize(cv2.cvtColor(
                cv2.imread(pic), cv2.COLOR_BGR2RGB), (width, height))))
        self.obj.append(
            tk.Label(self.app, image=self.render, borderwidth=0))  # ตั้งค่ารูปภาพ
        self.obj[len(self.obj)-1].image = self.render  # ตั้งค่ารูปภาพ
        self.obj[len(self.obj)-1].place(x=x, y=y)  # กำหนดตำแหน่ง

    # run program
    def run(self):

        # ปุ่มจัดการ cctv
        self.load = PIL.Image.open('public/sign/config.png')  # โหลดรูป
        self.load.thumbnail((40, 40))  # ปรับขนาด
        self.render = PIL.ImageTk.PhotoImage(
            self.load)  # เปลี่ยนเป็น PhotoImage
        self.button1 = tk.Button(self.window, image=self.render, bg='#19191A',
                                 fg="white", font=Style.font(12), command=self.createNewWindow, borderwidth=0, highlightthickness=0)  # สร้างปุ่ม Manage Camera
        self.button1.image = self.render  # ตั้งค่ารูปภาพ
        self.button1.place(x=1570, y=1000)  # กำหนดพิกัด

    # เพิ่มจำนวน path
    def add(self):
        self.newWindow2 = tk.Toplevel(
            self.window, bg='#19191A')  # สร้าง window รับข้อความ
        self.newWindow2.geometry('500x70+300+997')   #
        self.newWindow2.overrideredirect(1)  # ให้เขียนทับอบู่บนสุด
        self.e = tk.Entry(self.newWindow2)  # สร้างกล่องรับข้อความ
        self.e.pack(ipadx=200, ipady=8, expand=1)  # กำหนดพิกัด
        self.text = self.e.get()  # รับค่าเก็บใน text
        self.button6 = tk.Button(self.newWindow2, text="Enter", font=Style.font(10),
                                 bg='#19191A', fg="white", width=13, borderwidth=4, command=self.save)  # สร้างปุ่ม enter
        self.button6.pack(side="right")  # กำหนดพิกัด
        self.button7 = tk.Button(self.newWindow2, text="Help", font=Style.font(10),
                                 bg='#19191A', fg="white", width=13, borderwidth=4, command=self.helpUI)  # สร้างปุ่ม enter
        self.button7.pack(side="right")  # กำหนดพิกัด
        # self.mylist.insert(self.text)

    # บันทึกลง list
    def save(self):
        if len(self.e.get()) > 0:
            if not self.e.get().isspace():
                entry_list.append(self.e.get())  # เพิ่มค่าที่ได้รับมา
                # เพิ่มค่าที่ได้รับมา ไปใน mylist
                self.mylist.insert(tk.END, self.e.get())
        self.newWindow2.destroy()  # ทำลายหน้า รับข้อความ

    # ลบข้อมูลใน list

    def delete(self):
        sl = self.mylist.curselection()  # เลือกจาก curser
        for index in sl[::-1]:
            self.mylist.delete(index)  # ลบข้อมูลใน list
        entry_list.clear()  # ลบข้อมูลใน list
        for i in range(self.mylist.size()):
            entry_list.append(self.mylist.get(i))

    # ลบข้อมูลทั้งหมด
    def deleteAll(self):
        entry_list.clear()  # ลบข้อมูลใน list
        self.mylist.delete(0, tk.END)  # ลบข้อมูลใน list

    # import ไฟล์การตั้งค่า
    def importTxT(self):
        self.window.filename = filedialog.askopenfilenames(
            initialdir="./", title="Select File", filetypes=(("txt", "*.txt"), ("all", "*.*")))
        if len(self.window.filename) > 0:
            file = open(self.window.filename[0], "r")
            for path in file:
                if path is not "\n":
                    if path.endswith("\n"):
                        path = path[:-1]
                    entry_list.append(path)  # เพิ่มค่าที่ได้รับมา
                    # เพิ่มค่าที่ได้รับมา ไปใน mylist
                    self.mylist.insert(tk.END, path)

    # export ไฟล์การตั้งค่า
    def exportTxT(self):
        file = filedialog.asksaveasfile(
            initialdir="./", mode='w', defaultextension=".txt", filetypes=(("txt", "*.txt"), ("all", "*.*")))
        if file is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        for path in entry_list:
            file.write(path + '\n')
        file.close()

    def getIP(self):
        pass

    def saveFromUi(self):
        if len(self.Path_Entry1.get())>0 and len(self.Path_Entry2.get())>0:
            self.property = ""
            self.property += self.Path_Entry1.get()+" | "
            self.property += "Select = "
            self.property += "0 " if self.combo1.get() == "Face detection" else "1 " if self.combo1.get() == "People counting" else "2 " 
            self.property +=" | "
            self.property += "name = "+self.Path_Entry2.get()
            if self.combo1.get() == "People counting":
                self.property +=" | "
                self.property += "x = " if self.combo2.get() == "Vertical" else "y = "
                self.property += self.Path_Entry3.get() + " | "
                self.property += "time = " + self.Path_Entry4.get() + " - " + self.Path_Entry5.get() + " | "
                self.property += "direction = "
                self.property +=  "1" if self.combo3.get() == "Up -> Down" else "2" if self.combo3.get(
                ) == "Down -> Up" else "3" if self.combo3.get() == "Left -> Right" else "4"
            
            elif self.combo1.get() == "Heatmap":
                self.property += " | "
                self.property += "time = " + self.Path_Entry6.get()
                self.property += self.combo4.get()[0]
            
            self.property += " | "
            self.property += "confident = " + (self.Config_Entry1.get() if (self.Config_Entry1.get()!="" and self.Config_Entry1.get()!="1-100") else "30")
            self.property += " | "
            self.property += "crop_time = " + (self.Config_Entry2.get() if (self.Config_Entry2.get()!="" and self.Config_Entry2.get()!="0-60") else "1.5")
            self.property += " | "
            self.property += "tracking = " + ('on' if repr(str(self.var.get())) == repr(str(1)) else 'off')
            if len(self.property) > 0:
                entry_list.append(self.property)  # เพิ่มค่าที่ได้รับมา
                # เพิ่มค่าที่ได้รับมา ไปใน mylist
                self.mylist.insert(tk.END, self.property)
        self.app.destroy()  # ทำลายหน้า รับข้อความ

    def helpUI(self):
        self.newWindow2.destroy()
        self.app = tk.Toplevel(
            self.window, bg='#19191A')  # สร้าง window รับข้อความ
        self.app.geometry('770x600+800+500')   #
        self.app.overrideredirect(1)  # ให้เขียนทับอบู่บนสุด
        if os.name == 'nt': # os checking  #windows?
            self.height_os = 19
            self.width_os = 19
        else:
            self.height_os = 22
            self.width_os = 15
        self.data_ui = []
        self.list_prop = []
        self.list_prop.append(tk.Label(self.app,
                                       text="Property", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=10, y=10)
        self.list_prop.append(tk.Label(self.app,
                                       text="Path", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=40)
        self.Path_Entry1 = tk.Entry(self.app)  # สร้างกล่องรับข้อความ
        self.Path_Entry1.place(x=100, y=40)  # กำหนดพิกัด
        self.list_prop.append(tk.Label(self.app,
                                       text="Name", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=70)

        self.Path_Entry2 = tk.Entry(self.app)  # สร้างกล่องรับข้อความ
        self.Path_Entry2.place(x=100, y=70)  # กำหนดพิกัด

        self.list_prop.append(tk.Label(self.app,
                                       text="Confident", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=320, y=40)
        self.Config_Entry1 = EntryWithPlaceholder(self.app,"1-100")  # สร้างกล่องรับข้อความ
        self.Config_Entry1.place(x=400, y=40)  # กำหนดพิกัด
        
        self.list_prop.append(tk.Label(self.app,
                                       text="Crop Time", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=320, y=70)
        self.Config_Entry2 = EntryWithPlaceholder(self.app,"0-60")  # สร้างกล่องรับข้อความ
        self.Config_Entry2.place(x=400, y=70)  # กำหนดพิกัด
        
        self.list_prop.append(tk.Label(self.app,
                                       text="second; 0 mean not save crop-image", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=500, y=70)

        self.var = tk.IntVar()
        self.list_prop.append(tk.Label(
            self.app, text="Tracking", bg='#19191A', fg='#ffffff'))
        self.list_prop[len(self.list_prop)-1].place(x=320, y=100)
        self.list_prop.append(tk.Radiobutton(self.app, text="on", variable=self.var,
                                            value=1, bg='#19191A', selectcolor='#000000', fg='#ffffff'))
        self.list_prop[len(self.list_prop)-1].place(x=400, y=100)
        self.list_prop.append(tk.Radiobutton(self.app, text="off", variable=self.var,
                                            value=2, bg='#19191A', selectcolor='#000000', fg='#ffffff'))
        self.list_prop[len(self.list_prop)-1].place(x=450, y=100)

        self.list_prop.append(tk.Label(self.app,
                                       text="Select", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=100)
        self.combo1 = ttk.Combobox(self.app,
                                   values=[
                                       "Face detection",
                                       "People counting",
                                       "Heatmap"])
        self.combo1.current(0)
        self.combo1.place(x=100, y=100)
        self.combo1.bind("<<ComboboxSelected>>", self.callbackFunc)
        self.list_prop.append(tk.Button(self.app, text="Enter", font=Style.font(8),
                                        bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.saveFromUi))  # สร้างปุ่ม Add
        self.list_prop[len(self.list_prop)-1].place(x=600, y=500)

    def callbackFunc(self, event):
        self.showImg('public/Wallpaper/fill.png', x=0,
                     y=120, width=1000, height=1000)  # วาดพื้นหลัง
        self.list_prop.append(tk.Button(self.app, text="Enter", font=Style.font(8),
                                        bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.saveFromUi))  # สร้างปุ่ม Add
        self.list_prop[len(self.list_prop)-1].place(x=600, y=500)
        if self.combo1.get() == 'People counting':
            self.pcUI()
        elif self.combo1.get() == 'Heatmap':
            self.hmUI()


    def pcUI(self):
        self.list_prop.append(tk.Label(self.app,
                                       text="Axis of Line ", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=130)
        self.combo2 = ttk.Combobox(self.app,
                                   values=[
                                       "Vertical",
                                       "Horizon"])
        self.combo2.current(0)
        self.combo2.place(x=100, y=130)

        self.list_prop.append(tk.Label(self.app,
                                       text="Number in percent of frame", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=160)
        self.Path_Entry3 = EntryWithPlaceholder(self.app, "1-100")  # สร้างกล่องรับข้อความ
        self.Path_Entry3.place(x=200, y=160)  # กำหนดพิกัด
        self.list_prop.append(tk.Label(self.app,
                                       text="Time for Running Counting", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=190)
        self.list_prop.append(tk.Label(self.app,
                                       text="Start Time ", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=30, y=220)
        self.Path_Entry4 = EntryWithPlaceholder(self.app,"0-24")  # สร้างกล่องรับข้อความ
        self.Path_Entry4.place(x=100, y=220)  # กำหนดพิกัด
        self.list_prop.append(tk.Label(self.app,
                                       text="End Time ", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=30, y=250)
        self.Path_Entry5 = EntryWithPlaceholder(self.app,"0-24")  # สร้างกล่องรับข้อความ
        self.Path_Entry5.place(x=100, y=250)  # กำหนดพิกัด
        self.list_prop.append(tk.Label(self.app,
                                       text="Direction from Outside -> Inside", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=280)
        self.combo3 = ttk.Combobox(self.app,
                                   values=[
                                       "Up -> Down",
                                       "Down -> Up",
                                       "Left -> Right",
                                       "Right -> Left"])
        self.combo3.current(0)
        self.combo3.place(x=200, y=280)

    def hmUI(self):
        self.list_prop.append(tk.Label(self.app,
                                       text="Time for save image", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=20, y=130)
        self.list_prop.append(tk.Label(self.app,
                                       text="Number of time ", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=30, y=160)
        self.Path_Entry6 = tk.Entry(self.app)  # สร้างกล่องรับข้อความ
        self.Path_Entry6.place(x=150, y=160)  # กำหนดพิกัด
        self.list_prop.append(tk.Label(self.app,
                                       text="Name of time ", bg='#19191A', fg="white"))
        self.list_prop[len(self.list_prop)-1].place(x=30, y=190)
        self.combo4 = ttk.Combobox(self.app,
                                   values=[
                                       "second",
                                       "minute",
                                       "hour",
                                       "day"])
        self.combo4.current(2)
        self.combo4.place(x=150, y=190)

    # แสดง listbox

    def createNewWindow(self):
        self.newWindow = tk.Toplevel(
            self.window, bg='#19191A')  # สร้างหน้า listbox
        self.newWindow.geometry('770x340+800+700')  # กำหนดขนาดและตำแหน่ง
        self.newWindow.overrideredirect(1)  # ให้เขียนทับอบู่บนสุด

        # สร้าง scrollbar
        self.scrollbar = tk.Scrollbar(self.newWindow)  # สร้าง Scrollbar
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # กำหนดพิกัด

        # สร้าง Listbox

        if os.name == 'nt': # os checking  #windows?
            self.height_os = 19
            self.width_os = 19
        else:
            self.height_os = 22
            self.width_os = 15

        self.mylist = tk.Listbox(
            self.newWindow, width=130, height=self.height_os, yscrollcommand=self.scrollbar.set, selectmode=tk.EXTENDED)  # สร้าง Listbox
        # เก็บข้อมูลลง list
        i = 0
        for j in range(len(entry_list)):
            # เพิ่มข้อมูลจาก entry_list ไปยัง Listbox
            self.mylist.insert(i, entry_list[j])
            i += 1

        self.mylist.pack()  # กำหนดพิกัด
        self.scrollbar.config(command=self.mylist.yview)  # กำหนด scrollbar

        '''ป่มฟังก์ชั่น'''
        self.button2 = tk.Button(self.newWindow, text="Add", font=Style.font(8),
                                 bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.add)  # สร้างปุ่ม Add
        self.button2.pack(side="left")
        self.button3 = tk.Button(self.newWindow, text="Delete", font=Style.font(8),
                                 bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.delete)  # สร้างปุ่ม Delete
        self.button3.pack(side="left")
        self.button4 = tk.Button(self.newWindow, text="Delete All",
                                 font=Style.font(8), bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.deleteAll)  # สร้างปุ่ม Delete All
        self.button4.pack(side="left")
        self.button5 = tk.Button(self.newWindow, text="import",
                                 font=Style.font(8), bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.importTxT)  # สร้างปุ่ม Delete All
        self.button5.pack(side="left")
        self.button6 = tk.Button(self.newWindow, text="export",
                                 font=Style.font(8), bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.exportTxT)  # สร้างปุ่ม Delete All
        self.button6.pack(side="left")
        self.button7 = tk.Button(self.newWindow, text="Close", font=Style.font(8),
                                 bg='#19191A', fg="white", width=self.width_os, borderwidth=4, command=self.newWindow.destroy)  # สร้างปุ่ม Close
        self.button7.pack(side="left")

class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey'):
        super().__init__(master)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()

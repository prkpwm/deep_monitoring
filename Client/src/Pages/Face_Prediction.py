import json
import os
import shutil
import tkinter as tk
import tkinter.font as tkFont
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
from src.utils import google_utils
from src.utils.datasets import *
from src.utils.utils import *
from torchvision import transforms

''' LOAD MODEL '''
model_emotion = torch.load('./models/emotion_torch_81_48_3_res_50.pt')
model_gender = torch.load('./models/gender_torch_97_48_3_res_50.pt')
model_age = torch.load('./models/age_torch_11_224_3_res_50_new.pt')

'''model evaluate'''
model_emotion.eval()
model_gender.eval()
model_age.eval()


''' LABEL MODEL '''
sex = ['female', 'male', ]
emotion = ['Angry', 'Disgust', 'Fear',
           'Happy', 'Neutral', 'Sad', 'Surprise']
age = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
       '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80']

stat = []
stat_json = []
dashboard = []


class App:
    def __init__(self, window):
        self.window = window
        self.imd_id = 0
        self.obj = []
        self.obj_img = []
        self.btn_lock = False
        self.interrupt_process = False

    # run program
    def run(self):

        self.massage_1 = tk.Label(
            self.window, text='Face Prediction', bg='#19191A', fg="white", font=Style.font(11))
        self.obj.append(self.massage_1)
        self.obj[len(self.obj)-1].place(x=10, y=53)

        sign = [['public/sign/double left.png', self.doubleLeft], ['public/sign/left.png', self.left], ['public/sign/stop.png',
                self.stopProcess], ['public/sign/right.png', self.right], ['public/sign/double right.png', self.doubleRight]]
        shift = 0
        for i in range(len(sign)):
            self.showImgWithComm(pic=sign[i][0], x=860+shift, y=1040,
                                    size=30, color='#19191A', comm=sign[i][1])
            shift += 40
        self.showImgWithComm(pic='public/sign/save.png', x=10, y=1040,
                             size=30, color='#19191A', comm=self.exportTxT)
        self.showImgWithComm(pic='public/sign/face_ID.png', x=50, y=1040,
                             size=30, color='#19191A', comm=self.preeparate)

        self.interrupt_process = False
        Thread(target=self.preprocess).start()

    # บันทึกไฟล์ สำหรับ API
    def exportTxT(self):
        file = filedialog.asksaveasfile(
            initialdir="./", mode='w', defaultextension=".json", filetypes=(("json", "*.json"), ("all", "*.*")))
        if file is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        global stat_json
        transform = {'data': [self.getData(s) for s in stat_json]}
        transform = json.dumps(transform)
        file.write(str(transform))
        file.close()
        stat_json = []

    # แปลงข้อมูลเป็น dictionary type
    def getData(self, data):
        result = {}
        for d in data:
            find_colon = d.find(':')
            if find_colon > -1:
                k, v = d.split(':')
                result[k] = v if k != 'Age' else int(v)
            else:
                result['id'] = d
        return result
    # อ่านไฟล์รูปทั้งหมด
    def preprocess(self):
        self.imd_id = 0
        if not os.path.exists(Setting.separated_path):
            os.makedirs(Setting.separated_path)
        folders = Setting.separated_path
        self.filename = []
        for dirpath, dirnames, files in os.walk(folders):
            if not os.listdir(dirpath) :
                os.rmdir(dirpath)
            dirpath = dirpath.replace(Setting.output_path, "")
            for file_name in files:
                self.filename.append(dirpath+"/"+file_name)
        #Thread(target=self.PredictionProcess).start()

    def preeparate(self):
        Thread(target=self.separate).start()

    # แยกภาพใบหน้าที่ชัดเจน
    def separate(self):
        folders = Setting.output_path
        detector = dlib.get_frontal_face_detector()
        if not os.path.exists(Setting.separated_path):
            os.makedirs(Setting.separated_path)
        self.filename = []
        self.dirpath = []
        for dirpath, dirnames, files in os.walk(folders):
            if not os.listdir(dirpath) :
                os.rmdir(dirpath)
            dirpath = dirpath.replace(Setting.output_path, "")
            for file_name in files:
                self.filename.append(dirpath+"/"+file_name)
                self.dirpath.append(dirpath)
        k = 0
        for filename in self.filename:
            if not self.interrupt_process:
                self.textLoad(
                    str(int(k*100/len(self.filename)))+"%   "+filename)
                try:
                    img = cv2.imread(Setting.output_path + filename)
                    for i in range(1, 3):
                        rects = detector(img, i)
                        if len(rects) != 0:
                            for (i, rect) in enumerate(rects):
                                coords = (x, y, w, h) = face_utils.rect_to_bb(rect)
                                if len(coords) == 4:
                                    if not os.path.exists(Setting.separated_path+self.dirpath[k]):
                                        os.makedirs(Setting.separated_path+self.dirpath[k])
                                    cv2.imwrite(Setting.separated_path+filename, img)
                            break
                    os.remove(Setting.output_path+filename)
                except:
                    pass
                k += 1
        if not self.interrupt_process:
            self.textLoad("clear")
            #Thread(target=self.preprocess).start()

    # แสดงข้อความ
    def textLoad(self, text):
        self.img = tk.Label(
            self.window, text=text, borderwidth=0, font=Style.font(11), justify=tk.LEFT, width=50, fg="#ffffff", bg="#19191A")
        self.img.place(x=100, y=1050)

    # ทำนาย
    def predictMassage(self, x, y):
        photo = cv2.imread('{}'.format(self.filename[self.imd_id]))
        arr = Prediction(self.window, image=photo).run()
        sex = arr[0][0]+","+str(round(arr[0][1].item()*100))+"%\n"
        age = arr[1][0]+"\n"
        emotion = arr[2][0]+","+str(round(arr[2][1].item()*100))+"%"

        global stat
        global stat_json
        global dashboard
        self.filename[self.imd_id] = self.filename[self.imd_id].replace(
            '\\', "/")
        stat.append([arr[0][0], arr[1][0], arr[2]
                     [0], self.filename[self.imd_id]])
        stat_json.append([arr[0][0], arr[1][0], arr[2]
                     [0], self.filename[self.imd_id]])

        dashboard.append([arr[0][0], arr[1][0], arr[2]
                     [0], self.filename[self.imd_id]])
        if round(arr[0][1].item()*100) > 80:
            self.showText(x, y, sex, "#01DF01")
        elif round(arr[0][1].item()*100) > 50:
            self.showText(x, y, sex, "#FFFF00")
        else:
            self.showText(x, y, sex, "#FF0000")

        self.showText(x, y+20, age, "#FFFFFF")

        if round(arr[2][1].item()*100) > 80:
            self.showText(x, y+40, emotion, "#01DF01")
        elif round(arr[2][1].item()*100) > 50:
            self.showText(x, y+40, emotion, "#FFFF00")
        else:
            self.showText(x, y+40, emotion, "#FF0000")


    # ทำลาย component
    def DestroyComp(self):
        for obj in self.obj:
            obj.after(0, obj.destroy)
        self.obj = []
        self.imd_id = 0

    # หยุดการทำงาน
    def stopProcess(self):
        self.interrupt_process = True

    # แสดงผลก่อนหน้า
    def left(self):
        if self.imd_id - 36*2 >= 0:
            self.imd_id -= 36*2
        else:
            self.imd_id = 0
        self.interrupt_process = False
        if not self.btn_lock:
            Thread(target=self.PredictionProcess).start()

    # แสดงผลเพิ่มเติม
    def right(self):
        self.interrupt_process = False
        self.loadFilename()
        if not self.btn_lock:
            Thread(target=self.PredictionProcess).start()
        
    # แสดงแรกสุด
    def doubleLeft(self):
        self.imd_id = 0
        self.interrupt_process = False
        if not self.btn_lock:
            Thread(target=self.PredictionProcess).start()

    # แสดงผลเพิ่มเติมทั้งหมด
    def doubleRight(self):
        self.loadFilename()
        self.interrupt_process = False
        Thread(target=self.threadDoubleRight).start()

    def threadDoubleRight(self):
        self.interrupt_process = False
        while self.imd_id < len(self.filename):
            for obj in self.obj_img:
                obj.after(60000, obj.destroy)
            if not self.btn_lock:
                self.PredictionProcess()

    # บันทึกไฟล์
    def saveFile(self):
        global stat
        pathname = time.strftime("prediction/%Y/%m/%d/")
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        filename = time.strftime("%H-%M-%S")
        file_pred = open("{}{}.txt".format(pathname, filename), "w")
        for massage in stat:
            file_pred.write("%s\n" % massage)
        file_pred.close()
        stat = []

    # อ่านไฟล์รูปทั้งหมด
    def loadFilename(self):
        folders = Setting.separated_path
        self.filename = []
        for dirpath, dirnames, files in os.walk(folders):
            if not os.listdir(dirpath) :
                os.rmdir(dirpath)
            dirpath = dirpath.replace(Setting.output_path, "")
            for file_name in files:
                self.filename.append(dirpath+"/"+file_name)

    # แสดงรูปและสั่งทำงานการทำนาย
    def PredictionProcess(self):
        
        self.shift_y = 0
        for i in range(6):
            self.shift_x = 0
            for j in range(6):
                if not self.interrupt_process:
                    if self.imd_id < len(self.filename):
                        try:
                            self.showImg('{}'.format(
                                self.filename[self.imd_id]), x=10+self.shift_x, y=80+self.shift_y, size=128)
                            self.predictMassage(10+self.shift_x, 80 + self.shift_y)
                            des = str(self.filename[self.imd_id])
                            des = des.replace('\\\\', "/")
                            des = des.replace(Setting.separated_path, Setting.predicted_path)
                            des = des.split('/')[:-1]
                            source = self.filename[self.imd_id]
                            destination = ""
                            for x in des:
                                destination = destination+str(x) + "/" 
                            if not os.path.exists(destination):
                                os.makedirs(destination)
                            try:
                                dest = shutil.move(source, destination)  
                            except:
                                pass
                            
                            self.imd_id += 1
                            self.shift_x += 310
                        except:
                            pass
                    else:
                        break
                self.btn_lock = True
            self.shift_y += 160
        self.btn_lock = False
        if self.imd_id >= len(self.filename):
            self.saveFile()

    # แสดงข้อความ
    def showText(self, x, y, text, color):
        self.text = tk.Label(self.window, text=text, borderwidth=0,
                             bg='#000000', fg=color, font=Style.font(11))  # ตั้งค่าข้อความ
        self.obj.append(self.text)
        self.obj[len(self.obj)-1].place(x=x+150, y=y+50)  # กำหนดตำแหน่ง

    '''โหลดรูป'''
    def showImg(self, pic, x, y, size):
        # จัดรูปให้พร้อมใช้งาน
        self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.resize(cv2.cvtColor(
                cv2.imread(pic), cv2.COLOR_BGR2RGB), (size, size))))
        # ตั้งค่า widget
        self.widget = tk.Canvas(self.window, width=300,
                                height=150, bg="#000000")
        self.widget.place(x=x, y=y)  # กำหนดตำแหน่ง
        self.img = tk.Label(self.window, image=self.photo,
                            borderwidth=0)  # ตั้งค่ารูปภาพ
        self.obj_img.append(self.img)
        self.obj_img[len(self.obj_img)-1].image = self.photo  # ตั้งค่ารูปภาพ
        self.obj_img[len(self.obj_img)-1].place(x=x+10, y=y+10)  # กำหนดตำแหน่ง

    def showImgWithComm(self, pic, x, y, size, color, comm):
        self.load = PIL.Image.open(pic)  # โหลด
        self.load.thumbnail((size, size))  # ปรับขนาด
        self.render = PIL.ImageTk.PhotoImage(
            self.load)  # เปลี่ยนเป็น PhotoImage
        self.img = tk.Button(self.window, image=self.render,
                             bg=color, borderwidth=5, command=comm)  # ตั้งค่าปุ่ม
        self.obj.append(self.img)
        self.obj[len(self.obj)-1].image = self.render  # ตั้งค่ารูปภาพ
        self.obj[len(self.obj)-1].place(x=x, y=y)  # กำหนดตำแหน่ง

class Prediction:
    def __init__(self, window, image):
        self.window = window
        self.image = image

    def run(self):
        return self.torchPrediction()

    # ทำนาย
    def torchPrediction(self):
        input_image = PIL.Image.fromarray(self.image)

        preprocess1 = transforms.Compose([
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor1 = preprocess1(input_image)
        # create a mini-batch as expected by the model
        input_batch1 = input_tensor1.unsqueeze(0)

        preprocess2 = transforms.Compose([
            transforms.Resize(48),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor2 = preprocess2(input_image)
        # create a mini-batch as expected by the model
        input_batch2 = input_tensor2.unsqueeze(0)

        preprocess3 = transforms.Compose([
            transforms.Resize(48),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor3 = preprocess3(input_image)
        # create a mini-batch as expected by the model
        input_batch3 = input_tensor3.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch1 = input_batch1.to('cuda')
            input_batch2 = input_batch2.to('cuda')
            input_batch3 = input_batch3.to('cuda')

        with torch.no_grad():
            output1 = model_emotion(input_batch2)
            output2 = model_age(input_batch1)
            output3 = model_gender(input_batch3)

        answer = []
        list_sex = []
        list_age = []
        list_emotion = []
        for i in range(2):
            list_sex.append(
                [torch.nn.functional.softmax(output3[0], dim=0)[i], sex[i]])
        answer.append(['Sex : '+str(max(list_sex)[1]), max(list_sex)[0]])
        sum = 0
        for i in range(80):
            sum += (torch.nn.functional.softmax(
                output2[0], dim=0)[i].item()*int(age[i]))
        if max(list_sex)[1] =='male':
            answer.append(['Age : '+str(round(sum- 5)), sum])
        else:
            answer.append(['Age : '+str(round(sum- 5)), sum])
        for i in range(7):
            list_emotion.append(
                [torch.nn.functional.softmax(output1[0], dim=0)[i], emotion[i]])
        answer.append(
            ['Emotion : '+str(max(list_emotion)[1]), max(list_emotion)[0]])
        return answer

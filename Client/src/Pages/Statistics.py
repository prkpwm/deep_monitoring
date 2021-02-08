import math
import tkinter as tk
import tkinter.font as tkFont

import cv2
import matplotlib.pyplot as plt
import PIL.Image
import src.Components.Style as Style
import src.Pages.Face_Prediction as Face_Prediction
import src.Pages.People_Counting as People_Counting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame


class App:
    def __init__(self, window):
        self.window = window
        self.obj = []

    # run program
    def run(self):
        stat = Face_Prediction.dashboard
        arr_sex = [0]*2
        arr_emotion = [0]*7
        arr_age = [0]*81
        for i in range(len(stat)):
            for j in range(len(Face_Prediction.sex)):
                if stat[i][0].split(":")[1].replace(" ", "") == Face_Prediction.sex[j]:
                    arr_sex[j] += 1
                    break
            for j in range(len(Face_Prediction.age)):
                if stat[i][1].split(":")[1].replace(" ", "") == Face_Prediction.age[j]:
                    arr_age[j] += 1
                    break
            for j in range(len(Face_Prediction.emotion)):
                if stat[i][2].split(":")[1].replace(" ", "") == Face_Prediction.emotion[j]:
                    arr_emotion[j] += 1
                    break
    
        list_age = [8, 18, 35, 55, 100]
        ans_age = [0]*5
        k = 0
        for i in range(len(list_age)):
            for j in range(k, len(arr_age)):
                if j <= list_age[i]:
                    ans_age[i] += arr_age[j]
                else:
                    k = j
                    break

        age_sum = math.fsum(ans_age)
        if age_sum != 0:
            self.showImg('public/Wallpaper/dash.png', x=10,
                         y=60, width=1900, height=1000)
            for i in range(5):
                ans_age[i] = int(ans_age[i]/age_sum*100)

            digit_path = ['public/digit/0.png', 'public/digit/1.png', 'public/digit/2.png', 'public/digit/3.png', 'public/digit/4.png',
                          'public/digit/5.png', 'public/digit/6.png', 'public/digit/7.png', 'public/digit/8.png', 'public/digit/9.png']

            gender_sum = math.fsum(arr_sex)
            for i in range(2):
                arr_sex[i] = int(arr_sex[i]/gender_sum*100)

            precent_sum = math.fsum(arr_sex)
            while precent_sum < 100:
                arr_sex[0] += 1
                precent_sum += 1
            arr_sex.reverse()

            emotion_sum = math.fsum(arr_emotion)
            for i in range(7):
                arr_emotion[i] = int(arr_emotion[i]/emotion_sum*100)

            pos = [6, 5, 4, 0, 1, 3, 2]
            ans_emotion = []
            for i in range(7):
                for j in range(7):
                    if i == pos[j]:
                        ans_emotion.append(arr_emotion[j])
            arr_emotion = ans_emotion

            precent_sum = math.fsum(arr_emotion)
            while precent_sum < 100:
                arr_emotion[1] += 1
                precent_sum += 1
            shift_y = 0
            for i in range(5):
                if ans_age[i] != 0:
                    self.showImg('public/sign/barx.png', x=300,
                                 y=550+shift_y, width=ans_age[i]*6, height=50)
                shift_y += 102

            #arr_people = [[8,30],[9,15],[10,5],[11,45],[12,10],[13,60],[14,21],[15,90]]
            if len(People_Counting.stat) != 0:
                arr_people = People_Counting.stat
                shift_x = 0
                vshift_x = 0
                multi = 1
                while shift_x+vshift_x < 720:
                    vshift_x = len(arr_people)*multi
                    shift_x = vshift_x*len(arr_people) / \
                        2+vshift_x*len(arr_people)/5
                    multi += 0.1

                if shift_x+vshift_x > 900:
                    multi -= 0.2
                    vshift_x = len(arr_people)*multi

                shift_x = 0
                for i in range(len(arr_people)):
                    self.showImg('public/sign/bary.png', x=1100+shift_x, y=550 + (
                        435-arr_people[i][1]*4), width=int(vshift_x/2), height=arr_people[i][1]*4)
                    self.showText(x=1100+shift_x+vshift_x/4,
                                  y=1000, text=arr_people[i][0])
                    shift_x += vshift_x/2+vshift_x/5

                shift_x = 0
                total = math.fsum(x[1] for x in arr_people)
                for i in range(len(str(total))):
                    a = int(total % 10)
                    self.showImg(digit_path[a], x=320 +
                                 shift_x, y=200, width=80, height=80)
                    total = total/10
                    shift_x -= 70

            shift_x = 0
            for i in range(2):
                a = int(arr_sex[i]/10)
                b = int(arr_sex[i] % 10)
                if a != 0 and a!=10:
                    self.showImg(
                        digit_path[a], x=550+shift_x - 40, y=310, width=50, height=50)
                elif a==10:
                    self.showImg(
                        digit_path[0], x=550+shift_x - 40, y=310, width=50, height=50)
                    self.showImg(
                        digit_path[1], x=550+shift_x - 80, y=310, width=50, height=50)
                self.showImg(digit_path[b], x=550 +
                                shift_x, y=310, width=50, height=50)
                shift_x += 200

            shift_x = 0
            for i in range(7):
                a = int(arr_emotion[i]/10)
                b = int(arr_emotion[i] % 10)
        
                if a != 0 and a!=10:
                    self.showImg(
                        digit_path[a], x=960+shift_x - 30, y=330, width=30, height=30)
                elif a==10:
                    self.showImg(
                        digit_path[0], x=960+shift_x - 30, y=330, width=30, height=30)
                    self.showImg(
                        digit_path[1], x=960+shift_x - 60, y=330, width=30, height=30)
                self.showImg(digit_path[b], x=960 +
                             shift_x, y=330, width=30, height=30)
                shift_x += 140
        else:
            self.showImg('public/Wallpaper/dash2.PNG', x=10,
                         y=60, width=1900, height=1000)

    # หยุดการทำงานและทำลาย
    def DestroyComp(self):
        for obj in self.obj:
            obj.after(0, obj.destroy)
        self.obj = []
    # แสดงรูป

    def showImg(self, pic, x, y, width, height):
        self.render = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.resize(cv2.cvtColor(
                cv2.imread(pic), cv2.COLOR_BGR2RGB), (width, height))))
        self.obj.append(
            tk.Label(self.window, image=self.render, borderwidth=0))
        self.obj[len(self.obj)-1].image = self.render
        self.obj[len(self.obj)-1].place(x=x, y=y)
    # แสดงข้อความ

    def showText(self, x, y, text):
        self.text = tk.Label(self.window, text=text, borderwidth=0,
                             bg='#000000', fg='#ffffff', font=Style.font(11))
        self.obj.append(self.text)
        self.obj[len(self.obj)-1].place(x=x, y=y)

import tkinter as tk
import src.Components.Menu_Bar as Menu_Bar

import os

from threading import Thread
import time

if __name__ == '__main__':
    root = tk.Tk() #สร้าง windows
    root.attributes('-fullscreen', True) #ตั้งค่า fullscreen
    #root.wm_attributes('-transparentcolor','black')
    root.geometry('1920x1080') #ตั้งค่า ความละเอียด
    root.configure(background='#19191A') #ตั้งพื้นหลังสี
    root.tk.call('wm', 'iconphoto', root._w,
                 tk.PhotoImage(file='public/Logos/eye.png'))  #ตั้ง icon
    root.title("Deep Monitoring") #ตั้งชื่อโปรแกรม
    Menu_Bar.App(root) #เริ่มเมนูบาร์
    root.mainloop() #วนซ้ำ

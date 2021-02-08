import glob
import os
import pickle
import threading
import time
import tkinter
from threading import Thread
from timeit import default_timer as timer

import imutils
import numpy as np
import PIL.Image
import PIL.ImageTk
import src.Components.Live as Live
import src.Components.Style as Style
import src.Pages.Setting as Setting
import torch.backends.cudnn as cudnn
from imutils import face_utils
from PIL import Image, ImageFilter
from src.utils import google_utils
from src.utils.datasets import *
from src.utils.utils import *
from torchvision import transforms
import dlib
detector = dlib.get_frontal_face_detector()
''' LOAD MODEL '''
weights_yolov5 = './models/yolov5s.pt'

imd_id = 0
process_id = 0
run_id = 0
process_ = 0
merge_img = []
obj = []
pause_vid = False


class Detection:
    def __init__(self, window, width=1595, height=800, vdo=0, x=0, y=0, name="camara", confident=30, crop_time=1.5, tracking=False):
        self.window = window
        self.width = width
        self.height = height
        self.weights_yolov5 = weights_yolov5
        self.output_path = Setting.output_path
        self.x = x
        self.y = y
        self.vdo = vdo
        self.name = name
        self.imgsz = 416
        self.conf_thres = confident/100
        self.crop_time = crop_time
        self.iou_thres = confident/100
        self.itr_count = 0
        self.waiting = False
        self.waiting_time = 0.01
        self.bufffer = 0
        self.process_id = 0
        self.tracking = tracking
        self.vdo_type = ('.mov', '.avi', '.mp4', '.mpg',
                         '.mpeg', '.m4v', '.wmv', '.mkv')
        self.camera_type = ('rtsp', 'http')
        self.img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
        self.render_tool = True if self.vdo.startswith(
            'rtsp') else False  # check cctv

    # run program
    def run(self):
        self.destroy = False
        global pause_vid
        pause_vid = False
        global process_
        process_ += 1
        self.process_id = process_

        self.thread = threading.Thread(target=self.yolov5, daemon=True)
        self.thread.start()

        self.size_pack = [[1, 1595, 900, 0, 0, 1], [4, 798, 453, 798, 453, 2], [9, 533, 303, 533, 303, 3], [
            16, 397, 227, 397, 227, 4], [100, 317, 180, 317, 180, 5]]

        self.nums = 0
        self.nums_size = 0
        for i in range(len(self.size_pack)):  # set size
            if Live.num_dis <= self.size_pack[i][0]:
                self.nums = self.size_pack[i][0]
                self.nums_size = self.size_pack[i][5]
                break

        global merge_img
        merge_img = [[-1]]*self.nums

        self.wall = cv2.resize(cv2.cvtColor(cv2.imread(
            "public/Wallpaper/wall3.png"), cv2.COLOR_BGR2RGB), (self.width, self.height))  # draw wallpaper

        self.arr_point = []
        print("Thread Start")

    # หยุดการทำงาน vdo
    def stop(self):
        global process_
        process_ = 0
        self.destroy = True

    # ทำการ detect โดย yolov5
    def yolov5(self):
        agnostic_nms = True  # กรอบซ้อน
        augment = True  # ปรับแต่งภาพ
        classes = None  # จำนวนคลาส
        conf_thres = self.conf_thres  # ค่าความมั่นใจ
        # device='cpu'
        device = '0'  # อุปกรณ์ที่ใช้ในการ detection
        fourcc = 'mp4v'  # format video
        half = True  # function cuda
        imgsz = self.imgsz  # ขนาดรูปจากการ train
        # Intersection over Union (IOU) threshold to use when removing overlapping instances over target count; if None, then only use score to determine which instances to remove.
        iou_thres = self.iou_thres

        output = self.output_path  # path output
        source = self.vdo  # แหล่งที่มา
        view_img = True  # ดูภาพ

        weights = self.weights_yolov5  # โมเดล

        self.img_check = self.vdo.endswith(
            self.vdo_type) in self.img_formats  # img type check
        self.folder_check = 1 if os.path.isdir(
            self.vdo) else 0  # folder type check

        webcam = source in ['0', '1', '2'] or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')  # อ่านแหล่งที่มา

        # Initialize
        device = torch_utils.select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = torch.load(weights, map_location=device)[
            'model'].float()  # load to FP32
        model.to(device).eval()
        # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
        # model.fuse()
        imgsz = check_img_size(
            imgsz, s=model.model[-1].stride.max())  # check img_size
        if half:
            model.half()  # to FP16
        # Second-stage classifier

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            dataset = LoadImages(source, img_size=imgsz)
        # Get names and colors

        colors = (255, 255, 255)
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # run once
        _ = model(img.half() if half else img) if device.type != 'cpu' else None

        counting = 0

        t4 = torch_utils.time_synchronized()
        t6 = torch_utils.time_synchronized()
        line_len = 20
        bound = 10
        ck = 0
        try:  # unpack error prevent
            for path, img, im0s, video_des in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference

                pth = '{}'.format(self.output_path) + \
                    time.strftime("%Y/%m/%d/")+self.name+"/"
                if not os.path.exists(pth):
                    os.makedirs(pth)

                t1 = torch_utils.time_synchronized()
                # Process detections
                if self.destroy:
                    global pause_vid
                    pause_vid = True
                    break

                pred = model(img, augment=augment)[0]
                # Apply NMS
                pred = non_max_suppression(
                    pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

                t2 = torch_utils.time_synchronized()
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, self.im0 = path[i], '%g: ' % i, im0s[i].copy()
                    else:
                        p, s, self.im0 = path, '', im0s

                    position_detected = []
                    position_detected_2 = []
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :16] = scale_coords(
                            img.shape[2:], det[:, :16], self.im0.shape).round()
                        # Write results
                        for *xyxy, conf, cls in det:
                            #plot_one_box(xyxy, im0, color=colors, line_thickness=2)

                            x = int(xyxy[0].item())
                            y = int(xyxy[1].item())
                            w = int(xyxy[2].item()-x)
                            h = int(xyxy[3].item()-y)

                            # ตำแหน่งการวาดกรอบ
                            indx = [[[x-bound, y-bound], [x+line_len-bound, y-bound]], [[x-bound, y-bound], [x-bound, y+line_len-bound]], [[x+w+bound, y+h+bound], [x+w-line_len+bound, y+h+bound]], [[x+w+bound, y+h+bound], [x+w+bound, y+h-line_len+bound]],
                                    [[x+w+bound, y-bound], [x+w-line_len+bound, y-bound]], [[x+w+bound, y-bound], [x+w+bound, y+line_len-bound]], [[x-bound, y+h+bound], [x-bound, y+h-line_len+bound]], [[x-bound, y+h+bound], [x+line_len-bound, y+h+bound]]]

                            # เก็บตำแหน่งกรอบ
                            position_detected.append(indx)
                            position_detected_2.append([x, y, w, h])

                        # print('\n')
                    #print('(%.3fs)' % (t2 - t1))
                    if self.tracking:
                        self.autoPop()
                        for idx in position_detected_2:
                            if self.centroy(idx):
                                box_ratio = abs(idx[2]-idx[3])
                                if box_ratio < idx[2]*0.5 and box_ratio < idx[3]*0.5:
                                        cv2.imwrite(pth + time.strftime("%H-%M-%S")+"_"+str(idx[0])+"-"+str(idx[1])+".jpg",
                                                    self.im0[idx[1]:idx[1]+idx[3], idx[0]:idx[0]+idx[2]])
                                        self.showDetected(
                                            self.im0[idx[1]:idx[1]+idx[3], idx[0]:idx[0]+idx[2]])
                    else:
                        # บันทึกและแสดงรูปจากการ detect
                        if t1 - t6 > self.crop_time or self.img_check or self.folder_check and self.crop_time != 0:
                            for idx in position_detected_2:
                                box_ratio = abs(idx[2]-idx[3])
                                if box_ratio < idx[2]*0.5 and box_ratio < idx[3]*0.5:
                                        cv2.imwrite(pth + time.strftime("%H-%M-%S")+"_"+str(idx[0])+"-"+str(idx[1])+".jpg",
                                                    self.im0[idx[1]:idx[1]+idx[3], idx[0]:idx[0]+idx[2]])
                                        self.showDetected(
                                            self.im0[idx[1]:idx[1]+idx[3], idx[0]:idx[0]+idx[2]])
                            t6 = torch_utils.time_synchronized()

                    # วาดกรอบทุกตำแหน่งที่ได้มา
                    for indx in position_detected:
                        for i in range(len(indx)):
                            self.im0 = cv2.line(
                                self.im0, (indx[i][0][0], indx[i][0][1]), (indx[i][1][0], indx[i][1][1]), (255, 255, 255), 3)
                    try:  # decode error prevent
                        if os.name == 'nt':  # os checking  # windows?
                            self.showImgV1() if self.render_tool else self.showImg()
                        else:
                            self.mergeImg() if self.render_tool else self.showImg()
                    except:
                        pass
        except:
            self.yolov5()  # auto reconect

    # แสดงผลส่วนที่ได้จากการ detect

    def showDetected(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ปรับสีเป็น BGR2RGB
        image = cv2.resize(img, (128, 128))  # ปรับขนาดภาพ
        global imd_id
        img_pos = [[1630, 80], [1775, 80], [1630, 220], [1775, 220], [1630, 360], [1775, 360],
                   [1630, 500], [1775, 500], [1630, 640], [1775, 640], [
                       1630, 780], [1775, 780], [1630, 920],
                   [1775, 920]]  # ตำแหน่งการวางภาพจาก detect
        if imd_id < 14:
            self.imgDetected(
                image, x=img_pos[imd_id % 14][0], y=img_pos[imd_id % 14][1], size=256)  # แสดงภาพ
        else:
            global obj
            obj[(imd_id % 14)].destroy()
            #self.obj.pop((imd_id % 14)+1)
            self.imgDetected(
                image, x=img_pos[imd_id % 14][0], y=img_pos[imd_id % 14][1], size=256)  # แสดงภาพ

        imd_id += 1  # เพิ่มจำนวนรูปที่ได้รับมา

    def imgDetected(self, pic, x, y, size):  # show output of detection
        self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(pic))
        global obj
        obj.append(tkinter.Label(self.window, image=self.photo,
                                 borderwidth=0, bg='#19191A'))
        obj[len(obj)-1].image = self.photo
        obj[len(obj)-1].place(x=x, y=y)

    '''แสดงรูป'''

    def showImgV1(self):  # for vdo
        img = cv2.cvtColor(self.im0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height))
        photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(img))
        img = tkinter.Label(
            self.window, image=photo, borderwidth=0)
        img.image = photo
        time.sleep(self.waiting_time)
        img.place(x=self.x, y=self.y)
        img.after(3000, img.destroy)

    def showImg(self):  # for image
        img0 = cv2.cvtColor(self.im0, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img0, (self.width, self.height))
        img2 = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(img1))
        img3 = tkinter.Label(
            self.window, borderwidth=0, image=img2)
        img3.image = img2
        img3.place(x=self.x, y=self.y)
        img3.after(1000, img3.destroy)

    def mergeImgCondition(self):  # all image is send?
        global merge_img
        for i in range(self.nums):
            if len(merge_img[i]) == 1:
                return False
        return True

    def mergeImg(self):  # merge img to one (decrease refresh rate)
        img = cv2.resize(cv2.cvtColor(
            self.im0, cv2.COLOR_BGR2RGB), (self.width, self.height))
        global merge_img
        for i in range(self.nums):
            if len(merge_img[i]) == 1:
                if i < len(Live.entry_list):
                    merge_img[self.process_id - 1] = img
                else:
                    merge_img[i] = self.wall
        if self.mergeImgCondition():
            if self.process_id == 1:
                k = 0
                for h in range(self.nums_size):
                    vis = merge_img[k]
                    for i in range(1, self.nums_size):
                        vis = np.concatenate((vis, merge_img[k+1]), axis=1)
                        k += 1
                    if h == 0:
                        self.vis2 = vis
                    else:
                        self.vis2 = np.concatenate((self.vis2, vis), axis=0)
                    k += 1
                self.photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(self.vis2))
                self.img = tkinter.Label(
                    self.window, image=self.photo, borderwidth=0)
                self.img.image = self.photo
                time.sleep(0.04)
                self.img.place(x=10, y=80)
                self.img.after(3000, self.img.destroy)
                for i in range(len(Live.entry_list)):
                    merge_img[i] = [-1]

    def nearest_point(self, x, y):
        min_point = 1000000
        min_position = 0
        for i in range(len(self.arr_point)):
            if pow(pow(self.arr_point[i][0]-x, 2)+pow(self.arr_point[i][1]-y, 2), 0.5) < min_point:
                min_point = pow(
                    pow(self.arr_point[i][0]-x, 2)+pow(self.arr_point[i][1]-y, 2), 0.5)
                min_position = i
        return min_point, min_position

    def centroy(self, idx):
        if len(self.arr_point) == 0:
            if self.vertifyFace(self.im0[idx[1]:idx[1]+idx[3], idx[0]:idx[0]+idx[2]]):
                self.arr_point.append([(idx[0]+idx[0]+idx[2]) / 2, (idx[1]+idx[1]+idx[3]) / 2, time.time()])
                return True
        else:
            min_point, min_position = self.nearest_point((idx[0]+idx[0]+idx[2]) / 2, (idx[1]+idx[1]+idx[3]) / 2)
            if min_point < idx[2]:
                self.arr_point[min_position] = [
                    (idx[0]+idx[0]+idx[2]) / 2, (idx[1]+idx[1]+idx[3]) / 2, time.time()]
            else:
                if self.vertifyFace(self.im0[idx[1]:idx[1]+idx[3], idx[0]:idx[0]+idx[2]]):
                    self.arr_point.append([(idx[0]+idx[0]+idx[2]) / 2, (idx[1]+idx[1]+idx[3]) / 2, time.time()])
                    return True
        return False

    def autoPop(self):
        try:
            for i in range(len(self.arr_point)):
                if time.time()-self.arr_point[i][2] > self.crop_time:
                    self.arr_point.pop(i)
                    break
        except:
            self.autoPop()

    def vertifyFace(self, img):
        rects = detector(img, 1)
        if len(rects) != 0:
            for (i, rect) in enumerate(rects):
                coords = (x, y, w, h) = face_utils.rect_to_bb(rect)
                if len(coords) == 4:
                    return True
        return False

import os
import tkinter
import tkinter.font as tkFont

import imutils
import numpy as np
import PIL.Image
import PIL.ImageTk
import src.Components.Style as Style
import torch.backends.cudnn as cudnn
from imutils import face_utils
from PIL import Image, ImageFilter
from src.Pages.Face_Detection import Detection
from src.utils import google_utils
from src.utils.datasets import *
from src.utils.utils import *
from torchvision import transforms


class Heat_Map(Detection):
    def __init__(self, window, width=1595, height=800, vdo=0, x=0, y=0, save_t=3600, name="cctv",confident=30,crop_time=1.5,tracking=False):
        super().__init__(window, width=width, height=height, vdo=vdo, x=x, y=y, name=name,confident=confident,crop_time=crop_time,tracking=tracking)
        self.pos = self.martrix_position()
        self.save_t = save_t

    # กำหนดค่าเริ่มต้น
    def martrix_position(self):
        pos = []
        for i in range(1920):
            for j in range(1080):
                if i % 20 == 0 and j % 20 == 0:
                    pos.append([i, j, 0])
                j += 20
            i += 20
        return pos

    # ทำการ detect โดย yolov5
    def yolov5(self):
        agnostic_nms = True  # กรอบซ้อน
        augment = False  # ปรับแต่งภาพ
        classes = None  # จำนวนคลาส
        conf_thres = self.conf_thres  # ค่าความมั่นใจ
        # device='cpu'
        device = '0'  # อุปกรณ์
        fourcc = 'mp4v'  # format video
        half = True  # function cuda
        imgsz = self.imgsz  # ขนาดรูปจากการ train
        iou_thres = self.iou_thres  #
        output = self.output_path  # path output
        source = self.vdo  # แหล่งที่มา
        view_img = True  # ดูภาพ
        weights = self.weights_yolov5  # โมเดล
        self.waiting_time = self.waiting_time+0.1

        self.img_check = self.vdo.endswith(self.vdo_type) in self.img_formats 
        self.folder_check = 1 if os.path.isdir(self.vdo) else 0

        webcam = source in ['0','1','2'] or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')  # อ่านแหล่งที่มา

        # Initialize
        device = torch_utils.select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)[
            'model'].float()  # load to FP32

        model.to(device).eval()

        # torch.save(torch.load(weights, map_loca         tion=device), weights)  # update model if SourceChangeWarning
        # model.fuse()

        imgsz = check_img_size(
            imgsz, s=model.model[-1].stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        save_img = False
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = False
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = (255, 255, 255)

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # run once
        _ = model(img.half() if half else img) if device.type != 'cpu' else None

        counting = 0
        t3 = torch_utils.time_synchronized()
        t4 = torch_utils.time_synchronized()
        t5 = torch_utils.time_synchronized()
        t6 = torch_utils.time_synchronized()
        xtemp = 0
        ytemp = 0
        line_len = 20
        bound = 10
        try: #unpack error prevent
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
                        # Rescale boxes from img_size to self.im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], self.im0.shape).round()
                        # Write results
                        for *xyxy, conf, cls in det:
                            #plot_one_box(xyxy, self.im0, color=colors, line_thickness=2)
                            x = int(xyxy[0].item())
                            y = int(xyxy[1].item())
                            w = int(xyxy[2].item()-x)
                            h = int(xyxy[3].item()-y)
                            # Rectangle coordinates

                            # ตำแหน่งการวาดกรอบ
                            indx = [[[x-bound, y-bound], [x+line_len-bound, y-bound]], [[x-bound, y-bound], [x-bound, y+line_len-bound]], [[x+w+bound, y+h+bound], [x+w-line_len+bound, y+h+bound]], [[x+w+bound, y+h+bound], [x+w+bound, y+h-line_len+bound]],
                                    [[x+w+bound, y-bound], [x+w-line_len+bound, y-bound]], [[x+w+bound, y-bound], [x+w+bound, y+line_len-bound]], [[x-bound, y+h+bound], [x-bound, y+h-line_len+bound]], [[x-bound, y+h+bound], [x+line_len-bound, y+h+bound]]]
                            # เก็บตำแหน่งกรอบ
                            position_detected.append(indx)
                            position_detected_2.append([x,y,w,h])

                            # เก็บตำแหน่ง
                            for i in range(len(self.pos)):
                                if (x+x+w)/2 - 10 < self.pos[i][0] and (y+y+h)/2 - 10 < self.pos[i][1]:
                                    if self.pos[i][2] < 601:
                                        self.pos[i][2] += 1
                                    break

                        # print('\n')
                    #print('(%.3fs)' % (t2 - t1))
                    # บันทึกภาพและรีเซต heat map
                    if t1-t5 > self.save_t:
                        if not os.path.exists("Heat Map"):
                            os.makedirs("Heat Map")
                        try: #decode error prevent
                            cv2.imwrite(
                                'Heat Map/frame-' + time.strftime("%d-%m-%Y-%H-%M-%S")+"_on_"+self.name + ".jpg", self.im0)
                            f = open('Heat Map/frame-' + time.strftime("%d-%m-%Y-%H-%M-%S")+"_on_"+self.name+".txt", "w")
                            f.write(str(self.pos))
                            f.close()
                        except:
                            pass
                        self.pos = self.martrix_position()
                        t5 = torch_utils.time_synchronized()
                    # บันทึกและแสดงรูปจากการ detect
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
                        # บันทึกและแสดงรูปจากการ detect

                    # แสดงที่ผ่านการ detect แล้ว
                    self.im0 = self.heat_map(self.im0)
                    try:#decode error prevent
                        if os.name == 'nt': # windows?
                            self.showImgV1() if self.render_tool else self.showImg()
                        else:
                            self.mergeImg() if self.render_tool else self.showImg()
                    except:
                        pass
        except:
            self.yolov5()  #auto reconect

    # แสดงค่าสีจากจำนวนครั้งที่นับได้
    def heat_map(self, img):
        overlay = img.copy()
        opacity = 0.5
        for i in range(len(self.pos)):
            if self.pos[i][2] > 0:
                if(self.pos[i][2] < 400):
                    cv2.circle(
                        overlay, (self.pos[i][0], self.pos[i][1]), 10, (0, 255, 0), -1)
                elif(self.pos[i][2] < 1000):
                    cv2.circle(
                        overlay, (self.pos[i][0], self.pos[i][1]), 10, (0, 255, 255), -1)
                else:
                    cv2.circle(
                        overlay, (self.pos[i][0], self.pos[i][1]), 10, (0, 0, 255), -1)
        img = cv2.addWeighted(
            overlay, opacity, img, 1 - opacity, 0, img)
        return img

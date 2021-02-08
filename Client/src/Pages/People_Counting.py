import os
import tkinter
import tkinter.font as tkFont
from threading import Thread

import imutils
import numpy as np
import PIL.Image
import PIL.ImageTk
import src.Components.Style as Style
import torch.backends.cudnn as cudnn
from imutils import face_utils
from src.Pages.Face_Detection import Detection
from src.utils import google_utils
from src.utils.datasets import *
from src.utils.utils import *

stat = []


class PeopleCount(Detection):
    def __init__(self, window, width=1595, height=800, vdo=0, x=0, y=0, coord=[], name="cctv", start_t=8, end_t=24, direction=1,confident=30,crop_time=1.5,tracking=False):
        super().__init__(window, width=width, height=height, vdo=vdo, x=x, y=y, name=name,confident=confident,crop_time=crop_time,tracking=tracking)
        self.coord = coord
        self.axis = 1 if coord[0] == 0 else 0
        self.direction = int(direction)
        self.start_t = start_t
        self.end_t = end_t

    # ทำการ detect โดย yolov5

    def yolov5(self):
        agnostic_nms = True  # กรอบซ้อน
        augment = False  # ปรับแต่งภาพ
        classes = None  # จำนวนคลาส
        conf_thres = self.conf_thres  # self.conf_thres # ค่าความมั่นใจ
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

        webcam = source in ['0','1','2'] or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')  # อ่านแหล่งที่มา

        self.img_check = self.vdo.endswith(self.vdo_type) in self.img_formats
        self.folder_check = 1 if os.path.isdir(self.vdo) else 0

        # Initialize
        device = torch_utils.select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        google_utils.attempt_download(weights)
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
        in_count = 0
        out_count = 0
        pos_lock = []
        epos_lock = []
        t3 = torch_utils.time_synchronized()
        t4 = torch_utils.time_synchronized()
        t5 = torch_utils.time_synchronized()
        t6 = torch_utils.time_synchronized()
        xtemp = 0
        ytemp = 0
        line_len = 20
        bound = 10
        last_pos_x = 0
        last_pos_y = 0
        last = self.start_t - 1
        frt = True
        dynamics_size = 50
        try:  # unpack error prevent
            for path, img, im0s, video_des in dataset:
                if frt: #set line position
                    self.coord[0] = int(self.coord[0]*video_des[0]/100)
                    self.coord[1] = int(self.coord[1]*video_des[1]/100)
                    self.coord[2] = int(self.coord[2]*video_des[0]/100)
                    self.coord[3] = int(self.coord[3]*video_des[1]/100)
                    frt = False

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
                    # normalization gain whwh

                    bound_detected = []
                    position_detected = []
                    position_detected_2 = []
                    self.imsize = self.im0.shape
                    if det is not None and len(det):
                        # Rescale boxes from img_size to self.im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], self.im0.shape).round()
                        # Write results
                        for *xyxy, conf, cls in det:
                            # plot_one_box(xyxy, self.im0, color=colors, line_thickness=2)
                            x = int(xyxy[0].item())
                            y = int(xyxy[1].item())
                            w = int(xyxy[2].item()-x)
                            h = int(xyxy[3].item()-y)
                            # Rectangle coordinates
                            # รูปจากการ detect
                            dynamics_size = w
                            # ตำแหน่งการวาดกรอบ
                            indx = [[[x-bound, y-bound], [x+line_len-bound, y-bound]], [[x-bound, y-bound], [x-bound, y+line_len-bound]], [[x+w+bound, y+h+bound], [x+w-line_len+bound, y+h+bound]], [[x+w+bound, y+h+bound], [x+w+bound, y+h-line_len+bound]],
                                    [[x+w+bound, y-bound], [x+w-line_len+bound, y-bound]], [[x+w+bound, y-bound], [x+w+bound, y+line_len-bound]], [[x-bound, y+h+bound], [x-bound, y+h-line_len+bound]], [[x-bound, y+h+bound], [x+line_len-bound, y+h+bound]]]
                            # เก็บตำแหน่งกรอบ
                            bound_detected.append(indx)
                            position_detected.append([x, y, w, h])
                            position_detected_2.append([x, y, w, h])

                        # print('\n')
                    #print('(%.3fs)' % (t2 - t1))
                    # Stream results
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
                    for indx in bound_detected:
                        for i in range(len(indx)):
                            self.im0 = cv2.line(
                                self.im0, (indx[i][0][0], indx[i][0][1]), (indx[i][1][0], indx[i][1][1]), (255, 255, 255), 3)

                    # people counting
                    # deadlock
                    for pos in position_detected:
                        distance = dynamics_size
                        cen_x = (pos[0]+pos[0]+pos[2])/2
                        cen_y = (pos[1]+pos[1]+pos[3])/2
                        cal_x = abs(cen_x-self.coord[0])
                        cal_y = abs(cen_y-self.coord[1])

                        time_syn = torch_utils.time_synchronized()
                        distance /= 2

                        if cal_y < distance*2 and cal_y > distance and self.direction % 2 == 1: # check position in out-line (axis x)
       
                            if len(epos_lock) == 0:
                                epos_lock.append([cen_x, cen_y, time_syn])
                            else:
                                for i in range(len(epos_lock)):
                                    if abs(cen_x - epos_lock[i][0]) < distance:
                                        epos_lock[i] = [
                                            cen_x, cen_y, time_syn]
                                if self.checkDistanceInLine(cen_x, epos_lock, distance, 1):
                                    epos_lock.append([cen_x, cen_y, time_syn])
                        elif cal_x < distance*2 and cal_x > distance: # check position in out-line (axis y)
              
                            if len(epos_lock) == 0:
                                epos_lock.append([cen_x, cen_y, time_syn])
                            else:
                                for i in range(len(epos_lock)):
                                    if abs(cen_y - epos_lock[i][1]) < distance:
                                        epos_lock[i] = [
                                            cen_x, cen_y, time_syn]
                                if self.checkDistanceInLine(cen_y, epos_lock, distance, 0):
                                    epos_lock.append([cen_x, cen_y, time_syn])

                        if cal_y < distance and self.direction % 2 == 1: # check position in in-line (axis x)
                         
                            if len(pos_lock) == 0:
                                pos_lock.append([cen_x, cen_y, time_syn])
                                if self.checkDistanceOutLine(cen_x, epos_lock, distance, 0)-cen_y < 0: # check
                                    if self.direction % 2 == 1:
                                        in_count += 1
                                    else:
                                        out_count += 1
                                else:
                                    if self.direction % 2 == 1:
                                        out_count += 1
                                    else:
                                        in_count += 1
                                counting += 1
                            else:
                                for i in range(len(pos_lock)):
                                    if abs(cen_x - pos_lock[i][0]) < distance:
                                        if time_syn-pos_lock[i][2] < 1:
                                            pos_lock[i] = [
                                                cen_x, cen_y, time_syn]
                                        else:
                                            pos_lock[i] = [
                                                cen_x, cen_y, time_syn]
                                            if self.checkDistanceOutLine(cen_x, epos_lock, distance, 0)-cen_y < 0:
                                                if self.direction % 2 == 1:
                                                    in_count += 1
                                                else:
                                                    out_count += 1
                                            else:
                                                if self.direction % 2 == 1:
                                                    out_count += 1
                                                else:
                                                    in_count += 1
                                            counting += 1
                                if self.checkDistanceInLine(cen_x, pos_lock, distance, 1):
                                    pos_lock.append([cen_x, cen_y, time_syn])
                                    if self.checkDistanceOutLine(cen_x, epos_lock, distance, 0)-cen_y < 0:
                                        if self.direction % 2 == 1:
                                            in_count += 1
                                        else:
                                            out_count += 1
                                    else:
                                        if self.direction % 2 == 1:
                                            out_count += 1
                                        else:
                                            in_count += 1
                                    counting += 1
                        elif cal_x < distance: # check position in in-line (axis y)
                        
                            if len(pos_lock) == 0:
                                pos_lock.append([cen_x, cen_y, time_syn])
                                if self.checkDistanceOutLine(cen_y, epos_lock, distance, 1)-cen_x < 0:
                                    if self.direction % 2 == 1:
                                        in_count += 1
                                    else:
                                        out_count += 1
                                else:
                                    if self.direction % 2 == 1:
                                        out_count += 1
                                    else:
                                        in_count += 1
                                counting += 1
                            else:
                                for i in range(len(pos_lock)):
                                    if abs(cen_x - pos_lock[i][0]) < distance:
                                        if time_syn-pos_lock[i][2] < 1:
                                            pos_lock[i] = [
                                                cen_x, cen_y, time_syn]
                                        else:
                                            pos_lock[i] = [
                                                cen_x, cen_y, time_syn]
                                            if self.checkDistanceOutLine(cen_y, epos_lock, distance, 1)-cen_x < 0:
                                                if self.direction % 2 == 1:
                                                    in_count += 1
                                                else:
                                                    out_count += 1
                                            else:
                                                if self.direction % 2 == 1:
                                                    out_count += 1
                                                else:
                                                    in_count += 1
                                            counting += 1
                                if self.checkDistanceInLine(cen_y, pos_lock, distance, 0):
                                    pos_lock.append([cen_x, cen_y, time_syn])

                                    if self.checkDistanceOutLine(cen_y, epos_lock, distance, 1)-cen_x < 0:
                                        if self.direction % 2 == 1:
                                            in_count += 1
                                        else:
                                            out_count += 1
                                    else:
                                        if self.direction % 2 == 1:
                                            out_count += 1
                                        else:
                                            in_count += 1
                                    counting += 1

                        pos_lock = [pos_lock[i] for i in range(
                            len(pos_lock)) if time_syn-pos_lock[i][2] <= 1]
                        epos_lock = [epos_lock[i] for i in range(
                            len(epos_lock)) if time_syn-epos_lock[i][2] <= 1 ]
                            
                    #print(counting,in_count,out_count)
                    # แสดงจำนวนคนที่นับได้
                    self.im0 = cv2.putText(self.im0, "IN:" + str(
                        in_count), (int(video_des[0]*0.8), int(video_des[1]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    self.im0 = cv2.putText(self.im0, "OUT:" + str(
                        out_count), (int(video_des[0]*0.8), int(video_des[1]*0.2)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)

                    current_time = int(time.strftime("%H"))
                    if current_time > self.start_t and current_time < self.end_t:
                        if current_time - last >= 1:
                            global stat
                            stat.append([current_time, in_count])
                            last = current_time
                            counting = 0
                            in_count = 0
                            out_count = 0

                    # วาดเส้น
                    self.im0 = cv2.line(
                        self.im0, (self.coord[0], self.coord[1]), (self.coord[2], self.coord[3]), (255, 255, 255), 3)
                    try:  # decode error prevent
                        if os.name == 'nt': # os checking  #windows?
                            self.showImgV1() if self.render_tool else self.showImg()
                        else:
                            self.mergeImg() if self.render_tool else self.showImg()
                    except:
                        pass
                    t5 = torch_utils.time_synchronized()
        except:
            self.yolov5()  #auto reconect
    # ตรวจสอบระยะในเส้น

    def checkDistanceInLine(self, pos, pos_lock, distance, axis):
        for i in range(len(pos_lock)):
            if abs(pos - pos_lock[i][0 if axis == 1 else 1]) < distance:
                return False #in area
        return True #out area

    # ตรวจสอบระยะก่อนเส้น
    def checkDistanceOutLine(self, pos, epos_lock, distance, axis):
        try:
            MIN = 9999
            position_MIN = 0
            for i in range(len(epos_lock)):
                if abs(pos - epos_lock[i][axis]) < MIN:
                    MIN = abs(pos - epos_lock[i][axis])
                    position_MIN = i
            res = epos_lock[position_MIN][0 if axis == 1 else 1]
            return res #nearest position
        except:
            pass
        return 9999

import os
import cv2
import math
import torch
import random
import argparse
import numpy as np

from Cascade_LD import lane_detect

K = 100


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=int, default=0, help='violation vehicle detection methods')
    args = parser.parse_args()

    # # run yolov5 algorithm
    # os.system(
    #     'python .\yolov5_sort\detect.py --source .\data --weights yolov5s.pt --conf 0.35'
    # )
    #
    # # run sort algorithm
    # os.system('python .\yolov5_sort\sort.py')

    # read the result of yolov5 & sort
    data = np.loadtxt('output/demo.txt', delimiter=',')

    # load data by opencv
    vc = cv2.VideoCapture('data/demo.mp4')

    # get the frame width and height
    frame_width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(frame_width)

    pos = 0
    frame_id = 2

    track = []
    for i in range(10000):
        track.append([])
    i = 0

    while (vc.isOpened()):
        ret, frame = vc.read()
        # print(frame)
        if frame is None:
            break

        # method 0: 车辆压线检测算法
        if args.method == 0:
            frame = cv2.resize(frame, (640, 360))
            frame_tensor = torch.FloatTensor(frame)
            frame_tensor = frame_tensor.permute((2, 0, 1))
            # print(frame_tensor.shape)
            # print(frame_tensor.type())
            lane_map = lane_detect(frame_tensor)
            # print(lane_map.shape)
            while (data[pos][0] == frame_id):
                x1 = int((data[pos][2] - 0.5 * data[pos][4]) * 640)
                y1 = int((data[pos][3] - 0.5 * data[pos][5]) * 320)
                x2 = int((data[pos][2] + 0.5 * data[pos][4]) * 640)
                y2 = int((data[pos][3] + 0.5 * data[pos][5]) * 320)
                cnt = 0
                # print(y2)
                for xx in range(x1, x2):
                    for yy in range(y1, y2):
                        if xx >= 640 or yy >= 360:
                            continue
                        if lane_map[yy][xx] == 1:
                            cnt += 1
                if cnt > 5:
                    plot_one_box([x1, y1, x2, y2], frame, label="id:" + str(data[pos][1]) + ", status:warning")
                else:
                    plot_one_box([x1, y1, x2, y2], frame, label="id:" + str(data[pos][1]) + ", status:ok")
                pos += 1
            frame_id += 1

        # method 1: 车辆超速检测算法
        if args.method == 1:
            while (data[pos][0] == frame_id):
                id = int(data[pos][1] % 1000)
                v = 0
                x = data[pos][2]
                y = data[pos][3]
                if len(track[id]) > 0:
                    # print(track[id][len(track[id] - 1)])
                    last_frame = track[id][len(track[id]) - 1][2]
                    if (frame_id - last_frame >= 10):
                        track[id] = []
                track[id].append((x, y, frame_id))
                if (len(track[id]) >= 10):
                    # print(track[id][len(track[id]) - 10])
                    delta_x = x - track[id][len(track[id]) - 10][0]
                    delta_y = y - track[id][len(track[id]) - 10][1]
                    dist = K * math.sqrt(delta_x * delta_x + delta_y * delta_y)
                    v = dist / (10 / 30)
                    x1 = (data[pos][2] - 0.5 * data[pos][4]) * frame_width
                    y1 = (data[pos][3] - 0.5 * data[pos][5]) * frame_height
                    x2 = (data[pos][2] + 0.5 * data[pos][4]) * frame_width
                    y2 = (data[pos][3] + 0.5 * data[pos][5]) * frame_height
                    if v > 50:
                        plot_one_box([x1, y1, x2, y2], frame, label="id:" + str(data[pos][1]) + ", v:" +
                                                                    str('' if v == 0 else round(v,
                                                                                                2)) + ", status:overspeed")
                    else:
                        plot_one_box([x1, y1, x2, y2], frame, label="id:" + str(data[pos][1]) + ", v:" +
                                                                    str('' if v == 0 else round(v, 2)) + ", status:ok")
                pos += 1
            frame_id += 1

        cv2.imshow('result', frame)
        # q键退出
        if (cv2.waitKey(30) & 0xff == ord('q')):
            break
    vc.release()
    cv2.destroyAllWindows()

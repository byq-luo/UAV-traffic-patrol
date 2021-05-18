import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_lane_num(pos, lane_det):
    tmp_list = []
    face_to = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    for step in face_to:
        new_pos = np.array(pos)
        while new_pos[0] >= 0 and new_pos[0] < lane_det.shape[0] and new_pos[1] >= 0 and new_pos[1] < \
                lane_det.shape[1]:
            val = lane_det[new_pos[0]][new_pos[1]]
            if val != 0:
                if tmp_list == [] or tmp_list[0] != val:
                    tmp_list.append(val)
                break
            else:
                new_pos += step
        if len(tmp_list) == 2:
            break
    if len(tmp_list) != 2:
        return -1;
    return int((tmp_list[0] + tmp_list[1]) / 2)


def check_reverse(pos, lane_det, v, avg_v_list):
    '''
    :param pos: 当前车辆中心位置
    :param lane_det: 车道线检测结果
    :param v: 当前车辆速度
    :param avg_v_list: 每条车道历史平均速度方向
    :return: 是否逆行，新的平均车速方向
    '''
    lane_det = lane_det
    # print(lane_det.shape)
    lane_type = get_lane_num(pos, lane_det)
    if lane_type == -1:
        return False, avg_v_list
    if lane_type not in avg_v_list:
        avg_v_list[lane_type] = np.array([0, 0])
    avg_v_list[lane_type] += v.astype('int32')
    if v.dot(avg_v_list[lane_type]) < 0:
        return True, avg_v_list
    else:
        return False, avg_v_list


if __name__ == '__main__':
    print(np.array([0, 1]) + np.array([1, 0]))
    a = np.array([[1, 1], [1, 1]])
    print(a[0][0])
    print(np.array([1, 1]) * np.array([2, 3]))

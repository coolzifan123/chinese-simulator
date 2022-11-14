from Renderer.stroke_gen import draw
import numpy as np
import cv2
import random
# 啊10,哎8，凹5，白5，比4，wu20

def rgb_trans(split_num, break_values):
    slice_per_split = split_num // 8
    break_values_head, break_values_tail = break_values[:-1], break_values[1:]

    results = []

    for split_i in range(8):
        break_value_head = break_values_head[split_i]
        break_value_tail = break_values_tail[split_i]

        slice_gap = float(break_value_tail - break_value_head) / float(slice_per_split)
        for slice_i in range(slice_per_split):
            slice_val = break_value_head + slice_gap * slice_i
            slice_val = int(round(slice_val))
            results.append(slice_val)

    return results

def get_colors(color_num):
    split_num = (color_num // 8 + 1) * 8

    r_break_values = [0, 0, 0, 0, 128, 255, 255, 255, 128]
    g_break_values = [0, 0, 128, 255, 255, 255, 128, 0, 0]
    b_break_values = [128, 255, 255, 255, 128, 0, 0, 0, 0]

    r_rst_list = rgb_trans(split_num, r_break_values)
    g_rst_list = rgb_trans(split_num, g_break_values)
    b_rst_list = rgb_trans(split_num, b_break_values)

    assert len(r_rst_list) == len(g_rst_list)
    assert len(b_rst_list) == len(g_rst_list)

    rgb_color_list = [(r_rst_list[i], g_rst_list[i], b_rst_list[i]) for i in range(len(r_rst_list))]
    return rgb_color_list

stroke_list = [10,8,5,5,4,17]
file_name_list = [1,5,28,55,134,6762]

# filepath_list = [f'home/liuyt/FontRL/data/font/2/GB2_{i+1}M.txt' for i in range(7)]

def draw_stroke(filepath,img,color,shape):
    stroke = np.loadtxt(filepath,dtype=np.int)
    col, row = stroke.shape
    ptStart = (stroke[0,0],stroke[0,1])
    for i in range(col-1):
        x,y = stroke[i+1]
        # x-=10
        # y-=10
        ptEnd = (x, y)
        thickness = 3
        lineType = 4
        point_color = color
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
        ptStart = ptEnd



font = 18
for i in range(6):
    # img = np.zeros((238, 281, 3), np.uint8)
    stroke_num = stroke_list[i]
    a = get_colors(stroke_num)
    color_index = 0
    file_name = file_name_list[i]
    file_path_list = [f'home/liuyt/FontRL/data/font/{font}/GB{file_name}_{j+1}M.txt' for j in range(stroke_num)]
    target = cv2.imread(f'home/liuyt/FontRL/data/font/{font}/GB{file_name}_R.bmp')
    shape = target.shape
    img = np.zeros((shape[0], shape[1], 3), np.uint8)

    random.shuffle(file_path_list)
    for j in range(stroke_num):
        file_path = file_path_list[j]
        draw_stroke(file_path,img,a[stroke_num-color_index-1],shape)
        color_index += 1
    cv2.imshow('iThin', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'image/{font}/r{i}.png',img)


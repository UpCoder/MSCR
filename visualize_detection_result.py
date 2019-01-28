# -*-coding=utf-8 -*-

import numpy as np
import util
import cv2

def draw_bbox(image_data, line, color, is_gt, border_width=1):
    line = util.str.remove_all(line, '\xef\xbb\xbf')
    if is_gt:
        data = line.split(' ')
        points = [int(v) for v in data[1:5]]
        points = np.asarray(
            [[points[0], points[1]], [points[0], points[3]], [points[2], points[3]], [points[2], points[1]]], np.int32)
        s = data[0]
        print('gt_points is ', np.shape(points), points)
    else:
        data = line.split(' ')
        points = [int(v) for v in data[2:6]]

        points = np.asarray(
            [[points[0], points[1]], [points[0], points[3]], [points[2], points[3]], [points[2], points[1]]], np.int32)
        s = data[0]
        print('det_points is ', np.shape(points), points)
    cnts = util.img.points_to_contours(points)
    util.img.draw_contours(image_data, cnts, -1, color = color, border_width = border_width)
    # cv2.putText(image_data, s, tuple(points[0]), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

       
def visualize(image_root, det_root, output_root, gt_root = None, multiphase_multislice_flag=False):
    def read_gt_file(image_name):
        gt_file = util.io.join_path(gt_root, '%s.txt'%(image_name))
        return util.io.read_lines(gt_file)

    def read_det_file(image_name):
        det_file = util.io.join_path(det_root, '%s.txt'%(image_name))
        return util.io.read_lines(det_file)
    
    def read_image_file(image_name):
        if multiphase_multislice_flag:
            idx = 1 # 上下两层，取中间
        else:
            idx = 2 # NC ART PV 取PVphase
        img = util.img.imread(util.io.join_path(image_root, image_name))
        single_slice = np.concatenate([np.expand_dims(img[:, :, idx], axis=2), np.expand_dims(img[:, :, idx], axis=2),
                                       np.expand_dims(img[:, :, idx], axis=2)], axis=2)
        return single_slice
    if not multiphase_multislice_flag:
        image_names = util.io.ls(image_root, '.jpg')
        print('image_names is ', image_names)
    else:
        image_names = util.io.ls(image_root, '_PPV.jpg')
    print('image_root is ', image_root)
    print('image_names is ', image_names)
    import os
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    for image_idx, image_name in enumerate(image_names):
        print('%d / %d: %s'%(image_idx + 1, len(image_names), image_name))
        image_data = read_image_file(image_name) # in BGR
        image_name = image_name.split('.')[0]
        det_image = image_data.copy()
        if not multiphase_multislice_flag:
            det_lines = read_det_file(image_name)
        else:
            det_lines = read_det_file(image_name.split('_PPV')[0])
        for line in det_lines:
            draw_bbox(det_image, line, color = util.img.COLOR_GREEN, is_gt=False)
        output_path = util.io.join_path(output_root, '%s_pred.jpg'%(image_name))
        util.img.imwrite(output_path, det_image)
        print("Detection result has been written to ", util.io.get_absolute_path(output_path))
        
        if gt_root is not None:
            gt_lines = read_gt_file(image_name.split('_PPV')[0])
            for line in gt_lines:
                draw_bbox(image_data, line, color = util.img.COLOR_BGR_RED, is_gt=True)
            util.img.imwrite(util.io.join_path(output_root, '%s_gt.jpg'%(image_name)), image_data)


def visualize_one_image(image_root, det_root, output_root, gt_root=None, multiphase_multislice_flag=False):
    '''
    将gt的bounding box和pred的bounding box画在一起
    :param image_root:
    :param det_root:
    :param output_root:
    :param gt_root:
    :param multiphase_multislice_flag:
    :return:
    '''
    def read_gt_file(image_name):
        gt_file = util.io.join_path(gt_root, '%s.txt' % (image_name))
        return util.io.read_lines(gt_file)

    def read_det_file(image_name):
        det_file = util.io.join_path(det_root, '%s.txt' % (image_name))
        return util.io.read_lines(det_file)

    def read_image_file(image_name):
        if multiphase_multislice_flag:
            idx = 1  # 上下两层，取中间
        else:
            idx = 2  # NC ART PV 取PVphase
        img = util.img.imread(util.io.join_path(image_root, image_name))
        single_slice = np.concatenate([np.expand_dims(img[:, :, idx], axis=2), np.expand_dims(img[:, :, idx], axis=2),
                                       np.expand_dims(img[:, :, idx], axis=2)], axis=2)
        return single_slice

    if not multiphase_multislice_flag:
        image_names = util.io.ls(image_root, '.jpg')
        print('image_names is ', image_names)
    else:
        image_names = util.io.ls(image_root, '_PPV.jpg')
    print('image_root is ', image_root)
    print('image_names is ', image_names)
    import os
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    for image_idx, image_name in enumerate(image_names):
        print('%d / %d: %s' % (image_idx + 1, len(image_names), image_name))
        image_data = read_image_file(image_name)  # in BGR
        image_name = image_name.split('.')[0]
        det_image = image_data.copy()
        if not multiphase_multislice_flag:
            det_lines = read_det_file(image_name)
        else:
            det_lines = read_det_file(image_name.split('_PPV')[0])
        for line in det_lines:
            draw_bbox(det_image, line, color=util.img.COLOR_GREEN, is_gt=False)


        if gt_root is not None:
            gt_lines = read_gt_file(image_name.split('_PPV')[0])
            for line in gt_lines:
                draw_bbox(det_image, line, color=util.img.COLOR_BGR_RED, is_gt=True)
            # util.img.imwrite(util.io.join_path(output_root, '%s_gt.jpg' % (image_name)), image_data)
        output_path = util.io.join_path(output_root, '%s_pred.jpg' % (image_name))
        util.img.imwrite(output_path, det_image)
        print("Detection result has been written to ", util.io.get_absolute_path(output_path))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='visualize detection result of pixel_link')
    parser.add_argument('--image', type=str, required = True,help='the directory of test image')
    parser.add_argument('--gt', type=str, default=None,help='the directory of ground truth txt files')
    parser.add_argument('--det', type=str, required = True, help='the directory of detection result')
    parser.add_argument('--output', type=str, required = True, help='the directory to store images with bboxes')
    # parser.add_argument('--multiphase_multislice_flag', type=bool, required=True, default=False,
    #                     help='flag the data whether is multiphase multislice')
    args = parser.parse_args()
    multiphase_multislice_flag = True
    print('**************Arguments*****************')
    print(args)
    print('****************************************')
    print('multiphase_multislice_flag is ', multiphase_multislice_flag)
    visualize_one_image(image_root=args.image, gt_root=args.gt, det_root=args.det, output_root=args.output,
                        multiphase_multislice_flag=multiphase_multislice_flag)

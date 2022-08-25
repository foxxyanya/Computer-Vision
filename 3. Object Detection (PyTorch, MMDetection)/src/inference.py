import os
import argparse
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from cfg import cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ch', '--checkpoint', default='best_model/yolox_l_8x8_300e_coco/epoch_10.pth',
        help='path to model\'s checkpoint'
    )
    parser.add_argument(
        '-i', '--inference_path', default='input/inference_data',
        help='path to inference images'
    )
    parser.add_argument(
        '-o', '--output_path', default='output_data',
        help='path to output images'
    )
    args = vars(parser.parse_args())

    model = init_detector(cfg, args['checkpoint'])
    model.cfg = cfg
    try:
        os.mkdir(args['output_path'])
    except:
        pass
    for im_name in os.listdir(args['inference_path']):
        im_path = os.path.join(args['inference_path'], im_name)
        im = mmcv.imread(im_path)
        result = inference_detector(model, im)
        colors = [(0, 255, 0), (255, 0, 0)]
        new_im_name = im_name[:-4] + "-output" + im_name[-4:]
        model.show_result(im, result, bbox_color=colors, text_color=colors, thickness=2., font_size=25,
                          out_file=os.path.join(args['output_path'], new_im_name))
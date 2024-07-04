import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict
from utils.biou import boundary_iou as biou_cal


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="SegNet", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="camvid", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=8,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="./checkpoint/camvid/SegNetbs8gpu1_train/model_200.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--best', action='store_true', help="Get the best result among last few checkpoints")
    parser.add_argument('--save', action='store_true', help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args




def test(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)

    data_list = []
    biou_list = []
    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        
        # output_msk = np.zeros(output.shape)
        # gt_msk = np.zeros(output.shape)

        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])
        
        # 分别计算每个类别的mask，因为gt中包含11个类别的mask值，0-10为具体的类别mask，11为void mask
        # for i in range(args.classes):
        #     output_msk[i] = (output == i).astype(np.uint8)
        #     gt_msk[i] = (gt == i).astype(np.uint8)

        biou = biou_cal(gt, output, dilation_ratio=0.05, cls_num=11)        # 计算biou, shape:(11,)
        biou_list.append(biou)                                              # biou_list.shape(len(test_loader), 11)

        # save the predicted image
        if args.save:
            save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=True)
    #这加了acc
    meanIoU, per_class_iu, acc = get_iou(data_list, args.classes)
    biou_list = np.array(biou_list)
    mean_biou = np.mean(biou_list, axis=0)
    return meanIoU, per_class_iu, acc, mean_biou


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    if not args.best:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                print("=====> loading checkpoint '{}'".format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['model'])
                # model.load_state_dict(convert_state_dict(checkpoint['model']))
            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

        print("=====> beginning validation")
        print("validation set length: ", len(testLoader))
        mIOU_val, per_class_iu, acc, biou = test(args, testLoader, model)
        print('mIOU_val:', mIOU_val)
        print('per_class_iu:', per_class_iu)
        # 打印acc
        print('acc:', acc)
        print('biou:', biou)

    # Get the best test result among the last 10 model records.
    else:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                dirname, basename = os.path.split(args.checkpoint)
                epoch = int(os.path.splitext(basename)[0].split('_')[1])
                mIOU_val = []
                acc_val = []
                per_class_iu = []
                biou_val = []
                for i in range(epoch - 2, epoch + 1):
                    basename = 'model_' + str(i) + '.pth'
                    resume = os.path.join(dirname, basename)
                    checkpoint = torch.load(resume)
                    model.load_state_dict(checkpoint['model'])
                    print("=====> beginning test the " + basename)
                    print("validation set length: ", len(testLoader))
                    mIOU_val_0, per_class_iu_0, acc, biou = test(args, testLoader, model)
                    mIOU_val.append(mIOU_val_0)
                    per_class_iu.append(per_class_iu_0)
                    #存acc
                    acc_val.append(acc)
                    biou_val.append(biou)

                index = list(range(epoch - 9, epoch + 1))[np.argmax(mIOU_val)]
                print("The best mIoU among the last 10 models is", index)
                print("mIoU_val:",mIOU_val)
                print("acc_val:",acc_val)
                # print('biou_val:', biou_val)
                per_class_iu = per_class_iu[np.argmax(mIOU_val)]
                acc_val = acc_val[np.argmax(mIOU_val)]
                biou_val = biou_val[np.argmax(mIOU_val)]
                mIOU_val = np.max(mIOU_val)

                print("Max mIoU:",mIOU_val)
                print("Per_class_iu of max mIoU:",per_class_iu)
                #打印acc
                #acc_val = acc_val[np.argmax(mIOU_val)]
                print("Acc of max mIoU:",acc_val)
                print("Acc of max biou:",biou_val)

            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # Save the result
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        args.logFile = 'test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Mean IoU: %.4f" % mIOU_val)
        logger.write("\nPer class IoU: ")
        for i in range(len(per_class_iu)):
            logger.write("%.4f\t" % per_class_iu[i])
    logger.flush()
    logger.close()


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)

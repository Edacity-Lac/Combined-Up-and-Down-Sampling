import os
import time,math
import datetime
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
import torch.onnx
from models import FastDepthV2, weights_init#,FastDepth
import utils, loss_func
from load_pretrained import load_pretrained_encoder, load_pretrained_fastdepth
from nyudepthv2_labled import create_data_loaders, worker_init_fn
# from nyudepthv2 import worker_init_fn, create_data_loaders
global args, writer
import pandas as pd
results_df = pd.DataFrame(columns=['Epoch', 'Total Loss', 'Validation Loss', 'Delta1 Accuracy', 'Average Time', 'RMSE'])


def train(model,optimizer,train_loader,val_loader,criteria=loss_func.DepthLoss(),epoch=0,batch=0):
    global results_df
    my_lr=0.01
    best_delta1, best_val_loss, best_epoch = None, None, None
    batch_count = batch
    if args.gpu and torch.cuda.is_available():
        model.cuda()
        criteria = criteria.cuda()

    print(f'{datetime.datetime.now().time().replace(microsecond=0)} Starting to train..')
    
    while epoch <= args.epochs-1:
        print(my_lr)
        print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Epoch#: {epoch+1} / {args.epochs}')
        model.train()
        interval_loss, total_loss= 0,0
        for i , (input,target) in enumerate(train_loader):
            batch_count += 1
            if args.gpu and torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()
            #input, target = input.float(), target.float()

            pred = model(input)
            loss = criteria(args.criterion,pred,target,epoch)
            # if epoch >3 and epoch < 20:
            #     my_lr=0.001
            if epoch >=21:
                my_lr=0.005

            optimizer = optim.SGD(model.parameters(), lr=my_lr, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if np.isnan(total_loss):
                raise ValueError('Loss is None. try to resume from last epoch and continue training')
            interval_loss += loss.item()
            '''if i % args.print_freq==0 and i>0:
                current_loss = interval_loss / args.print_freq
                print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Batch #: {i:5d}/{len(train_loader):0d} Train Loss:{current_loss:.4f}.')
                writer.add_scalars(f'Loss',{'Train': current_loss}, batch_count)
                img_grid_rgb = torchvision.utils.make_grid(input[:2], normalize=True)
                img_grid_depth = torchvision.utils.make_grid(target[:2], normalize=True)
                img_grid_pred = torchvision.utils.make_grid(pred[:2], normalize=True)
                writer.add_image("RGB", img_grid_rgb,global_step = batch_count)
                writer.add_image("Ground Truth Depth", img_grid_depth,global_step = batch_count)
                writer.add_image("Predicted Depth", img_grid_pred,global_step = batch_count)
                writer.flush()
                interval_loss = 0'''
        else:#只有前面的for循环没有因为break终止，正常退出，才会启用else
            print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Finish Epoch #{epoch+1} Total Loss:{total_loss/len(train_loader):.4f}. Saving checkpoint..')
            # 定义保存模型的完整路径
            if epoch >29:
                save_path = f'{args.weights_dir}/{args.backbone}/{args.criterion}/FastDepth_{epoch}.pth'
                # 检查包含模型文件的目录是否存在
                if not os.path.exists(os.path.dirname(save_path)):
                    # 如果目录不存在，创建目录
                    os.makedirs(os.path.dirname(save_path))
                # 现在目录已经存在，可以安全地保存模型
                torch.save({'epoch': epoch,'batch':batch_count,'model_state_dict': model.state_dict(),'optimizer_state_dict':
                            optimizer.state_dict(),'loss': total_loss/len(train_loader),'train_set':args.train_set,'val_set':args.val_set,'args':args},f'{args.weights_dir}/{args.backbone}/{args.criterion}/FastDepth_{epoch}.pth')
            print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Detour, running validation..')
            val_loss, delta1_acc , timer,rmse = evaluate_model(model,val_loader)
            writer.add_scalars(f'Logs/Loss',{'Train': total_loss/len(train_loader),'Validation Loss':val_loss}, epoch+1)
            writer.add_scalars(f'Logs/Accuarcy',{'Delta1 Accuarcy': delta1_acc}, epoch+1)
            writer.add_scalars(f'Logs/Time',{'Average time':timer},epoch+1)
            writer.add_scalars(f'Logs/RMSE',{'RMSE':rmse},epoch+1)
            results_df = results_df.append({
                'Epoch': epoch + 1,
                'Total Loss': total_loss / len(train_loader),
                'Validation Loss': val_loss,
                'Delta1 Accuracy': delta1_acc,
                'Average Time': timer,
                'RMSE': rmse
            }, ignore_index=True)
            save_dir = f'{args.save_dir}/{args.backbone}/{args.criterion}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 获取当前日期并格式化为 'YYYYMMDD' 格式
            current_date = datetime.date.today()

            # 构建文件名，包含日期
            file_name = f'training_results_{current_date}_{args.encoder_decoder_choice}.csv'
            # 保存DataFrame到CSV文件
            # results_df.to_csv(f'{save_dir}/training_results.csv', index=False)
            results_df.to_csv(f'{save_dir}/{file_name}', index=False)

            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_delta1 = delta1_acc
                best_epoch = epoch+1#1
                print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Best Validation Loss Update! best loss: {best_val_loss:.4f} Delta1 Acc:{delta1_acc:.4f}, Epoch: {best_epoch}')
            model.train()
            epoch+=1
    torch.save({'epoch': epoch,'batch':batch_count,'model_state_dict': model.state_dict(),'optimizer_state_dict':
                        optimizer.state_dict(),'loss': total_loss/len(train_loader),'train_set':args.train_set,'val_set':args.val_set,'args':args}, f'{args.weights_dir}/{args.backbone}/{args.criterion}/FastDepth_Final.pth')

                               
def evaluate_model(model,test_loader,criterion=loss_func.DepthLoss(),save_pic=True):
    if args.gpu and torch.cuda.is_available():
        model.cuda()
        criterion = criterion.cuda() 
    test_loss, delta1,timer,test_acc_L2 = 0,0,0,0
    if save_pic:
        imgs_dict = {}
    with torch.no_grad():
        model.eval()
        for index, (input,target) in enumerate(test_loader):
            if args.gpu and torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()
            input, target = input.float(), target.float()
            print(f'\r{index}/{len(test_loader)}', end='')
            time1 = time.time()
            # pred = model(input)
            pred = model(input)

            time1 = time.time() - time1
            
            test_loss += criterion(args.criterion,pred,target).item()
            
            valid_mask = ((target>0) + (pred>0)) > 0
            pred_output = pred[valid_mask]
            target_masked =  target[valid_mask]
            abs_diff = (pred_output - target_masked).abs() 
            RMSE = torch.sqrt(torch.mean((abs_diff).pow(2)))
        
            maxRatio = torch.max(pred_output / target_masked, target_masked / pred_output)
            d1 = float((maxRatio < 1.25).float().mean())
            delta1 += d1
            test_acc_L2+= RMSE.item()
            timer+=time1
            if save_pic:
                imgs_dict[d1] = [input,pred,target]
    if save_pic:
        utils.save_best_samples(imgs_dict)
    return test_loss/len(test_loader), delta1/len(test_loader),timer/len(test_loader),test_acc_L2/len(test_loader)

if __name__ == '__main__':
    args = utils.parse_args()
    print('Arguments are', args)
    if args.mode == 'train':
        writer = SummaryWriter(f'{args.tensorboard_dir}/{args.criterion}')

        if args.resume != None:
            resume_path = os.path.join(args.resume,'FastDepthV2_L1_Best.pth')
            assert os.path.isfile(resume_path), "No checkpoint found. abort.."
            print('Checkpoint found, loading...')
            checkpoint = torch.load(resume_path)
            args = checkpoint['args']
            epoch = checkpoint['epoch']+1
            batch = checkpoint['batch']
            model_state_dict = checkpoint['model_state_dict']
            if args.backbone == 'mobilenet':
                model = FastDepth()
            else:
                model = FastDepthV2()
            model.load_state_dict(model_state_dict)
            optimizer = optim.SGD(model.parameters(), lr = args.learning_rate ,weight_decay=args.weight_decay)
            train_set = checkpoint['train_set']
            val_set = checkpoint['val_set']
            train_loader = torch.utils.data.DataLoader(train_set, batch_size= args.bsize, shuffle=True,num_workers=args.workers, pin_memory=True, sampler=None,worker_init_fn=worker_init_fn,drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_set,batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True,drop_last=True)

            print('Finished loading')
            train(model,optimizer,train_loader,val_loader,epoch=epoch,batch=batch)

        else:
            print('Creating new model..')
            train_loader, val_loader = create_data_loaders(args)
            if args.backbone == 'mobilenet':
                model = FastDepth()
            else:
                model = FastDepthV2()
            model.encoder.apply(weights_init)
            model.decoder.apply(weights_init)
            optimizer = optim.SGD(model.parameters(), lr = args.learning_rate ,weight_decay=args.weight_decay)
            print('Model created')
            train(model,optimizer,train_loader,val_loader)

    else:
        print('error!')



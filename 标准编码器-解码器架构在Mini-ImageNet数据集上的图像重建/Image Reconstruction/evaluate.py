import pytorch_ssim
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import einops
import util
import wandb

def evaluate(model, dataloader, criterion, device,experiment,epoch):
    model.eval()
    cudnn.benchmark = False
    val_loss = 0
    psnr_list = []
    ssim_list = []
    mse_list = []
    mae_list = []
    wandb_images = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device=device)
            output = model(images)
            target = images.clone()
            target = target * 0.3081 + 0.1307
            target = target.to(device=device)
            val_loss += criterion(output, target).item() # sum up batch loss
            mae_list.append(criterion(output, target).item())
            psnr_list.append(util.psnr(target.squeeze().cpu().numpy().astype(np.float32), output.squeeze().cpu().numpy().astype(np.float32)))
            ssim_list.append(pytorch_ssim.ssim(target.cpu(), output.cpu()))
            mse_list.append(util.mse(target.squeeze().cpu().numpy().astype(np.float32), output.squeeze().cpu().numpy().astype(np.float32)))
            if epoch==100:
                out = (output[20].squeeze(0).detach().cpu()*255).numpy().astype(np.uint8)
                tar = (target[20].squeeze(0).detach().cpu()*255).numpy().astype(np.uint8)
                out = einops.rearrange(out, 'c h w ->  h w c')
                tar = einops.rearrange(tar, 'c h w ->  h w c')
                wandb_images.append(wandb.Image(tar, caption="Target Image",mode="RGB"))
                wandb_images.append(wandb.Image(out, caption="Reconstructed Image",mode="RGB"))
    if epoch == 100:
         experiment.log({"Examples": wandb_images})
    psnr = np.mean(psnr_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)
    ssim = np.mean(ssim_list)
    experiment.log({
        'psnr': psnr,
        'mse':mse,
        'mae':mae,
        'ssim':ssim,
        'epoch': epoch
    })
    model.train()





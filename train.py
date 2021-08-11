import torch
from models.rattnet.model_rattnet_index import rattnet_index
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loss.perceptual_loss import PerceptualLoss
import torch.backends.cudnn as cudnn
from dataset import *
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='the decay of learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=50, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='./dataset/train', help='the path of the training set')
    parser.add_argument('--valset_dir', type=str, default='./dataset/val', help='the path of the validation set')
    parser.add_argument('--val_freq', type=int, default=10, help='the frequency to do the validation')
    parser.add_argument('--load_checkpoint', type=str, default='', help='the path to load the checkpoint')
    parser.add_argument('--save_dir', type=str, default='./result', help='the path to save the training result')
    parser.add_argument('--save_name', type=str, default='1', help='the name of the current experiment')
    parser.add_argument('--save_freq', type=int, default=10, help='the frequency to save the training image and checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--alpha', type=float, default=5e-4, help='the weight of the attention consistency loss')
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint_dict['model'])
    else:
        model.load_state_dict(checkpoint_dict['model'])
    print("Loaded checkpoint '{}'" .format(
          checkpoint_path))
    return model, optimizer, epoch

def convert_to_numpy(img):
    return img.cpu().permute(0,2,3,1).data.numpy()[0]


if __name__ == '__main__':

    args = parse_args()

    train_set = TrainSetLoader(dataset_dir=args.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size, shuffle=True)

    val_set = TestSetLoader(dataset_dir=args.valset_dir, val=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=args.batch_size, shuffle=True)

    model = rattnet_index(args.backbone).cuda()
    model = torch.nn.DataParallel(model)

    loss_1 = PerceptualLoss().cuda()
    loss_2 = torch.nn.L1Loss().cuda()

    optimizer = torch.optim.Adam([paras for paras in model.parameters() if paras.requires_grad == True], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    loss_epoch = []
    loss_list = []

    epoch_ = 0

    if args.load_checkpoint!='':
        model, optimizer, epoch_ = load_checkpoint(args.load_checkpoint, model, optimizer)


    epoch_offset = max(0, epoch_)
    save_path_train = os.path.join(args.save_dir, args.save_name, 'train')
    if not os.path.isdir(save_path_train):
        os.makedirs(save_path_train)
    save_path_val = os.path.join(args.save_dir, args.save_name, 'val')
    if not os.path.isdir(save_path_val):
        os.makedirs(save_path_val)

    log_dir = os.path.join(args.save_dir, args.save_name, "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epoch_offset, args.n_epochs):

        model.train()
        for i, (im_l, im_r, gt_l, gt_r, disp, disp_r, mask, mask_r) in enumerate(train_loader):
            b, c, h, w = im_l.shape
            im_l, im_r, gt_l, gt_r  = Variable(im_l).cuda(), Variable(im_r).cuda(), Variable(gt_l).cuda(), Variable(gt_r).cuda()
            disp, disp_r = Variable(disp).cuda(), Variable(disp_r).cuda()
            mask, mask_r = Variable(mask).cuda(), Variable(mask_r).cuda()
            
            out_l, out_r, index_l, index_r = model(im_l, im_r)
                
            loss_p = loss_1(out_l, gt_l, batch_size=args.batch_size) + loss_1(out_r, gt_r, batch_size=args.batch_size)
            loss_d = loss_2(index_l*mask, disp*mask) + loss_2(index_r*mask_r, disp_r*mask_r)
                    

            loss =  loss_p + args.alpha*loss_d
                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.data.cpu())
            print('Epoch----%5d, Iter----%5d, loss---%f' % (epoch, i, float(loss.data.cpu())))
        
        if epoch % args.save_freq==0:
            col1 = np.concatenate([convert_to_numpy(im_l), convert_to_numpy(out_l), convert_to_numpy(gt_l)], axis=1)
            col2 = np.concatenate([convert_to_numpy(im_r), convert_to_numpy(out_r), convert_to_numpy(gt_r)], axis=1)
            new_img = Image.fromarray(np.uint8(np.concatenate([col1, col2], axis=0).clip(0,1) * 255.0), 'RGB')
            new_img.save(os.path.join(save_path_train, "epoch%06d.png"%(epoch)), quality=95)
        scheduler.step()
        
        loss_train = float(np.array(loss_epoch).mean())
        loss_list.append(loss_train)
        print('Epoch----%5d, loss---%f' % (epoch, loss_train))
        writer.add_scalar('loss_train', loss_train, epoch)
        loss_epoch = []

        if epoch % args.save_freq == 0:
            save_name = '{}/{}.pth'.format(save_path_train, epoch)
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({'epoch': epoch,
                        'model': model_state_dict, 
                        'optimizer': optimizer.state_dict(),
                        },save_name)            
            print("Save checkpoint at epoch {})" .format(epoch)) 
        
        if epoch % args.val_freq == 0:
            model.eval()
            for i, (im_l, im_r, gt_l, gt_r, file_name) in enumerate(val_loader):
                b, c, h, w = im_l.shape
                with torch.no_grad():
                    im_l, im_r, gt_l, gt_r  = Variable(im_l).cuda(), Variable(im_r).cuda(), Variable(gt_l).cuda(), Variable(gt_r).cuda()
                    out_l, out_r, _,_ = model(im_l, im_r)
                    
                col1 = np.concatenate([convert_to_numpy(im_l), convert_to_numpy(out_l), convert_to_numpy(gt_l)], axis=1)
                col2 = np.concatenate([convert_to_numpy(im_r), convert_to_numpy(out_r), convert_to_numpy(gt_r)], axis=1)
                new_img = Image.fromarray(np.uint8(np.concatenate([col1, col2], axis=0).clip(0,1) * 255.0), 'RGB')
                new_img.save(os.path.join(save_path_val,"epoch%06d_%06d.png"%(epoch,i)), quality=95)

from models.rattnet.model_rattnet_index import rattnet_index
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import *
import argparse
from PIL import Image
import torch


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--testset_dir', type=str, default='./dataset/test')
    parser.add_argument('--load_checkpoint', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./result')
    parser.add_argument('--save_name', type=str, default='1')
    return parser.parse_args()

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint_dict['model'])
    else:
        model.load_state_dict(checkpoint_dict['model'])
    print("Loaded checkpoint '{}'" .format(
          checkpoint_path))
    return model

if __name__ == '__main__':

    args = parse_args()

    test_set = TestSetLoader(dataset_dir=args.testset_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size, shuffle=True)

    
    model = rattnet_index().cuda()

    if args.load_checkpoint!='':
        model= load_checkpoint(args.load_checkpoint, model)

    save_path = os.path.join(args.save_dir, args.save_name, 'test', args.load_checkpoint.split("/")[-1][:-4])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model.eval()

    for i, (im_l, im_r, file_name) in enumerate(test_loader):
        b, c, h, w = im_l.shape
        with torch.no_grad():
            im_l, im_r= Variable(im_l).cuda(), Variable(im_r).cuda()
            out_l, out_r, _,_ = model(im_l, im_r)

        for bidx in range(b):
            input_l_s = im_l.cpu().permute(0,2,3,1).data.numpy()[bidx]
            input_r_s = im_r.cpu().permute(0,2,3,1).data.numpy()[bidx]
            out_l_s = out_l.cpu().permute(0,2,3,1).data.numpy()[bidx]
            out_r_s = out_r.cpu().permute(0,2,3,1).data.numpy()[bidx]
            col1 = np.concatenate([input_l_s, out_l_s], axis=1)
            col2 = np.concatenate([input_r_s, out_r_s], axis=1)
            new_img = Image.fromarray(np.uint8(np.concatenate([col1, col2], axis=0).clip(0,1) * 255.0), 'RGB')
            new_img.save(os.path.join(save_path,"img"+str(file_name[bidx])+".png"), quality=95)
            
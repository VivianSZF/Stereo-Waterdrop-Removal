import torch
import torch.nn as nn
from torch.nn import functional as F

class row_attention_maxindex(nn.Module):
    def __init__(self, in_channels=1024, row_size=3, stride=2):
        super(row_attention_maxindex, self).__init__()
        self.attention_1 = row_attention_perlevel_0(1024, row_size=3, stride=stride)
        self.attention_2 = row_attention_perlevel(512, row_size=3, stride=stride)
        self.attention_3 = row_attention_perlevel(256, row_size=3, stride=stride)
        # self.attention_4 = row_attention_perlevel(64)
    
    def forward(self, left_p, right_p):
        # left_1, right_1 = self.attention_1(left_p[3], right_p[3])
        # left_2, right_2 = self.attention_2(left_p[2], right_p[2])
        # left_3, right_3 = self.attention_3(left_p[1], right_p[1])
        # left_4, right_4 = self.attention_4(left_p[0], right_p[0])
        # return [left_1, left_2, left_3, left_4], [right_1, right_2, right_3, right_4]
        

        left_1, right_1, index_l1, index_r1 = self.attention_1(left_p[2], right_p[2])
        left_2, right_2, index_l2, index_r2 = self.attention_2(left_p[1], right_p[1], pre_l=index_l1, pre_r=index_r1)
        left_3, right_3, index_l3, index_r3 = self.attention_3(left_p[0], right_p[0], pre_l=index_l2, pre_r=index_r2)
        return [left_1, left_2, left_3], [right_1, right_2, right_3], [index_l1, index_l2, index_l3], [index_r1, index_r2, index_r3]



class row_attention_perlevel(nn.Module):
    def __init__(self, in_channels=1024, row_size=3, stride=2, pooling=False):
        super(row_attention_perlevel, self).__init__()
        self.row_size = row_size
        self.stride = stride
        rep_channels = in_channels//2
        if rep_channels == 0:
            rep_channels = 1
        self.theta_r1 = nn.Conv2d(in_channels+1, rep_channels, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='reflect')
        self.theta_r2 = nn.Conv2d(in_channels+1, rep_channels, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='reflect')
        self.theta_r3 = nn.Conv2d(in_channels+1, rep_channels, kernel_size=3, stride=1, padding=4, dilation=4, padding_mode='reflect')
        self.theta_r = nn.Conv2d(rep_channels * 3, rep_channels, 1, 1, 0, bias=False)
        if pooling:
            self.phi_r1 = nn.Sequential(nn.Conv2d(in_channels, rep_channels, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='reflect'),
                            nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi_r2 = nn.Sequential(nn.Conv2d(in_channels, rep_channels, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='reflect'),
                            nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi_r3 = nn.Sequential(nn.Conv2d(in_channels, rep_channels, kernel_size=3, stride=1, padding=4, dilation=4, padding_mode='reflect'),
                            nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.phi_r1 = nn.Conv2d(in_channels, rep_channels, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='reflect')
            self.phi_r2 = nn.Conv2d(in_channels, rep_channels, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='reflect')
            self.phi_r3 = nn.Conv2d(in_channels, rep_channels, kernel_size=3, stride=1, padding=4, dilation=4, padding_mode='reflect')
        self.phi_r = nn.Conv2d(rep_channels * 3, rep_channels, 1, 1, 0, bias=False)

        self.self_attention = co_non_local_block(in_channels=in_channels, pooling=pooling)
    def sample_rows(self, x, row_size=None, stride=None):
        if row_size is None:
            row_size = self.row_size
        if stride is None:
            stride = self.stride
        b, c, h, w = x.size()
        # F.pad
        remainder_h = (h-row_size) % stride
        if remainder_h!=0:
            x = F.pad(x, (0,0,0,remainder_h), mode="reflect")
        new_rows = x.unfold(2, row_size, stride).permute(0,2,1,4,3)
        #b, n_rows, c, row_size, w
        return new_rows, remainder_h
    
    def aggregate_rows(self, x, target_h, remainder, row_size=None, stride=None):
        if row_size is None:
            row_size = self.row_size
        if stride is None:
            stride = self.stride
        b, n_blocks, c, row_size, w = x.size()
        output_map = torch.zeros(b, c, target_h+remainder, w).cuda()
        count_map = torch.zeros(b, c, target_h+remainder, w).cuda()
        for i in range(n_blocks):
            output_map[:,:,i*stride:i*stride+row_size, :] = x[:,i,:,:,:]
            count_map[:,:,i*stride:i*stride+row_size, :] += 1
        output_map/=count_map
        # print(torch.min(count_map))
        
        return output_map[:,:,:target_h, :]
    
    def enlarge(self, x, index):
        in_ = torch.cat([x,index],dim=1)
        query = self.theta_r(torch.cat([self.theta_r1(in_), self.theta_r2(in_), self.theta_r3(in_)], dim=1))
        key = self.phi_r(torch.cat([self.phi_r1(x), self.phi_r2(x), self.phi_r3(x)], dim=1))
        return query, key

    def forward(self, left_, right_, pre_l, pre_r):
        pre_l_up = F.interpolate(pre_l, scale_factor=2, mode='nearest')
        pre_l_up = pre_l_up * 2
        pre_r_up = F.interpolate(pre_r, scale_factor=2, mode='nearest')
        pre_r_up = pre_r_up * 2

        b,c,h,w = left_.size()

        query_l, key_l = self.enlarge(left_, pre_l_up/w)
        query_r, key_r = self.enlarge(right_, pre_r_up/w)

        left_rows, remainder_l = self.sample_rows(left_)
        right_rows, remainder_r = self.sample_rows(right_)
        pre_l_rows, _ = self.sample_rows(pre_l_up)
        pre_r_rows, _ = self.sample_rows(pre_r_up)
        query_l_rows, _ = self.sample_rows(query_l)
        query_r_rows, _ = self.sample_rows(query_r)
        key_l_rows, _ = self.sample_rows(key_l)
        key_r_rows, _ = self.sample_rows(key_r)

        b, n_blocks, c, row_size, w = left_rows.size()

        
        out_left, out_right, index_l, index_r = self.self_attention(left_rows.reshape(-1,c,row_size,w), right_rows.reshape(-1,c,row_size,w), \
                                        pre_l_rows.reshape(-1,1,row_size,w), pre_r_rows.reshape(-1,1,row_size,w), \
                                        query_l_rows.reshape(-1,c//2,row_size,w), key_l_rows.reshape(-1,c//2,row_size,w), \
                                        query_r_rows.reshape(-1,c//2,row_size,w), key_r_rows.reshape(-1,c//2,row_size,w))
        out_left = out_left.view(b, n_blocks, c, row_size, w)
        out_right = out_right.view(b, n_blocks, c, row_size, w)
        index_l = index_l.view(b, n_blocks, 1, row_size, w)
        index_r = index_r.view(b, n_blocks, 1, row_size, w)

        final_left = self.aggregate_rows(out_left, left_.size(2), remainder_l)
        final_right = self.aggregate_rows(out_right, left_.size(2), remainder_r)
        final_left_index = self.aggregate_rows(index_l, left_.size(2), remainder_l)
        final_right_index = self.aggregate_rows(index_r, left_.size(2), remainder_r)
        return final_left, final_right, final_left_index, final_right_index


class co_non_local_block(nn.Module):
    def __init__(self, in_channels, rep_channels=None, pooling=False, bn=True):
        super(co_non_local_block, self).__init__()
        
        if rep_channels is None:
            rep_channels = in_channels//2
            if rep_channels == 0:
                rep_channels = 1
        self.rep_channels = rep_channels

        self.theta = nn.Conv2d(in_channels+1, rep_channels, kernel_size=1, stride=1, padding=0)

        if pooling:
            self.phi = nn.Sequential(nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0),
                                        nn.MaxPool2d(kernel_size=(2, 2)))
            self.g = nn.Sequential(nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0),
                                        nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.phi = nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0)
            self.g = nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0)

        if bn:
            self.up = nn.Sequential(nn.Conv2d(rep_channels, in_channels, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(in_channels))
            nn.init.constant_(self.up[1].weight, 0)
            nn.init.constant_(self.up[1].bias, 0)
        else:
            self.up = nn.Conv2d(rep_channels, in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.up.weight, 0)
            nn.init.constant_(self.up.bias, 0)
    
    def forward(self, left, right, pre_l, pre_r, query_l, key_l, query_r, key_r):
        """
         x: query frame b*c*h*w
         ref: reference images b*c*h*w
        """
        b,c,h,w = left.size()
        pre_l = pre_l / w
        pre_r = pre_r / w

        theta1 = self.theta(torch.cat([left, pre_l],dim=1)).view(b, self.rep_channels, -1).permute(0, 2, 1)+ \
                query_l.view(b, self.rep_channels, -1).permute(0, 2, 1)
        phi1 = self.phi(right).view(b, self.rep_channels, -1) + \
                key_r.view(b, self.rep_channels, -1)
        g1 = self.g(right).view(b, self.rep_channels, -1).permute(0, 2, 1)

        theta2 = self.theta(torch.cat([right, pre_r],dim=1)).view(b, self.rep_channels, -1).permute(0, 2, 1)+ \
                query_r.view(b, self.rep_channels, -1).permute(0, 2, 1)
        phi2 = self.phi(left).view(b, self.rep_channels, -1)+ \
                key_l.view(b, self.rep_channels, -1)
        g2 = self.g(left).view(b, self.rep_channels, -1).permute(0, 2, 1)

        inter1 = torch.bmm(theta1, phi1)
        inter_atten1 = F.softmax(inter1, dim = -1)
        other_indices1 = torch.arange(w).cuda().repeat(b, h*w, h)
        max_indices1 = (other_indices1 * inter_atten1).sum(dim=-1)
        index_l = (torch.arange(h*w).cuda().repeat(b,1) % w)-max_indices1
        index_l = index_l.view(b, h, w)

        after_atten1 = torch.bmm(inter_atten1, g1)
        after_atten1 = after_atten1.permute(0, 2, 1).contiguous()
        after_atten1 = after_atten1.view(b, self.rep_channels, h, w)
        l_n = left + self.up(after_atten1)

        inter2 = torch.bmm(theta2, phi2)
        inter_atten2 = F.softmax(inter2, dim = -1)
        other_indices2 = torch.arange(w).cuda().repeat(b, h*w, h)
        max_indices2 = (other_indices2 * inter_atten2).sum(dim=-1)
        index_r = (torch.arange(h*w).cuda().repeat(b,1) % w)-max_indices2
        index_r = index_r.view(b, h, w)

        after_atten2 = torch.bmm(inter_atten2, g2)
        after_atten2 = after_atten2.permute(0, 2, 1).contiguous()
        after_atten2 = after_atten2.view(b, self.rep_channels, h, w)
        r_n = right + self.up(after_atten2)


        return l_n, r_n, index_l, index_r
    
class row_attention_perlevel_0(nn.Module):
    def __init__(self, in_channels=1024, row_size=3, stride=2):
        super(row_attention_perlevel_0, self).__init__()
        self.row_size = row_size
        self.stride = stride
        self.self_attention = co_non_local_block_0(in_channels=in_channels, pooling=False)
    def sample_rows(self, x, row_size=None, stride=None):
        if row_size is None:
            row_size = self.row_size
        if stride is None:
            stride = self.stride
        b, c, h, w = x.size()
        # F.pad
        remainder_h = (h-row_size) % stride
        if remainder_h!=0:
            x = F.pad(x, (0,0,0,remainder_h), mode="reflect")
        new_rows = x.unfold(2, row_size, stride).permute(0,2,1,4,3)
        #b, n_rows, c, row_size, w
        return new_rows, remainder_h
    
    def aggregate_rows(self, x, target_h, remainder, row_size=None, stride=None):
        if row_size is None:
            row_size = self.row_size
        if stride is None:
            stride = self.stride
        b, n_blocks, c, row_size, w = x.size()
        output_map = torch.zeros(b, c, target_h+remainder, w).cuda()
        count_map = torch.zeros(b, c, target_h+remainder, w).cuda()
        for i in range(n_blocks):
            output_map[:,:,i*stride:i*stride+row_size, :] = x[:,i,:,:,:]
            count_map[:,:,i*stride:i*stride+row_size, :] += 1
        output_map/=count_map
        # print(torch.min(count_map))
        
        return output_map[:,:,:target_h, :]

    def forward(self, left_, right_):
        left_rows, remainder_l = self.sample_rows(left_)
        right_rows, remainder_r = self.sample_rows(right_)

        b, n_blocks, c, row_size, w = left_rows.size()

        
        out_left, out_right, index_l, index_r = self.self_attention(left_rows.reshape(-1,c,row_size,w), right_rows.reshape(-1,c,row_size,w))
        out_left = out_left.view(b, n_blocks, c, row_size, w)
        out_right = out_right.view(b, n_blocks, c, row_size, w)
        index_l = index_l.view(b, n_blocks, 1, row_size, w)
        index_r = index_r.view(b, n_blocks, 1, row_size, w)

        final_left = self.aggregate_rows(out_left, left_.size(2), remainder_l)
        final_right = self.aggregate_rows(out_right, left_.size(2), remainder_r)
        final_left_index = self.aggregate_rows(index_l, left_.size(2), remainder_l)
        final_right_index = self.aggregate_rows(index_r, left_.size(2), remainder_r)
        return final_left, final_right, final_left_index, final_right_index

class co_non_local_block_0(nn.Module):
    def __init__(self, in_channels, rep_channels=None, pooling=True, bn=True):
        super(co_non_local_block_0, self).__init__()
        
        if rep_channels is None:
            rep_channels = in_channels//2
            if rep_channels == 0:
                rep_channels = 1
        self.rep_channels = rep_channels

        self.theta = nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0)

        if pooling:
            self.phi = nn.Sequential(nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0),
                                        nn.MaxPool2d(kernel_size=(2, 2)))
            self.g = nn.Sequential(nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0),
                                        nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.phi = nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0)
            self.g = nn.Conv2d(in_channels, rep_channels, kernel_size=1, stride=1, padding=0)

        if bn:
            self.up = nn.Sequential(nn.Conv2d(rep_channels, in_channels, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(in_channels))
            nn.init.constant_(self.up[1].weight, 0)
            nn.init.constant_(self.up[1].bias, 0)
        else:
            self.up = nn.Conv2d(rep_channels, in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.up.weight, 0)
            nn.init.constant_(self.up.bias, 0)
    
    def forward(self, left, right):
        """
         x: query frame b*c*h*w
         ref: reference images b*c*h*w
        """
        b,c,h,w = left.size()

        theta1 = self.theta(left).view(b, self.rep_channels, -1).permute(0, 2, 1)
        phi1 = self.phi(right).view(b, self.rep_channels, -1)
        g1 = self.g(right).view(b, self.rep_channels, -1).permute(0, 2, 1)

        theta2 = self.theta(right).view(b, self.rep_channels, -1).permute(0, 2, 1)
        phi2 = self.phi(left).view(b, self.rep_channels, -1)
        g2 = self.g(left).view(b, self.rep_channels, -1).permute(0, 2, 1)

        inter1 = torch.bmm(theta1, phi1)
        inter_atten1 = F.softmax(inter1, dim=-1)
        #index
        other_indices1 = torch.arange(w).cuda().repeat(b, h*w, h)
        max_indices1 = (other_indices1 * inter_atten1).sum(dim=-1)
        index_l = (torch.arange(h*w).cuda().repeat(b,1) % w)-max_indices1
        index_l = index_l.view(b, h, w)

        after_atten1 = torch.bmm(inter_atten1, g1)
        after_atten1 = after_atten1.permute(0, 2, 1).contiguous()
        after_atten1 = after_atten1.view(b, self.rep_channels, h, w)
        l_n = left + self.up(after_atten1)

        inter2 = torch.bmm(theta2, phi2)
        inter_atten2 = F.softmax(inter2, dim=-1)
        #index
        other_indices2 = torch.arange(w).cuda().repeat(b, h*w, h)
        max_indices2 = (other_indices2 * inter_atten2).sum(dim=-1)
        index_r = (torch.arange(h*w).cuda().repeat(b,1) % w)-max_indices2
        index_r = index_r.view(b, h, w)

        after_atten2 = torch.bmm(inter_atten2, g2)
        after_atten2 = after_atten2.permute(0, 2, 1).contiguous()
        after_atten2 = after_atten2.view(b, self.rep_channels, h, w)
        r_n = right + self.up(after_atten2)


        return l_n, r_n, index_l, index_r
    

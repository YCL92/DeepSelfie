#!/usr/bin/env python
# coding: utf-8

# # Network Testing

# ## Includes

# In[ ]:


# mass includes
import os, sys, argparse
import numpy as np
import pyexiv2 as exiv2
import rawpy as rp
import torch as t
from torchvision.utils import save_image


# ## Modules

# In[ ]:


class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, root, device=None):
        save_list = [
            file for file in os.listdir(root)
            if file.startswith(self.model_name)
        ]
        save_list.sort()
        file_path = os.path.join(root, save_list[-1])
        state_dict = t.load(file_path, map_location=device)
        self.load_state_dict(t.load(file_path, map_location=device))
        print('Weights loaded: %s' % file_path)

        return


class channelAtt(BasicModule):
    def __init__(self, channels):
        super(channelAtt, self).__init__()

        # squeeze-excitation layer
        self.glb_pool = t.nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze_excite = t.nn.Sequential(
            t.nn.Linear(channels, int(channels / 16)), t.nn.LeakyReLU(0.2),
            t.nn.Linear(int(channels / 16), channels), t.nn.Sigmoid())

    def forward(self, x):
        scale = self.glb_pool(x)
        scale = self.squeeze_excite(scale.squeeze())
        x = scale.view((x.size(0), x.size(1), 1, 1)) * x

        return x


class encode(BasicModule):
    def __init__(self, in_channels, out_channels, max_pool=True):
        super(encode, self).__init__()

        # features
        if max_pool:
            self.features = t.nn.Sequential(
                t.nn.MaxPool2d((2, 2)),
                t.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                t.nn.LeakyReLU(0.2),
                t.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                channelAtt(out_channels), t.nn.LeakyReLU(0.2))
        else:
            self.features = t.nn.Sequential(
                t.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                t.nn.LeakyReLU(0.2),
                t.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                channelAtt(out_channels), t.nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.features(x)

        return x


class skipConn(BasicModule):
    def __init__(self, in_channels, out_channels, avg_pool=True):
        super(skipConn, self).__init__()

        # features
        if avg_pool:
            self.features = t.nn.Sequential(
                t.nn.AvgPool2d((2, 2)),
                t.nn.Conv2d(in_channels, out_channels, 1),
                channelAtt(out_channels), t.nn.Tanh())
        else:
            self.features = t.nn.Sequential(
                t.nn.Conv2d(in_channels, out_channels, 1),
                channelAtt(out_channels), t.nn.Tanh())

    def forward(self, x):
        x = self.features(x)

        return x


class decode(BasicModule):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 up_sample=True):
        super(decode, self).__init__()

        # features
        if up_sample:
            self.features = t.nn.Sequential(
                t.nn.Conv2d(in_channels, inter_channels, 1),
                t.nn.Conv2d(inter_channels, inter_channels, 3, padding=1),
                t.nn.LeakyReLU(0.2),
                t.nn.Upsample(scale_factor=2, mode='nearest'),
                t.nn.Conv2d(inter_channels, out_channels, 3, padding=1),
                channelAtt(out_channels), t.nn.LeakyReLU(0.2))
        else:
            self.features = t.nn.Sequential(
                t.nn.Conv2d(in_channels, inter_channels, 1),
                t.nn.Conv2d(inter_channels, inter_channels, 3, padding=1),
                t.nn.LeakyReLU(0.2),
                t.nn.Conv2d(inter_channels, out_channels, 3, padding=1),
                t.nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.features(x)

        return x


class gainEst(BasicModule):
    def __init__(self):
        super(gainEst, self).__init__()
        self.model_name = 'gainEst'

        # encoders
        self.head = encode(3, 64, max_pool=False)
        self.down1 = encode(64, 96, max_pool=True)
        self.down2 = encode(96, 128, max_pool=True)
        self.down3 = encode(128, 192, max_pool=True)

        # bottleneck
        self.bottleneck = t.nn.Sequential(
            t.nn.MaxPool2d(2, 2), t.nn.Conv2d(192, 256, 3, padding=1),
            t.nn.LeakyReLU(0.2), t.nn.Conv2d(256, 256, 3, padding=1),
            t.nn.Upsample(scale_factor=2, mode='nearest'),
            t.nn.Conv2d(256, 192, 3, padding=1), channelAtt(192),
            t.nn.LeakyReLU(0.2))

        # decoders
        self.up1 = decode(384, 384, 128, up_sample=True)
        self.up2 = decode(256, 256, 96, up_sample=True)
        self.up3 = decode(192, 192, 64, up_sample=True)
        self.seg_out = t.nn.Sequential(decode(128, 128, 64, up_sample=False),
                                       t.nn.Conv2d(64, 2, 1))

        # external actication
        self.sigmoid = t.nn.Sigmoid()

        # prediction
        self.features = t.nn.Sequential(
            t.nn.Conv2d(5, 64, 3, stride=2, padding=1), t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(64, 96, 3, stride=2, padding=1), t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(96, 128, 3, stride=2, padding=1), t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(128, 192, 3, stride=2, padding=1), t.nn.LeakyReLU(0.2),
            t.nn.Conv2d(192, 256, 3, stride=2, padding=1), t.nn.LeakyReLU(0.2))
        self.amp_out = t.nn.Sequential(t.nn.Linear(8 * 6 * 256,
                                                   128), t.nn.LeakyReLU(0.2),
                                       t.nn.Linear(128, 64),
                                       t.nn.LeakyReLU(0.2), t.nn.Linear(64, 2))

    def forward(self, thumb_img, struct_img):
        # segmentation
        out_head = self.head(struct_img)
        out_d1 = self.down1(out_head)
        out_d2 = self.down2(out_d1)
        out_d3 = self.down3(out_d2)
        out_bottleneck = self.bottleneck(out_d3)
        out_u1 = self.up1(t.cat([out_d3, out_bottleneck], dim=1))
        out_u2 = self.up2(t.cat([out_d2, out_u1], dim=1))
        out_u3 = self.up3(t.cat([out_d1, out_u2], dim=1))
        out_mask = self.seg_out(t.cat([out_head, out_u3], dim=1))

        # prediction
        out_features = self.features(
            t.cat([thumb_img, self.sigmoid(out_mask)], dim=1))
        out_amp = self.amp_out(out_features.view(out_features.size(0), -1))
        out_amp = t.clamp(out_amp, 0.0, 1.0)

        return out_mask, out_amp


class ispNet(BasicModule):
    def __init__(self):
        super(ispNet, self).__init__()

        # encoders
        self.head = encode(8, 64, max_pool=False)
        self.down1 = encode(64, 64, max_pool=True)
        self.down2 = encode(64, 64, max_pool=True)

        # skip connections
        self.skip1 = skipConn(1, 64, avg_pool=False)
        self.skip2 = skipConn(64, 64, avg_pool=True)
        self.skip3 = skipConn(64, 64, avg_pool=True)

        # decoders
        self.up1 = decode(128, 64, 64, up_sample=True)
        self.up2 = decode(128, 64, 64, up_sample=True)
        self.srgb_out = t.nn.Sequential(
            decode(128, 64, 64, up_sample=False),
            t.nn.Upsample(scale_factor=2, mode='nearest'),
            t.nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, color_map, mag_map, amp, wb):
        # to prevent saturation
        mag_map = amp.view(-1, 1, 1, 1) * mag_map
        mag_map = t.nn.functional.tanh(mag_map - 0.5)
        max_mag = 2.0 * amp.view(-1, 1, 1, 1)
        max_mag = t.nn.functional.tanh(max_mag - 0.5)
        mag_map = mag_map / max_mag

        # encoder outputs
        out_head = self.head(t.cat([color_map, mag_map, wb], dim=1))
        out_d1 = self.down1(out_head)
        out_d2 = self.down2(out_d1)

        # skip connection outputs
        out_s1 = self.skip1(mag_map)
        out_s2 = self.skip2(out_head)
        out_s3 = self.skip3(out_d1)

        # decoder outputs
        out_u1 = self.up1(t.cat([out_s3, out_d2], dim=1))
        out_u2 = self.up2(t.cat([out_s2, out_u1], dim=1))
        out_srgb = self.srgb_out(t.cat([out_s1, out_u2], dim=1))
        out_srgb = t.clamp(out_srgb, 0.0, 1.0)

        return out_srgb


class rawProcess(BasicModule):
    def __init__(self):
        super(rawProcess, self).__init__()
        self.model_name = 'rawProcess'

        # isp module
        self.isp_net = ispNet()

        # fusion
        self.fusion = t.nn.Sequential(t.nn.Conv2d(6, 128, 3, padding=1),
                                      channelAtt(128),
                                      t.nn.Conv2d(128, 3, 3, padding=1))

    def forward(self, raw_data, amp_high, amp_low, wb):
        # convert to color map and mgnitude map
        mag_map = t.sqrt(t.sum(t.pow(raw_data, 2), 1, keepdim=True))
        color_map = raw_data / (mag_map + 1e-4)

        # convert to sRGB images
        out_high = self.isp_net(color_map, mag_map, amp_high, wb)
        out_low = self.isp_net(color_map, mag_map, amp_low, wb)

        # image fusion
        out_fused = self.fusion(t.cat([out_high, out_low], dim=1))
        out_fused = t.clamp(out_fused, 0.0, 1.0)

        return out_fused


# ## Test

# In[ ]:


# normalization
def normalize(raw_data, bk_level, sat_level):
    normal_raw = t.empty_like(raw_data)
    for index in range(raw_data.size(0)):
        for channel in range(raw_data.size(1)):
            normal_raw[index, channel, :, :] = (
                raw_data[index, channel, :, :] -
                bk_level[channel]) / (sat_level - bk_level[channel])

    return normal_raw


# resize Bayer pattern
def downSample(raw_data, struct_img_size):
    # convert Bayer pattern to down-sized sRGB image
    batch, _, hei, wid = raw_data.size()
    raw_img = raw_data.new_empty((batch, 3, hei, wid))
    raw_img[:, 0, :, :] = raw_data[:, 0, :, :]  # R
    raw_img[:,
            1, :, :] = (raw_data[:, 1, :, :] + raw_data[:, 2, :, :]) / 2.0  # G
    raw_img[:, 2, :, :] = raw_data[:, 3, :, :]  # B

    # down-sample to small size
    if hei != struct_img_size[1] and wid != struct_img_size[0]:
        raw_img = t.nn.functional.interpolate(raw_img,
                                              size=(struct_img_size[1],
                                                    struct_img_size[0]),
                                              mode='bicubic')
    raw_img = t.clamp(raw_img, 0.0, 1.0)

    return raw_img


# image standardization (mean 0, std 1)
def standardize(srgb_img):
    struct_img = t.empty_like(srgb_img)
    adj_std = 1.0 / t.sqrt(srgb_img.new_tensor(srgb_img[0, :, :, :].numel()))
    for index in range(srgb_img.size(0)):
        mean = t.mean(srgb_img[index, :, :, :])
        std = t.std(srgb_img[index, :, :, :])
        adj_std = t.max(std, adj_std)
        struct_img[index, :, :, :] = (srgb_img[index, :, :, :] -
                                      mean) / adj_std

    return struct_img


# main entry
def main(args):
    # initialization
    att_size = (256, 192)
    amp_range = (1, 20)

    # choose GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = t.device(args.device)

    # define models
    gain_est_model = gainEst().to(device)
    gain_est_model.load('./saves', device=device)
    gain_est_model.eval()
    raw_process_model = rawProcess().to(device)
    raw_process_model.load('./saves', device=device)
    raw_process_model.eval()

    # search for valid files
    file_list = [file for file in os.listdir(args.input) if '.DNG' in file]
    file_list.sort()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #  loop to process
    for file in file_list:
        # read black, saturation, and whitebalance
        img_md = exiv2.ImageMetadata(os.path.join(args.input, file))
        img_md.read()

        blk_level = img_md['Exif.SubImage1.BlackLevel'].value
        sat_level = img_md['Exif.SubImage1.WhiteLevel'].value
        cam_wb = img_md['Exif.Image.AsShotNeutral'].value

        # convert flat Bayer pattern to 4D tensor (RGGB)
        raw_img = rp.imread(os.path.join(args.input, file))
        flat_bayer = raw_img.raw_image_visible
        raw_data = np.stack((flat_bayer[0::2, 0::2], flat_bayer[0::2, 1::2],
                             flat_bayer[1::2, 0::2], flat_bayer[1::2, 1::2]),
                            axis=2)

        with t.no_grad():
            # copy to device
            blk_level = t.from_numpy(np.array(blk_level,
                                              dtype=np.float32)).to(device)
            sat_level = t.from_numpy(np.array(sat_level,
                                              dtype=np.float32)).to(device)
            cam_wb = t.from_numpy(np.array(cam_wb,
                                           dtype=np.float32)).to(device)
            raw_data = t.from_numpy(raw_data.astype(np.float32)).to(device)
            raw_data = raw_data.permute(2, 0, 1).unsqueeze(0)

            # downsample
            if args.resize:
                raw_data = t.nn.functional.interpolate(raw_data,
                                                       size=args.resize,
                                                       mode='bicubic')

            # pre-processing
            raw_data = normalize(raw_data, blk_level, sat_level)
            cam_wb = cam_wb.view([1, 3, 1, 1]).expand(
                [1, 3, raw_data.size(2),
                 raw_data.size(3)])
            cam_wb = cam_wb.clone()
            thumb_img = downSample(raw_data, att_size)
            struct_img = standardize(thumb_img)

            # run model
            _, pred_amp = gain_est_model(thumb_img, struct_img)
            pred_amp = t.clamp(pred_amp * amp_range[1], amp_range[0],
                               amp_range[1])
            print('Predicted ratio(fg/bg) for %s: %.2f, %.2f.' %
                  (file, pred_amp[0, 0], pred_amp[0, 1]))
            amp_high, _ = t.max(pred_amp, 1)
            amp_low, _ = t.min(pred_amp, 1)
            pred_fused = raw_process_model(raw_data, amp_high, amp_low, cam_wb)

        # save to images
        save_image(
            pred_fused.cpu().squeeze(),
            os.path.join(args.output,
                         '%s' % file.replace('.DNG', '-fuse.png')))

        # fisheye lens calibration
        # modified from https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
        if args.calib:
            import cv2

            DIM = (4000, 3000)
            K = np.array([[1715.9053454852321, 0.0, 2025.0267134780845],
                          [0.0, 1713.8092418955127, 1511.2242172068645],
                          [0.0, 0.0, 1.0]])
            D = np.array([[0.21801544244553403], [0.011549797903321477],
                          [-0.05436236262851618], [-0.01888678272481524]])
            img = cv2.imread(
                os.path.join(args.output,
                             '%s' % file.replace('.DNG', '-fuse.png')))
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            calib_img = cv2.remap(img,
                                  map1,
                                  map2,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(
                os.path.join(args.output,
                             '%s' % file.replace('.DNG', '-calib.png')),
                calib_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./samples', help='input directory')
    parser.add_argument('--output',
                        default='./results',
                        help='output directory')
    parser.add_argument('--resize',
                        default=None,
                        type=tuple,
                        help='downsample to smaller size (hxw)')
    parser.add_argument('--device',
                        default='cpu',
                        help='device to be used (cpu or cuda)')
    parser.add_argument('--calib',
                        action='store_true',
                        help='perform fisheye calibration')
    args = parser.parse_args()
    main(args)


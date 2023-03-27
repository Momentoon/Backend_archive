import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
from firebase_admin import storage
cred = credentials.Certificate("D:/20NP/sd/momentoon-cd88f-firebase-adminsdk-fy3z0-e0bc9fd707.json")
default_app = firebase_admin.initialize_app(cred,{'storageBucket': 'momentoon-cd88f.appspot.com'})
bucket_name = "momentoon-cd88f.appspot.com"

from tqdm import tqdm
from torch.nn import functional as F
import pyrebase
import inference_image



class ResBlock(nn.Module):
    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)
        return output


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        return output


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.last_act = nn.Tanh()


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        if self.is_last:
            output = self.last_act(output)
        else:
            output = self.act(output)
        return output



class SimpleGenerator(nn.Module):
    def __init__(self, num_channel=32, num_blocks=4):
        super(SimpleGenerator, self).__init__()
        self.down1 = DownBlock(3, num_channel)
        self.down2 = DownBlock(num_channel, num_channel*2)
        self.down3 = DownBlock(num_channel*2, num_channel*3)
        self.down4 = DownBlock(num_channel*3, num_channel*4)
        res_blocks = [ResBlock(num_channel*4)]*num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up1 = UpBlock(num_channel*4, num_channel*3)
        self.up2 = UpBlock(num_channel*3, num_channel*2)
        self.up3 = UpBlock(num_channel*2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down4 = self.res_blocks(down4)
        up1 = self.up1(down4)
        up2 = self.up2(up1+down3)
        up3 = self.up3(up2+down2)
        up4 = self.up4(up3+down1)
        return up4



weight = torch.load('weight.pth', map_location='cpu')
model = SimpleGenerator()
model.load_state_dict(weight)
#torch.save(model.state_dict(), 'weight.pth')
model.eval()


config = {
}

bucket = storage.bucket()
firebase=pyrebase.initialize_app(config)
storage = firebase.storage()

path_UF = " " #Unfiltered, original image
path_BU = " "
path_RS = " "
codeToDownload = 'temp'
model_list = [] # TODO: Inserting model, currently we have only one temporary model.


path_filtered = " "
path_archive = " "


path_artist   = " "

#for i in range(0,2):
while 1:
    all_files = storage.child("images/UPLOAD").list_files()


    for file in all_files:

            print(file.name)
            if 'UPLOAD/'+codeToDownload in file.name:
                print(file.name)
                z=storage.child(file.name).get_url(None)
                storage.child(file.name).download(""+path_UF+"/"+file.name.replace("images/UPLOAD","") )
                storage.delete(file.name)
                #file.delete() # Delete file from UPLOAD Folder! we should use it carefully.

                for file in os.listdir(path_UF):
                        #TODO : make image resizeable or models able to resize even if the picture isn't fit to model.
                        print(file)
                        if codeToDownload in file:
                            if(file[4] == 'N'):# previous prototype images
                                storage.child(path_archive+file).put(""+path_UF+"/"+file) #original FILE stored in path_archive+file, put means location of the file borrowed.\
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                raw_image = cv2.imread(load_path)

                                tempx, tempy, tempz= raw_image.shape
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)

                                image = raw_image/127.5 - 1
                                image = image.transpose(2, 0, 1)
                                image = torch.tensor(image).unsqueeze(0)
                                output = model(image.float())
                                output = output.squeeze(0).detach().numpy()
                                output = output.transpose(1, 2, 0)
                                output = (output + 1) * 127.5
                                output = np.clip(output, 0, 255).astype(np.uint8)
                                #output = np.concatenate([raw_image, output], axis=1)

                                cv2.imwrite(save_path, output)


                                os.remove(path_UF+"/"+file)

                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                load_path = os.path.join(path_RS, "Result"+file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                print(save_path, ' ', tempy, ' ', tempx)
                                raw_image = cv2.imread(load_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)

                            if(file[4] == '0'): # temp artist 1
                                storage.child(path_archive+file).put(""+path_UF+"/"+file) #original FILE stored in path_archive+file, put means location of the file borrowed.\
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                raw_image = cv2.imread(load_path)

                                tempx, tempy, tempz= raw_image.shape
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image) # save image

                                image = raw_image/127.5 - 1
                                image = image.transpose(2, 0, 1)
                                image = torch.tensor(image).unsqueeze(0)
                                output = model(image.float())
                                output = output.squeeze(0).detach().numpy()
                                output = output.transpose(1, 2, 0)
                                output = (output + 1) * 127.5
                                output = np.clip(output, 0, 255).astype(np.uint8)
                                #output = np.concatenate([raw_image, output], axis=1)

                                cv2.imwrite(save_path, output)


                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                load_path = os.path.join(path_RS, "Result"+file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                print(save_path, ' ', tempy, ' ', tempx)
                                raw_image = cv2.imread(load_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)


                for file in os.listdir(path_RS):
                    storage.child(path_filtered+"FC"+file).put(""+path_RS+"/"+file) # FC : Filter completed, temporary name
                    os.remove(path_RS+"/"+file)

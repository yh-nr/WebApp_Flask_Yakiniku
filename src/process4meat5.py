# #標準ライブラリ
import io, base64                           #

# #画像変換
# from PIL import Image                       #

# pytorch系のimport
import torch                                #
import torch.nn as nn                       #
from torchvision import transforms          #
from torchvision.models import resnet18     #学習時に使ったのと同じ学習済みモデルをインポート
#import pytorch_lightning as pl              #

from process_common import resize_image, validate_base64_image



# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# #　ネットワークの定義
# class Net(pl.LightningModule):

#     def __init__(self):
#         super().__init__()

#         #学習時に使ったのと同じ学習済みモデルを定義
#         self.feature = resnet18(pretrained=True) 
#         self.fc = nn.Linear(1000, 2)

#     def forward(self, x):
#         #学習時に使ったのと同じ順伝播
#         h = self.feature(x)
#         h = self.fc(h)
#         return h
    
# PytorchLightningを使わないバージョン
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 学習時に使ったのと同じ学習済みモデルを定義
        self.feature = resnet18(pretrained=True)
        self.fc1 = nn.Linear(1000, 250)
        self.fc2 = nn.Linear(250, 5)

    def forward(self, x):    
        h = x                       #こうしておいた方が、ネットワークを書き換える時に便利な気がする。
        # h = self.DA_transform(h)    #データ拡張をオンラインで入れるならここ？
        h = self.feature(h)
        h = self.fc1(h)
        h = nn.functional.relu(h)
        h = self.fc2(h)
        return h
    

# 推論したラベルから犬か猫かを返す関数
def getName(label):
    label_names = ['カルビ','ハラミ','ロース','タン','レバー']
    return label_names[label]

# 入力：リサイズ前のbase64画像
# 出力：推論結果、確率、リサイズ後のbase64画像
def meat5_process(image_base64_original):
    if validate_base64_image(image_base64_original): 
        image_base64_resized, image_resized = resize_image(image_base64_original, 500) 
        Name_, NameProba_ = predict(image_resized)
        return Name_, NameProba_, image_base64_resized, image_resized
    else:
        return


def predict(img):

    # ネットワークの準備
    net = Net().cpu().eval()

    # 学習済みモデルの重み（dog_cat.pt）を読み込み
    
    net.load_state_dict(torch.load('./static/models/meat5_classification.pt', map_location=torch.device('cpu')))
    # net.load_state_dict(torch.load('./src/dog_cat.pt', map_location=torch.device('cpu')))
    
    # データの前処理
    img = transform(img)
    print(img.shape)
    img = img.unsqueeze(0) # 1次元増やす

    # 推論
    pred = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    NameProba_ = round((max(torch.softmax(net(img), dim=1)[0]) * 100).item(),2)
    print(f'pred:{type(pred)}')

    Name_ = getName(int(pred))
    print('呼び出されたよ２')
    return Name_, NameProba_

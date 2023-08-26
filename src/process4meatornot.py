# #標準ライブラリ
import io, base64
# pytorch系のimport
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

from process_common import resize_image, validate_base64_image

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
# ネットワークの定義
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 学習時に使ったのと同じ学習済みモデルを定義
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        # 学習時に使ったのと同じ順伝播
        h = self.feature(x)
        h = self.fc(h)
        return h
    

# 推論したラベルから犬か猫かを返す関数
def getName(label):
    if label==0:
        return '肉以外'
    elif label==1:
        return '肉'

# 入力：リサイズ前のbase64画像
# 出力：推論結果、確率、リサイズ後のbase64画像
def meatornot_process(image_base64_original):
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
    net.load_state_dict(torch.load('./static/models/dog_cat.pt', map_location=torch.device('cpu')))
    # net.load_state_dict(torch.load('./src/dog_cat.pt', map_location=torch.device('cpu')))

    # データの前処理
    img = transform(img)
    print(img.shape)
    img = img.unsqueeze(0) # 1次元増やす

    # 推論
    pred = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    NameProba_ = round((max(torch.softmax(net(img), dim=1)[0]) * 100).item(),2)

    Name_ = getName(pred)
    return Name_, NameProba_

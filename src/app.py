with open('.version', 'r') as file:
    version = file.read().strip()

print("Version:", version)


# 必要なモジュールのインポート
from process_common import send_message, ResultPost2Spreadsheet
from process4dogcat import dogcat_process
from process4meat5 import meat5_process 
from flask import Flask, request, render_template


# PredictProcessList
PPL = [meat5_process, dogcat_process]

# Flask のインスタンスを作成
app = Flask(__name__)


# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def request_route():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':

        data = request.json
        img_base64_original = data['image']
        model_index = data['model_index']

        print(model_index)
        try:Name_, NameProba_, base64_data, image_data = PPL[model_index](img_base64_original)
        except:return
        try:
            send_message(f'この画像は{NameProba_}%の確率で{Name_}です。\n（使用モデル：{model_index}）', image_data)
            ResultPost2Spreadsheet('テストタイトル', f'この画像は{NameProba_}%の確率で{Name_}です。\n（使用モデル：{PPL[model_index].__name__}）', base64_data, model_index)
        except:
            pass
        return render_template('result.html', Name=Name_, NameProba=NameProba_, image=base64_data)

    # GET メソッドの定義
    elif request.method == 'GET':    
        with open('helpdoc.txt', 'r', encoding='utf-8') as file:
            helpdoc = file.read()
        return render_template('index.html', helpdoc = helpdoc, version=version)


# アプリケーションの実行の定義
if __name__ == '__main__':
    # app.run(debug=True, host='192.168.0.137', port=80)
    app.run(debug=True)
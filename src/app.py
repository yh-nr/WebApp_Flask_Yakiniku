# 必要なモジュールのインポート
from process4dogcat import dogcat_process # animal.py から前処理とネットワークの定義を読み込み
from process4meatornot import meatornot_process # animal.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect

# PredictProcessList
PPL = [dogcat_process, meatornot_process]

# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# 拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def request_route():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':

        data = request.json
        img_base64_original = data['image']
        model_index = data['model_index']

        # with open('text_file.txt', 'w') as file:
        #     file.write(img_base64_original)
        print(model_index)
        Name_, NameProba_, base64_data = PPL[model_index](img_base64_original)
        return render_template('result.html', Name=Name_, NameProba=NameProba_, image=base64_data)

    # GET メソッドの定義
    elif request.method == 'GET':    
        with open('helpdoc.txt', 'r', encoding='utf-8') as file:
            helpdoc = file.read()
        return render_template('index.html', helpdoc = helpdoc)


# アプリケーションの実行の定義
if __name__ == '__main__':
    # app.run(debug=True, host='192.168.0.137', port=80)
    app.run(debug=True)
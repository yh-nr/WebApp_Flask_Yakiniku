# 必要なモジュールのインポート
from process4dogcat import dogcat_process # animal.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect




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
    print('検証')
    # リクエストがポストかどうかの判別
    if request.method == 'POST':

        data = request.json
        img_base64_original = data['image']

        with open('text_file.txt', 'w') as file:
            file.write(img_base64_original)

        animalName_, animalNameProba_, base64_data = dogcat_process(img_base64_original)
        print('process完了')
        return render_template('result.html', animalName=animalName_, animalNameProba=animalNameProba_, image=base64_data)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    # app.run(debug=True, host='192.168.0.137', port=80)
    app.run(debug=True)
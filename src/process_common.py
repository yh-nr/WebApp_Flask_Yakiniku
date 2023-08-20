#標準ライブラリ
import io, base64, re, os, requests                   #

#画像変換
from PIL import Image                           #


def resize_image(base64_string, output_size):

    
    # Base64データのヘッダー部分を削除（例: 'data:image/png;base64,'）
    header, base64_image = base64_string.split(',', 1)

    # Step 1: base64でエンコードされた画像データをデコードする
    img_data = base64.b64decode(base64_image)
    
    # Step 2: バイナリデータからPillowのImageオブジェクトを作成する
    img = Image.open(io.BytesIO(img_data))
    
    # Step 3: 指定されたサイズに短辺を合わせ、アスペクト比を保ったサイズを計算する
    width, height = img.size
    aspect_ratio = width / height
    if width < height:
        new_width = output_size
        new_height = int(output_size / aspect_ratio)
    else:
        new_height = output_size
        new_width = int(output_size * aspect_ratio)
    
    # Step 4: 計算されたサイズにリサイズする
    img_resized = img.resize((new_width, new_height), Image.ANTIALIAS).convert("RGB")
    
    # Step 5: リサイズした画像を再びbase64でエンコードする
    buffer = io.BytesIO()
    img_resized.save(buffer, format="PNG")
    base64_resized_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    base64_resized = 'data:image/png;base64,{}'.format(base64_resized_str)

    # リサイズした画像をbase64とpillowのimageとそれぞれ返す
    return base64_resized, img_resized


def validate_base64_image(base64_data):
    MAX_SIZE = 3264 * 2448 * 2
    ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/gif']
    
    # Decode base64 data and get the mime type (e.g., 'image/png')
    mime_match = re.search(r'data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+).*,.*', base64_data)
    if not mime_match:
        print('Invalid image format.')
        return False
    
    mime_type = mime_match.group(1)
    if mime_type not in ALLOWED_TYPES:
        print('Only JPEG, PNG, and GIF files are allowed.')
        return False
    
    # Calculate the size of the base64 image data
    size = len(base64_data) * 3 // 4 - base64_data.count('=', -2)  # adjusted for padding characters
    if size > MAX_SIZE:
        print('File size should be less than 2MB.')
        return False

    return True


# Line送信
def send_message(message_text, image):
    print('テスト')
    LineApiKey = os.getenv('line_key')
    if LineApiKey==None:
        LineApiKey = os.environ['LINE_NOTIFY_API_TOKEN']
    print(LineApiKey)
    headers = {'Authorization': 'Bearer ' + LineApiKey}
    data = {'message': f'{message_text}'}
    buf = io.BytesIO()
    image.save(buf, 'png')
    buf.seek(0)
    buf_r = io.BufferedReader(buf)
    files = {'imageFile': buf_r}
    requests.post('https://notify-api.line.me/api/notify', headers=headers, data=data, files=files)


def ResultPost2Spreadsheet(title, post_comment, image_base64, model_index):
    payload = {
        'title': "推論結果テスト",
        'message': post_comment,
        'image_base64': image_base64,
        'sheet_name': "test_predict"
    }
    
    post_url = os.getenv('spreadsheet_post_url')
    if post_url==None:
        post_url = os.environ['SUMMARY_POST_URL']
    
    response = requests.post(post_url, data=payload)

    # 応答を確認します。
    if response.status_code == 200:
        return('画像がスプレッドシートに正常にアップロードされました。')
    else:
        return('画像のアップロードに失敗しました。ステータスコード：' + str(response.status_code))




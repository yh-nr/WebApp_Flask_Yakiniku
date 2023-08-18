// 即時実行関数を使用して、コードをスコープ内に閉じ込める
(async function() {
    
    // HTMLからvideoの要素を取得
    // ユーザーのカメラからのメディアストリームを取得
    // 取得したメディアストリームをvideo要素のsrcObjectに設定
    const videoElement = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;

    // HTMLからcanvasの要素を取得
    // canvasから2D描画コンテキストを取得
    const canvasElement = document.getElementById('canvas');
    const context = canvasElement.getContext('2d');

    // HTMLからcaptureボタンの要素を取得
    const captureButton = document.getElementById('capture')
    
    // ファイルインプット要素を取得
    const fileInput = document.getElementById('file-input');


    // Caputure
    captureButton.addEventListener('click', async () => {
        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        const dataURL = canvasElement.toDataURL('image/png');
        submitBase64Image(dataURL)
    });


    // ファイルが選択されたときのイベントリスナーを設定
    fileInput.addEventListener('change', function() {
        // 選択されたファイルを取得（複数選択可能な場合は、files[0] の代わりにループ処理が必要）
        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = function(event) {
            let img = new Image();
            img.onload = function() {
                context.drawImage(img, 0, 0);
            }
            img.src = event.target.result;
            submitBase64Image(event.target.result);

        };

        // ファイルを Data URL（Base64 形式）として読み込む
        reader.readAsDataURL(file);  
    });

    
})(); // 即時実行関数の終了





// base64画像の検証
function validateBase64Image(base64Data) {
    const MAX_SIZE = 3264 * 2448 * 2;
    const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/gif'];
    
    // Decode base64 data and get the mime type (e.g., 'image/png')
    const mime = base64Data.match(/data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+).*,.*/);
    if (!mime || mime.length < 2) {
        alert('Invalid image format.');
        return false;
    }
    
    const mimeType = mime[1];
    if (!ALLOWED_TYPES.includes(mimeType)) {
        alert('Only JPEG and PNG files are allowed.');
        return false;
    }
    
    // Calculate the size of the base64 image data
    const size = Math.ceil((base64Data.length / 4) * 3);
    if (size > MAX_SIZE) {
        alert('File size should be less than 2MB.');
        return false;
    }

    return true;
}


// base64画像の検証
function submitBase64Image(img) {
    if(!validateBase64Image(img)){
        console.log('画像ファイルが不正です')
        return;
    }
        
    // サーバにPOSTリクエストを送信（相対パスを使用）
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: img })
    })
    // .then(response => response.json())
    // .then(data => {
    //     console.log('サーバからのレスポンス:', data);
    // })
    // .catch(error => {
    //     console.error('エラー:', error);
    // });
}
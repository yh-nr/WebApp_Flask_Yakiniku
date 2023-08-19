// 即時実行関数を使用して、コードをスコープ内に閉じ込める
(async function() {
    
    // HTMLからvideoの要素を取得
    // ユーザーのカメラからのメディアストリームを取得
    // 取得したメディアストリームをvideo要素のsrcObjectに設定
    const videoElement = document.getElementById('video');
    
    var constraints = { audio: false, video: { facingMode: "environment" } };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = stream;

    // HTMLからcanvasの要素を取得
    // canvasから2D描画コンテキストを取得
    const canvasElement = document.getElementById('canvas');
    const context = canvasElement.getContext('2d');

    // HTMLからcaptureボタンの要素を取得
    const captureButton = document.getElementById('capture')
    
    // ファイルインプット要素を取得
    const fileInput = document.getElementById('file-input');

    
    // // h1タグを取得
    const h1_title = document.getElementById('h1_title');
    // // videoの高さに基づいてh1タグのmargin-topを設定
    // video.addEventListener('loadedmetadata', function() {
    //     h1_title.style.marginTop = (video.clientHeight + 20) + 'px';
    // });


    // 撮影して推論ボタンが押されたら実行
    captureButton.addEventListener('click', async () => {
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        video.style.display = 'none';
        canvas.style.display = 'block';
        // canvasの高さに基づいてh1タグのmargin-topを設定
        result.textContent = '推論中．．．';
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
                // 画像のアスペクト比を計算
                const aspectRatio = img.width / img.height;
        
                // キャンバスの幅に合わせて、画像の新しい高さを計算
                const newWidth = canvas.width;
                const newHeight = newWidth / aspectRatio;
        
                // 画像をキャンバスに描画（指定した幅と高さで）
                canvas.height = newHeight;
                context.drawImage(img, 0, 0, newWidth, newHeight);
        
                video.style.display = 'none';
                canvas.style.display = 'block';
        
                // canvasの高さに基づいてh1タグのmargin-topを設定
                // h1_title.style.marginTop = (canvas.clientHeight + 20) + 'px';
            }
            result.textContent = '推論中．．．';
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
        console.log('画像ファイルが不正です');
        return;
    }
        
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: img })
    })
    .then(response => response.text())


    // [1]取得したHTMLをページに挿入
    .then(html => {
        document.getElementById('result').innerHTML = html;
    })
    // // [2]現在のページの内容を新しいHTMLで上書き
    // .then(html => {
    //     document.open();
    //     document.write(html);
    //     document.close();
    // })
    .catch(error => {
        console.error('エラー:', error);
    });
}


// let currentStream = null;
// let currentDeviceId = null;
// let devices = [];

// async function getDevices() {
//     devices = await navigator.mediaDevices.enumerateDevices();
//     return devices.filter(device => device.kind === 'videoinput');
// }

// async function switchCamera() {
//     const videoDevices = await getDevices();
//     if (videoDevices.length === 0) return;

//     let nextDeviceId = videoDevices[0].deviceId;
//     if (currentDeviceId) {
//         const currentDeviceIndex = videoDevices.findIndex(device => device.deviceId === currentDeviceId);
//         if (currentDeviceIndex + 1 < videoDevices.length) {
//             nextDeviceId = videoDevices[currentDeviceIndex + 1].deviceId;
//         }
//     }

//     const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: nextDeviceId } });
//     if (currentStream) {
//         currentStream.getTracks().forEach(track => track.stop());
//     }

//     currentDeviceId = nextDeviceId;
//     currentStream = stream;
//     videoElement.srcObject = stream;
// }

// const videoElement = document.getElementById('video');
// const switchCameraButton = document.getElementById('switch-camera');
// switchCameraButton.addEventListener('click', switchCamera);


const showHelp = document.getElementById("showHelp");
const overlay = document.getElementById("overlay");



showHelp.addEventListener("click", () => {
    overlay.style.display = "flex";
});

overlay.addEventListener("click", (event) => {
  if (event.target === overlay) {
    overlay.style.display = "none";
  }
});
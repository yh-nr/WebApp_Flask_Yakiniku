
// Variables
let currentStream = null;
let currentDeviceId = null;
let devices = [];

// Elements
const videoElement = document.getElementById('video');
const switchCameraButton = document.getElementById('switch-camera');

// Functions
async function getDevices() {
    devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(device => device.kind === 'videoinput');
}

async function switchCamera() {
    const videoDevices = await getDevices();
    if (videoDevices.length === 0) return;

    let nextDeviceId = videoDevices[0].deviceId;
    if (currentDeviceId) {
        const currentDeviceIndex = videoDevices.findIndex(device => device.deviceId === currentDeviceId);
        if (currentDeviceIndex + 1 < videoDevices.length) {
            nextDeviceId = videoDevices[currentDeviceIndex + 1].deviceId;
        }
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: nextDeviceId } });
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    currentDeviceId = nextDeviceId;
    currentStream = stream;
    videoElement.srcObject = stream;
}

function validateBase64Image(img) {
    if (!img.startsWith('data:image')) {
        alert('不正なフォーマットです。');
        return false;
    }
    return true;
}

function submitBase64Image(img) {
    if (!validateBase64Image(img)) {
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
    .then(html => {
        document.getElementById('result').innerHTML = html;
    })
    .catch(error => {
        console.error('エラー:', error);
    });
}

// Event Listeners
switchCameraButton.addEventListener('click', switchCamera);

(async function() {
    const videoElement = document.getElementById('video');
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (err) {
        console.error('Error accessing the camera:', err);
    }
})();

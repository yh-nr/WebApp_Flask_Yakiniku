(async function() {
    const videoElement = document.getElementById('video');
    const canvasElement = document.getElementById('canvas');
    const snapshotButton = document.getElementById('snapshot');
        
    const context = canvasElement.getContext('2d');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    
    snapshotButton.addEventListener('click', () => {
        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        const dataURL = canvasElement.toDataURL('image/png');
        const downloadLink = document.createElement('a');
        downloadLink.href = dataURL;
        downloadLink.download = 'snapshot.png';
        downloadLink.click();
    });
    
})();

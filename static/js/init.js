function resize_canvas() {
    let navvar_height = document.getElementById("navvar").offsetHeight + 10;
    let window_height = window.innerHeight;

    let content = document.getElementById("content");
    content.style.height = ( window_height - navvar_height - 5 ) + "px";
    content_width = content.clientWidth;
    content_height = content.clientHeight;
}

let content_width, content_height;
window.addEventListener("resize", resize_canvas);

window.onload = function () {
    resize_canvas();
    let canvas = document.getElementById("canvas");
    let ctx = canvas.getContext('2d');
    let img = new Image();

    img.src = img_url;
    img.onload = function(){
        let img_width = img.width;
        let img_height = img.height;

        let threshold = ( content_height * img_width ) / ( content_width * img_height);
        if(threshold > 1){
            canvas.width = content_width;
            canvas.height = content_width * img_height / img_width;
        }else {
            canvas.width = content_height * img_width / img_height;
            canvas.height = content_height;
        }

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        let xText = Math.random() * canvas.width;
        let yText = Math.random() * canvas.height;
        // let rotateText = Math.floor( Math.random() * 181 ) - 90;

        ctx.font = "100px 'ＭＳ ゴシック'";
        // ctx.rotate( rotateText * Math.PI / 180 );
        ctx.fillText("ザッパーン", xText, yText, canvas.width-xText);

    };
};


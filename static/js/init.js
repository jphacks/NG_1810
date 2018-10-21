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
        ctx.fillText(fill_text, xText, yText, canvas.width-xText);
    };
};

// canvas上のイメージを保存
function saveCanvas(){
    imageType = "image/jpeg";
    fileName = "jphacks.jpg";

    let canvas = document.getElementById("canvas");
    // base64エンコードされたデータを取得 「data:image/png;base64,iVBORw0k～」
    let base64 = canvas.toDataURL(imageType);
    // base64データをblobに変換
    let blob = Base64toBlob(base64);
    // blobデータをa要素を使ってダウンロード
    saveBlob(blob, fileName);
}

// Base64データをBlobデータに変換
function Base64toBlob(base64)
{
    // カンマで分割して以下のようにデータを分ける
    // tmp[0] : データ形式（data:image/png;base64）
    // tmp[1] : base64データ（iVBORw0k～）
    let tmp = base64.split(',');
    // base64データの文字列をデコード
    let data = atob(tmp[1]);
    // tmp[0]の文字列（data:image/png;base64）からコンテンツタイプ（image/png）部分を取得
    let mime = tmp[0].split(':')[1].split(';')[0];
    //  1文字ごとにUTF-16コードを表す 0から65535 の整数を取得
    let buf = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i++) {
        buf[i] = data.charCodeAt(i);
    }
    // blobデータを作成
    return new Blob([buf], { type: mime });
}

// 画像のダウンロード
function saveBlob(blob, fileName)
{
    let url = (window.URL || window.webkitURL);
    // ダウンロード用のURL作成
    let dataUrl = url.createObjectURL(blob);
    // イベント作成
    let event = document.createEvent("MouseEvents");
    event.initMouseEvent("click", true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
    // a要素を作成
    let a = document.createElementNS("http://www.w3.org/1999/xhtml", "a");
    // ダウンロード用のURLセット
    a.href = dataUrl;
    // ファイル名セット
    a.download = fileName;
    // イベントの発火
    a.dispatchEvent(event);
}
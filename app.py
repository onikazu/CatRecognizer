import os

import numpy as np

from flask import Flask, request, abort

from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage)

from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)


ACCESS_TOKEN = os.environ["LINE_ACCESS_TOKEN"]
SECRET = os.environ["LINE_SECRET_KEY"]

FQDN = "https://cat-recognizer.herokuapp.com/"

line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Requestbody: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return'OK'


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    # 取得した画像ファイル
    with open("static/"+event.message.id+".jpg", "wb") as f:
        f.write(message_content.content)

        test_url = "./static/"+event.message.id+".jpg"

        img = image.load_img(test_url, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        # モデルのロード
        model = load_model('dog_cat.h5')
        result_predict = model.predict(x)

        if result_predict < 0.5:
            text = f"This is cat(cat: {1-result_predict}, dog: {result_predict})"
        if result_predict >= 0.5:
            text = f"This is dog (cat: {1-result_predict}, dog: {result_predict})"

        #line_bot_api.reply_message(event.reply_token,ImageSendMessage(original_content_url=FQDN+"/static/"+event.message.id+".jpg",preview_image_url=FQDN+"/static/"+event.message.id+".jpg"))
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))

if __name__ == "__main__":
    app.run()

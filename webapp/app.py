# config: utf-8
import os
import locale
locale.setlocale(locale.LC_NUMERIC, 'ja_JP')
from flask import Flask, render_template
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc
from dbmodels import Item

app = Flask(__name__)

# DBとの接続
engine = create_engine("sqlite:///flaski/items_on_sale.db", connect_args={'check_same_thread': False})
session = sessionmaker(bind=engine)()

@app.route("/")
def index():
    CONFIDENCE_RATE = 0.3
    items = session.query(Item).filter((Item.pred_price * (1 - CONFIDENCE_RATE)  - Item.price) > 0).order_by((Item.pred_price * (1 - CONFIDENCE_RATE)  - Item.price)).all()
    # items = session.query(Item).all()
    return render_template("index.html", items=items)

@app.context_processor
def utility_processor():
    def format_currency(amount):
        return locale.format('%d', int(amount), True)
    return dict(format_currency=format_currency)

# メイン関数
if __name__ == "__main__":
    app.run(debug=True)
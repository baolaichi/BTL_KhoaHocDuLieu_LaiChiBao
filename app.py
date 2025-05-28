from flask import Flask, render_template, request, send_file
import pandas as pd
from model import (
    predict_spending,
    cluster_customers,
    classify_customers,
    create_histograms
)
import plotly.io as pio
import io

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    df = predict_spending(None)
    table = df.head(10).to_html(classes="table table-striped")
    return render_template("index.html", tables=table)

@app.route("/classify")
def classify():
    df = classify_customers(None)
    table = df.head(10).to_html(classes="table table-striped")
    return render_template("index.html", tables=table)

@app.route("/cluster")
def cluster():
    df, fig3d, fig2d = cluster_customers(None)
    table = df.head(10).to_html(classes="table table-bordered")
    plot3d = pio.to_html(fig3d, full_html=False)
    plot2d = pio.to_html(fig2d, full_html=False)
    return render_template("index.html", tables=table, plot3d=plot3d, plot2d=plot2d)

@app.route("/histograms")
def histograms():
    figs = create_histograms(None)
    hist_html = [pio.to_html(fig, full_html=False) for fig in figs]
    return render_template("index.html", histograms=hist_html)

@app.route("/download")
def download():
    df, _, _ = cluster_customers(None)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), download_name="ket_qua.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=8080)

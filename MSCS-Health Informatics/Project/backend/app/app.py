# app/app.py

from flask import Flask, render_template
app = Flask(__name__)
import views

@app.route('/')
def hello_world():
    return 'Hey, we have Flask in a Docker container!'

@app.route('/dashboard-stub')
def dashboard_stub():
    return render_template("d3-example.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=57366)

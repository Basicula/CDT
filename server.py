from flask import Flask, render_template, url_for, request, Response
import json

from delaunay_triangulation import *

app = Flask(__name__)
cdt = CDT()

@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/add_point", methods=["PUT"])
def add_point():
    form = request.form
    cdt.add_points([[float(form["x"]), float(form["y"])]])
    return Response(status=200)

@app.route("/add_constraint", methods=["PUT"])
def add_constraint():
    form = request.form
    constraint = json.loads(form["constraint"])
    if bool(form["closed"]):
        cdt.add_constraint_closed_region(constraint)
    else:
        cdt.add_constraint(constraint)
    return Response(status=200)

@app.route("/triangulate", methods=["POST"])
def triangulate():
    cdt.triangulate()
    result = {}
    result["triangles"] = cdt.triangles
    return result

@app.route("/clear", methods=["POST"])
def clear():
    cdt.clear()
    return Response(status=200)

if __name__ == "__main__":
    app.run(threaded=False, debug=True, port=1234)
from __future__ import print_function, division

import os
import sys
import time
import pandas as pd
sys.path.append('./util/')

from flask import Flask, url_for, render_template, request, jsonify, Response, json
from pdb import set_trace
from mar import MAR

app = Flask(__name__,static_url_path='/static')

global target
target=MAR()
global clf
clf = []

@app.route('/')
def hello():
    return render_template('hello.html')


@app.route('/load',methods=['POST'])
def load():
    global target
    file=request.form['file']
    target=target.create(file)
    pos, neg, total, recall, precision, f1 = target.get_numbers()
    return jsonify({"hasLabel": target.hasLabel, "flag": target.flag, "pos": pos, "done": pos+neg, "total": total,
                    "recall": recall, "precision": precision, "f1": f1})

# Depreciated
@app.route('/load_old',methods=['POST'])
def load_old():
    global target
    file=request.form['file']
    target.create_old(file)
    if target.last_pos==0:
        target.flag=False
    return jsonify({"flag": target.flag})

@app.route('/export',methods=['POST'])
def export():
    try:
        target.export()
        flag=True
    except:
        flag=False
    return jsonify({"flag": flag})

@app.route('/plot',methods=['POST'])
def plot():
    dir = "./static/image"
    for file in os.listdir(dir):
        os.remove(os.path.join(dir,file))
    name = target.plot()
    return jsonify({"path": name})

@app.route('/est',methods=['POST'])
def est():
    stat = request.form['stat']
    if stat == 'true':
        target.enable_est = True
    else:
        target.enable_est = False
    return jsonify({})

@app.route('/labeling',methods=['POST'])
def labeling():
    id = int(request.form['id'])
    label = request.form['label']
    target.code(id,label)
    pos, neg, total, recall, precision, f1 = target.get_numbers()
    target.save()
    return jsonify({"flag": target.flag, "pos": pos, "done": pos + neg, "total": total, "recall": recall,
                    "precision": precision, "f1": f1})

@app.route('/auto',methods=['POST'])
def auto():
    for id in request.form.values():
        if target.recall < 0.95:
            target.code(int(id),target.body["label"][int(id)])
        if target.recall > 0.9:
            target.get_numbers()
    pos, neg, total, recall, precision, f1 = target.get_numbers()
    return jsonify({"flag": target.flag, "pos": pos, "done": pos + neg, "total": total,
                    "recall": recall, "precision": precision, "f1": f1})

@app.route('/restart',methods=['POST'])
def restart():
    global target
    os.remove("./memory/"+target.name+".pickle")
    target = target.create(target.filename)
    pos, neg, total, recall, precision, f1 = target.get_numbers()
    return jsonify({"hasLabel": target.hasLabel, "flag": target.flag, "pos": pos, "done": pos + neg, "total": total,
                    "recall": recall, "precision": precision, "f1": f1})

@app.route('/train',methods=['POST'])
def train():
    global clf
    pos, neg, total, recall, precision, f1 = target.get_numbers()
    res={}
    if pos>0 or target.last_pos>0:
        uncertain_id, uncertain_prob, certain_id, certain_prob, clf = target.train(pne=True)
        res["uncertain"] = target.format(uncertain_id,uncertain_prob)
        res["certain"] = target.format(certain_id,certain_prob)
        # if target.last_pos > 0 and pos > 0:
        #     uncertain_id, uncertain_prob, certain_reuse_id, certain_reuse_prob = target.train_reuse(pne=True)
        #     res["reuse"] = target.format(certain_reuse_id, certain_reuse_prob)
    else:
        random_id = target.random()
        res["uncertain"] = target.format(random_id)
    if target.enable_est:
        res['est'] = target.est_num
    target.save()
    return jsonify(res)

@app.route('/susp',methods=['POST'])
def susp():
    res={}
    pos_id, pos_prob, neg_id, neg_prob = target.susp(clf)
    latest_id = target.latest_labeled()
    res["pos"] = target.format(pos_id,pos_prob)
    res["neg"] = target.format(neg_id,neg_prob)
    res["latest"] = target.format(latest_id)
    return jsonify(res)

@app.route('/search',methods=['POST'])
def search():
    import re
    res = {}

    query = request.form['query']
    cold_start = request.form['cold_start']

    keywords = re.sub(r'[\W_]+', ' ', query).split()

    # cold start == "query" passes query as a positive document to the model
    if cold_start == 'query':
        # transform and train model with query
        # query_line = [query, query, "yes", "yes", time.time(), 1]  # true positive
        query_line = [query, query, "no", "yes", time.time(), 0]  # false positive
        target.body = target.body.append(pd.DataFrame([query_line], columns=['Document Title', 'Abstract', 'label', 'code', 'time', 'fixed']), ignore_index=True)
        target.preprocess()
        target.save()
        res['bm25'] = []
    else:
        # chosed cold start algorithm
        getattr(target, cold_start)(keywords)
        ids, scores = getattr(target, f'{cold_start}_get')()

        res['bm25'] = target.format(ids, scores)

    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=False,use_debugger=False)

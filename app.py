from flask import Flask, Markup, request, render_template
import pickle
import numpy as np
import predict
from plotly.offline import plot
from plotly.graph_objs import Scatter

app = Flask(__name__)

@app.route('/')
def index():
        return render_template('home.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    if request.method=='POST':
        result=request.form
        prediction = predict.predict(result["yourcomment"])
        if any(i > 0.8 for i in prediction) or np.mean(prediction) > 0.4:
            conclusion = "This comment is very toxic!"
        elif any(i > 0.5 for i in prediction) or np.mean(prediction) > 0.25:
            conclusion = "This comment is somewhat toxic."
        elif any(i > 0.2 for i in prediction) or np.mean(prediction) > 0.2:
            conculsion = "This comment is slightly toxic."
        else:
            conclusion = "This comment is NOT toxic."
        my_plot_div = plot([Scatter(x=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], y = prediction, mode= 'markers', marker = dict(size= 10))], output_type='div')
    return render_template('results.html',yourinput = result["yourcomment"], prediction = prediction, conclusion = conclusion, div_placeholder=Markup(my_plot_div))



####################################################
if __name__ == '__main__':
    app.run()


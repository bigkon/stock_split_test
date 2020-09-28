from logging import getLogger
from io import BytesIO
from base64 import b64encode

from pandas import read_csv, to_datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from flask import Flask, render_template, request

from ..modeling.predictor import Predictor


L = getLogger(__name__)
pyplot.style.use('fivethirtyeight')
app = Flask(__name__, template_folder='ui')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    ctx = {}
    cols = ['date', 'open', 'close', 'low', 'high', 'volume']
    try:
        data = request.files['data']
        data = read_csv(data)
        missing_cols = sorted(set(cols) - set(data.columns.tolist()))
        if missing_cols:
            ctx['error'] = 'Missing required column(s): {}'.format(', '.join(missing_cols))
            return render_template('index.html', **ctx)
        data['date'] = to_datetime(data['date'])
        data.dropna(inplace=True)
        p = Predictor()
        predicted = p.predict(data)
        positive = data.loc[predicted == 1]
        if len(positive):
            fig, ax = pyplot.subplots(figsize=(11, 9))
            ax.plot(data['date'], data['close'])
            ax.plot(positive['date'], positive['close'], 'o', ms=10 * 2, mec='r', mfc='none', mew=2)
            pyplot.xticks(fontsize=10)
            pyplot.title('Predicted stock splits')
            pyplot.xlabel('Date')
            pyplot.ylabel('Price')
            img_file = BytesIO()
            pyplot.savefig(img_file, format='png')
            pyplot.clf()
            img_file.seek(0)
            ctx['image'] = b64encode(img_file.read()).decode()
            ctx['dates'] = positive['date'].dt.strftime('%Y-%m-%d').tolist()
        else:
            ctx['message'] = 'No possible stock splits detected'
    except Exception as e:
        ctx['error'] = str(e)
        L.exception('Error during predict method')
    return render_template('index.html', **ctx)

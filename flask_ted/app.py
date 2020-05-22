df.pkl
dict_.pkl
doc_topic_nmf.pkl
doc_topic.pkl
nmf_model.pkl
vectorizer_TF_IDF.pklfrom flask import Flask, render_template, request, url_for, redirect
from itertools import chain
import pickle
import sklearn
from sklearn.metrics import pairwise_distances
from string import Template

app = Flask(__name__)

with open('data/nmf_model.pkl', 'rb') as picklefile:
    nmf_model = pickle.load(picklefile)

with open('data/doc_topic_nmf.pkl', 'rb') as picklefile:
    doc_topic_nmf = pickle.load(picklefile)

with open('data/doc_topic.pkl', 'rb') as picklefile:
    doc_topic = pickle.load(picklefile)

with open('data/df.pkl', 'rb') as picklefile:
    df = pickle.load(picklefile)

with open('data/vectorizer_TF_IDF.pkl', 'rb') as picklefile:
    vectorizer_TF_IDF = pickle.load(picklefile)

with open('data/dict_.pkl', 'rb') as picklefile:
    df_dict = pickle.load(picklefile)

HTML_TEMPLATE = Template("""
      <h2>
        YouTube video link:
        <a href="https://www.youtube.com/watch?v=${youtube_id}">
          ${youtube_id}
        </a>
      </h2>

      <iframe src="https://www.youtube.com/embed/${youtube_id}" width="853" height="480" frameborder="0" allowfullscreen></iframe>""")


@app.route('/', methods = ["GET", "POST"])
def home():
    t = str(request.form.get('topic')) #, 'technology and robots'
    t = list(eval('["' + t + '"]'))
    vt = vectorizer_TF_IDF.transform(t)
    tt = nmf_model.transform(vt)
    dist_order = pairwise_distances(tt,doc_topic,metric='cosine').argsort()
    sorted_ind = list(chain(*dist_order.tolist()))

    with open('/Users/elena/Desktop/Metis/Project_4_Ted/Project-4-Ted/df.pkl', 'rb') as picklefile:
        df = pickle.load(picklefile)
    #df = df.dropna()
    df = df.reindex(columns=['Video ID', 'Title', 'Year', 'Positive Comments Score'])
    df = df.reindex(index=sorted_ind).head(5).sort_values('Positive Comments Score', ascending=False)
    #emb = 'https://www.youtube.com/watch?v=' + str(df['Video ID'][0])

    return render_template('index.html', recommendation=df.to_html()) #emb=emb



@app.route('/video')
def homepage():
    return """
    <h1>Watch it now!</h1>

    <iframe src="https://www.youtube.com/embed/j97WsAz3CDY" width="853" height="480" frameborder="0" allowfullscreen></iframe>
    """
#@app.route('/<name>')
#def user(name):

#    vidhtml =  HTML_TEMPLATE.substitute(youtube_id='YQHsXMglC9A')
#    return
    #return f"Hello {name}! Nothing is here!"

@app.route('/<vid>')
def videos(vid):
    return HTML_TEMPLATE.substitute(youtube_id=vid)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)


#@app.route('/posts')
#def posts():
#    return render_template('posts.html', posts = df_dict)




#@app.route("/predict2", methods = ["GET", "POST"])

#def predict2():
#    t = request.form.get('topic')
#    return render_template('predictor2.html', recommendation=t)

#@app.route('/<name>')
#def user(name):
#    return f"Hello {name}! Nothing is here!"


if __name__ == '__main__':
    app.run(debug=True)

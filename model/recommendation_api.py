from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import pickle
from flask import Flask, jsonify,request

app=Flask(__name__)
vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

df=pd.read_csv('papers.csv')


col_titlesentences = df.title.tolist()
col_id = df.id.tolist()

with open('feature_vectors.pkl', 'rb') as file:
    stored_vectors = pickle.load(file)



def cosine(test2):
    cosine_similarities = linear_kernel(test2, stored_vectors).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    related_docs_indices_list = related_docs_indices.tolist()
    IDS = []
    my_dict = {}
    for i in related_docs_indices:
        my_dict[col_id[i]] = col_titlesentences[i]
        IDS.append(col_id[i])
    result = []
    for i in related_docs_indices_list:
        result.append(col_titlesentences[i])

    return result, IDS



@app.route('/',methods=['GET','POST'])
def get_response():
    title_name = request.json.get('title_name')
    test2 = vectorizer.transform(title_name).toarray()
    result_knn, ids = cosine(test2)
    print(result_knn)
    return jsonify(result_knn)



if __name__=='__main__':
    app.run()
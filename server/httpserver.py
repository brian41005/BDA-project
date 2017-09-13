"""
Very simple HTTP server in python.
Usage::
    ./dummy-web-server.py [<port>]
Send a GET request::
    curl http://localhost
Send a HEAD request::
    curl -I http://localhost
Send a POST request::
    curl -d "foo=bar&bin=baz" http://localhost
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.externals import joblib
import keras
from sklearn import preprocessing
from urllib.parse import urlparse, parse_qs, unquote

Classification = ["world", "politics", "sport", "football", "culture", "business",
                  "lifeandstyle", "fashion", "environment", "technology", "travel"]
le = preprocessing.LabelEncoder()
le.fit(Classification)
tfidfvectorizer = joblib.load(
    '../categorical classification/tfidf_vectorizer.pkl')
model = keras.models.load_model(
    '../categorical classification/news_title_cls85.h5')


def predictclass(text):
    result = model.predict(tfidfvectorizer.transform(text).toarray())
    porb = np.argsort(result)[0][::-1]
    label = le.inverse_transform(porb)
    return label, result[0][porb]


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        url = unquote(self.path)

        title = parse_qs(urlparse(url).query)
        if title != {}:
            print(title['title'])
            label, porb = predictclass(title['title'])
            print(label[:3])
            output = ''
            for i in range(0, 11):
                output += '<tr><td>%s</td><td>%.5f</td></tr>' % (
                    label[i], porb[i])
            html = '<html><body><table border="1" style="font-size:48px;font-family:serif;" align="left">%s</table></body></html>' % (
                output)

            self.wfile.write(html.encode())

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        # Doesn't do anything with posted data
        self._set_headers()
        self.wfile.write("<html><body><h1>POST!</h1></body></html>")


def run(server_class=HTTPServer, handler_class=S, port=1234):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()

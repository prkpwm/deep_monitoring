from flask import Flask 
from flask import jsonify
import os
import json
app = Flask(__name__)

@app.route('/')
def hello_world():  
   return 'Hello World' 

@app.route('/getData')
def summary():
   file = open("../Client/test1.json", "r")
   file = file.readlines()
   return jsonify(file)

@app.route('/find')
def findFile():
   folders = '../Client'
   filename = []
   for dirpath, dirnames, files in os.walk(folders):
      dirpath = dirpath.replace('\\','/')
      for file_name in files:
         if file_name.endswith(".json"):
            filename.append(dirpath+"/"+file_name)
   Str = "["
   for filepath in filename:
      with open(filepath) as json_file:
         Str+=json_file.read()+"," 
   Str = Str[:-1]
   Str += "]"  
   return Str

if __name__ == '__main__':
   app.run(host="127.0.0.1", port=5000, debug=True)

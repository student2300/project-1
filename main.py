from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import main
import os
from typing import Dict, List, Any
from datetime import datetime
import json
import time
import pathlib
import requests
import cv2
import pytesseract 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
import git
from bs4 import BeautifulSoup
from PIL import Image
import speech_recognition as sr
import markdown
import pandas as pd


main.load_dotenv()

AI_TOKEN = os.getenv('AI_TOKEN')

PWD = os.getcwd()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

URL = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

Headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AI_TOKEN}"
}

###Functions for Tasks Start

def generate_data(email):
    os.system(f'uv run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py {email} --root ./data')
    return 0

def format_md(path='/data/format.md'):
    path = f'{PWD}/{path}'
    os.system(f'npx prettier@3.4.2 --write {path}')
    return 0

def convert_string_to_date(date_string):
    formats = [
        "%Y-%m-%d",              
        "%d-%b-%Y",              
        "%b %d, %Y",             
        "%Y/%m/%d %H:%M:%S",     
    ]
    
    for date_format in formats:
        try:
            date_object = datetime.strptime(date_string, date_format).date()
            return date_object
        except ValueError:
            continue
    
    print("Error: Unrecognized date format")
    return None

def count_days(rfile='/data/dates.txt', to='/data/dates-weekday.txt', day='monday'):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in rfile.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    day = day.lower()
    if day == 'monday':
        day = 0
    elif day == 'tuesday':
        day = 1
    elif day == 'wednesday':
        day = 2
    elif day == 'thursday':
        day = 3
    elif day == 'friday':
        day = 4
    elif day == 'saturday':
        day = 5
    elif day == 'sunday':
        day = 6
    file = open(rfile, 'r')
    dlist = []
    dobj = []
    for x in file:
        a = x.strip()
        dlist.append(a)
    file.close()
    for y in dlist:
        d_obj = convert_string_to_date(y)
        if d_obj:
            dobj.append(d_obj)
        else:
            pass
    day_counter = 0
    for z in dobj:
        w_day = z.weekday()
        if w_day == day:
            day_counter = day_counter + 1
    file2 = open(to, 'w')
    file2.write(str(day_counter))
    file2.close()
    return 0

def sort_contacts(rfile='/data/contacts.json', to='/data/contacts-sorted.json'):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in rfile.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    file = open(rfile, "r")
    content = file.read()
    file.close()

    data_lst = json.loads(content)

    sorted_data_list_ln = sorted(data_lst, key=lambda data: data['last_name'])
    sorted_data_list_fn = sorted(sorted_data_list_ln, key=lambda data: data['first_name'])
    result_list = sorted_data_list_fn


    file2 = open(to, "w")
    file2.write(json.dumps(result_list))
    file2.close()
    return 0

def recent_logs(rdir='/data/logs', to='/data/logs-recent.txt', n=10):
    rdir = f'{PWD}/{rdir}'
    to = f'{PWD}/{to}'
    if 'data' in rdir.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    file_list = os.listdir(rdir)
    file_objs =[]
    for x in file_list:
        o = {}
        path = f'{rdir}/{x}'
        if os.path.isfile(path):
            o['path'] = path
            o['name'] = x
            o['ctime'] = datetime.strptime(time.ctime(os.path.getctime(path)), "%a %b %d %H:%M:%S %Y")
            o['mtime'] = datetime.strptime(time.ctime(os.path.getmtime(path)), "%a %b %d %H:%M:%S %Y")
            file_objs.append(o)
        elif os.path.isdir(path):
            file_list.append(os.listdir(path))
    sorted_by_mtime = sorted(file_objs, key=lambda mt:mt['mtime'], reverse=True)
    most_recent = sorted_by_mtime[:n]
    file2 = open(to, 'a')
    for x in most_recent:
        f = open(x['path'], 'r')
        l = f.readline()
        f.close()
        file2.write(l)
    file2.close()
    return 0

def index_docs(rdir='/data/docs', to='/data/docs/index.json'):
    rdir = f'{PWD}/{rdir}'
    to = f'{PWD}/{to}'
    if 'data' in rdir.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    docs_dirs = os.listdir(rdir)
    all_files_dirs = []
    for x in docs_dirs:
        path = f'{rdir}/{x}'
        if os.path.isdir(path):
            fl = os.listdir(path)
            for y in fl:
                if os.path.isfile(f'{path}/{y}'):
                    all_files_dirs.append(f'{path}/{y}')
                elif os.path.isdir(f'{path}/{y}'):
                    docs_dirs.append(f'{x}/{y}')
        elif os.path.isfile(path):
            all_files_dirs.append(path)
    for x in all_files_dirs:
        ext = pathlib.Path(x).suffix
        if ext != '.md':
            all_files_dirs.remove(x)
    object_list = []
    for x in all_files_dirs:
        file = open(x, "r")
        for y in file:
            if y[0] == '#':
                o = {}
                key = x.split('/')[-1]
                value = y[2:-1]
                o[key] = value
                object_list.append(o)
                break
        file.close()
    f_object = {}
    for y in object_list:
        f_object.update(y)
    f2 = open(to, 'w')
    f2.write(json.dumps(f_object))
    f2.close()
    return 0

def extract_email(rfile='/data/email.txt', to='/data/email-sender.txt'):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in rfile.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    file = open(rfile, 'r')
    Content = file.read()
    file.close()
    response = requests.post(
        URL,
        headers=Headers,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role":"system", "content": "Extract the Senders email from the message. Return only the email"},
                {"role": "user", "content": Content}
            ]
        }
    )
    resJ = response.json()
    sender_email = resJ['choices'][0]['message']['content']
    file2 = open(to, 'w')
    file2.write(str(sender_email))
    file2.close()
    return 0

def extract_image_data(rfile='/data/credit-card.png', to='/data/credit-card.txt'):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in rfile.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    image = cv2.imread(rfile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 6")
    sp = text.split()
    text_parts = []
    for x in sp:
        if x != 'VALID':
           text_parts.append(x)
        else:
            break
    for i,x in enumerate(text_parts):
        if i == 0:
            cc_number = x
        else:
            cc_number = str(cc_number) + str(x)
    file2 = open(to, 'w')
    file2.write(cc_number)
    file2.close()
    return 0
 
def comments_similarity(rfile='/data/comments.txt', to='/data/comments-similar.txt', n=3):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in rfile.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    sentences=[]
    file = open(rfile,'r')
    for x in file:
        sentences.append(x)
    file.close()
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    upper_triangle_indices = np.triu_indices(similarity_matrix.shape[0], k=1)
    similar_pairs = [
        (i, j, similarity_matrix[i, j]) 
        for i, j in zip(*upper_triangle_indices)
    ]
    top_similar_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:n]
    result = []
    for pair in top_similar_pairs:
        sentence1 = sentences[pair[0]]
        sentence2 = sentences[pair[1]]
        similarity_score = pair[2]
        result.append({"sentence1": sentence1, "sentence2": sentence2, "similarity_score": similarity_score})
    file2 = open(to, 'a')
    for x in result:
        file2.write(str(x['sentence1'] + x['sentence2']))
    file2.close()
    return 0   

def db_interact(rdb, to, query):
    rdb = f'{PWD}/{rdb}'
    to = f'{PWD}/{to}'
    if 'data' in rdb.split('/'):
        pass
    else:
        return 1
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    connection = sqlite3.connect(rdb)
    cursor = connection.cursor()
    result = cursor.execute(query)
    file2 = open(to, 'w')
    for x in cursor.fetchall():
        file2.write(str(x))
    file2.close()
    return 0


### Functions for Tasks End

### Functions for Business Taks Start

def get_api_res(url, to, heads={"Content-Type": "application/josn"}):
    to = f'{PWD}/{to}'
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    response = requests.get(url, headers=heads)
    resJ = response.json()
    file = open(to, 'w')
    file.write(json.dumps(resJ))
    file.close()
    return 0

def handle_git_operations(url, to):
    to = f'{PWD}/{to}'
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    try:
        repo_url = url
        git.Repo.clone_from(repo_url, to)
    except Exception as e:
        print(e)
        return 1
    try:
        repo = git.Repo(to)
        readme = open(f'{to}/README.md', 'a')
        readme.write(f'It\'s a modifiction by python file at {datetime.now()}.\n')
        readme.close()
        repo.git.add(A=True)
        cmessage = 'Commit by Python File'
        repo.index.commit(cmessage)
    except Exception as e:
        print(e)
        return 1        
    return 0

def query_db(rdb, to, query):
    res = db_interact(rdb, to, query)
    return res

def scrap_web(url, to, redata):
    to = f'{PWD}/{to}'
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    reqdata = []
    for tag in redata['tags']:
        temp = soup.find_all(tag)
        reqdata.append(temp)
    for classes in redata['classes']:
        temp = soup.find_all(class_=classes)
        reqdata.append(temp)
    for ids in redata['ids']:
        temp = soup.find(id=ids)
        reqdata.append(temp)
    file = open(to, 'w')
    file.write('')
    file.close()
    file = open(to, 'a')
    for x in reqdata:
        if isinstance(x, list):
            for y in x:
                file.write(f'{y.get_text()}\n')
        else:
            file.write(f'{x.get_text()}\n')
    file.close()
    return 0

def compress_or_resize_image(image, action: str, to):
    image = f'{PWD}/{image}'
    to = f'{PWD}/{to}'
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    if action.lower() == 'resize':
        with Image.open(image) as img:
            size = img.size
            new_size = (size[0] // 2, size[1] // 2)
            resized_img = img.resize(new_size)
            resized_img.save(f'{to}')
    elif action.lower() == 'compress':
        with Image.open(image) as img:
            img.save(f'{to}', quality=50)
    return 0

def transcribe_audio(rfile, to):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    recognizer = sr.Recognizer()
    audio_file_path = rfile
    with sr.AudioFile(audio_file_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)
    file = open(to, 'w')
    try:
        transcription = recognizer.recognize_google(audio_data)
        file.write(transcription)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    file.close()
    return 0

def md_to_html(rfile, to):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    md_file = open(rfile, 'r')
    html_file = open(to, 'w')
    html_file.write(markdown.markdown(md_file.read()))
    md_file.close()
    html_file.close()
    return 0

def filter_csv_to_json(rfile, column, value, to):
    rfile = f'{PWD}/{rfile}'
    to = f'{PWD}/{to}'
    if 'data' in to.split('/'):
        pass
    else:
        return 1
    data = pd.read_csv(rfile)
    if column not in data.columns:
        return json.dumps({"error": f"Column '{column}' does not exist."}), 400
    filtered_data = data[data[column] == value]
    result = filtered_data.to_dict(orient='records')
    resJ = json.dumps(result, indent=2)
    file = open(to, 'w')
    file.write(resJ)
    file.close()
    return 0

### Functions for Business Taks End

### Tool List for Handling Operations Start

tool_listA = [
    {
        "type": "function",
        "function": {
            "name": "generate_data",
            "description": "Generate The Main Data",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Email Address"
                    }
                },
                "required": ["email"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_md",
            "description": "format the Markdown file Specified",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the markdown file"
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_days",
            "description": "Count the number of a specific weekday in a list of dates",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Path to file containing dates"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the destination file to write the number of days to"
                    },
                    "day": {
                        "type": "string",
                        "description": "The day of the week the user requested"
                    }
                },
                "required": ["rfile", "to", "day"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort the list of contacts in json",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Path to the json file containing contacts"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file where, the sorted contacts list is to be saved"
                    }
                },
                "required": ["rfile", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recent_logs",
            "description": "Get the top line from the most recent log files",
            "parameters": {
                "type": "object",
                "properties": {
                    "rdir": {
                        "type": "string",
                        "description": "Path to directory containing the log files"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file where the output logs are to be saved"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of most recent logs to saves"
                    }
                },
                "required": ["rdir", "to", "n"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "index_docs",
            "description": "Create an index for a list of documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "rdir": {
                        "type": "string",
                        "description": "Path to documents directory"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file where the index file is to be saved"
                    }
                },
                "required": ["rdir", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email",
            "description": "Extract the sender's email address from a message saved in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Path to file containing email"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file where the sender's email is to be saved"
                    }
                },
                "required": ["rfile", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_image_data",
            "description": "Extract the text from an Image",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Path to the image"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file where the extracted text is to be saved"
                    }
                },
                "required": ["rfile", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "comments_similarity",
            "description": "Find most similar pairs of sentences using embeddings",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Path to file containing sentences"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file where most similar pairs are to be saved"
                    }
                },
                "required": ["rfile", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_interact",
            "description": "Fetch some data from the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "rdb": {
                        "type": "string",
                        "description": "Path to database"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file where the index file is to be saved"
                    },
                    "query": {
                        "type": "string",
                        "descrioption": "SQL command to be given to the database to get the desired result"
                    }
                },
                "required": ["rdb", "to", "query"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


###Tool List for Handling Operatoins End

### Tool List for Handling Business Tasks Start

tool_listB = [
    {
        "type": "function",
        "function": {
            "name": "get_api_res",
            "description": "Fetch an API response",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "url of the api"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the json file to save the fetched data"
                    }
                },
                "required": ["url", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_git_operations",
            "description": "Clone and Commit git repositories",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "url of the repo to clone"
                    },
                    "to": {
                        "type": "string",
                        "description": "Directory to save the clone of the repository to save to"
                    }
                },
                "required": ["url", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "query a database about something",
            "parameters": {
                "type": "object",
                "properties": {
                    "rdb": {
                        "type": "string",
                        "description": "path to the database"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file to save the query to"
                    },
                    "query": {
                        "type": "string",
                        "description" : "SQL command to be run"
                    }
                },
                "required": ["rdb", "to", "query"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrap_web",
            "description": "Scrap The data from a specific webpage",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "url of the webpage"
                    },
                    "to": {
                        "type":"string",
                        "description": "Destination file to save the output to"
                    },
                    "redata": {
                        "type": "object",
                        "description": "Object Containing all the data that is to be scraped",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "description": "An array containing all the requested HTML tags",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "classes": {
                                "type": "array",
                                "description": "An Array Containing all class names to be scraped from",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "ids": {
                                "type": "array",
                                "description": "An Array containing all the id of the html elements to scrape from",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["tags", "classes", "ids"],
                        "additionalProperties": False
                    }
                },
                "required": ["url", "to", "redata"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compress_or_resize_image",
            "description": "Compress or Resize the image specified",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Path to the Image to be modified"
                    },
                    "action": {
                        "type": "string",
                        "decription": "Whether to compress or resize. It can only be either 'compress' or 'resize' "
                    },
                    "to": {
                        "type": "string",
                        "description": "Paht to the file, where the output file is to be saved"
                    }
                },
                "required": ["image", "action", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe the Audio file",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Audio file to transcribe"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the file to save the transcripted text to"
                    }
                },
                "required": ["rfile", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },{
        "type": "function",
        "function": {
            "name": "md_to_html",
            "description": "Convert the Markdown file to HTML",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Path to the Markdown File"
                    },
                    "to": {
                        "type": "string",
                        "description": "Path to the HTML output file"
                    }
                },
                "required": ["rfile", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_csv_to_json",
            "description": "filter csv data to json",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfile": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to Filter"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to filter"
                    },
                    "to" : {
                        "type": "string",
                        "description": "Path to the file to save the json output to"
                    }
                },
                "required": ["rfile", "column", "value", "to"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

### Tool List for Handling Business Tasks End

### askLLM Func

def ask_LLM(query: str, tools: list[Dict[str, Any]]):
    response = requests.post(
        URL,
        headers=Headers,
        json={
            "model": "gpt-4o-mini",
            "messages":[{"role":"user","content":query}],
            "tools": tools,
            "tool_choice": "auto"
        }
    )
    resj = response.json()
    return resj['choices'][0]['message']['tool_calls']

### run_LLM_func


def run_LLM_func(calls):
    results = []
    for x in calls:
        o = {}
        fn = x['function']['name']
        args = json.loads(x['function']['arguments'])
        if fn == 'generate_data':
            result = generate_data(email=args['email'])
        elif fn == 'format_md':
            result = format_md(path=args['path'])
        elif fn == 'count_days':
            result = count_days(rfile=args['rfile'], to=args['to'], day=args['day'])
        elif fn == 'sort_contacts':
            result = sort_contacts(rfile=args['rfile'], to=args['to'])
        elif fn == 'recent_logs':
            result = recent_logs(rdir=args['rdir'], to=args['to'], n=args['n'])
        elif fn == 'index_docs':
            result = index_docs(rdir=args['rdir'], to=args['to'])
        elif fn == 'extract_email':
            result = extract_email(rfile=args['rfile'], to=args['to'])
        elif fn == 'extract_image_data':
            result = extract_image_data(rfile=args['rfile'], to=args['to'])
        elif fn == 'comments_similarity':
            result = comments_similarity(rfile=args['rfile'], to=args['to'])
        elif fn == 'db_interact':
            result = db_interact(rdb=args['rdb'], to=args['to'], query=args['query'])
        elif fn == 'get_api_res':
            result = get_api_res(url=args['url'], to=args['to'])
        elif fn == 'handle_git_operations':
            result = handle_git_operations(url=args['url'], to=args['to'])
        elif fn == 'query_db':
            result = query_db(rdb=args['rdb'], to=args['to'], query=args['query'])
        elif fn == 'scrap_web':
            result = scrap_web(url=args['url'], to=args['to'], redata=args['redata'])
        elif fn == 'compress_or_resize_image':
            result = compress_or_resize_image(image=args['image'], action=args['action'], to=args['to'])
        elif fn == 'transcribe_audio':
            result = transcribe_audio(rfile=args['rfile'], to=args['to'])
        elif fn == 'md_to_html':
            result = md_to_html(rfile=args['rfile'], to=args['to'])
        elif fn == 'filter_csv_to_json':
            result = filter_csv_to_json(rfile=args['rfile'], column=args['column'], value=args['value'], to=args['to'])
        o['function'] = fn
        o['return'] = result
        results.append(o)
    out = 0
    for x in results:
        if x['return'] != 0:
            out = 1
    return out

tool_list = []
for x in tool_listA:
    tool_list.append(x)
for x in tool_listB:
    tool_list.append(x)

@app.post('/run')
async def run(task: str):
    resJ = ask_LLM(task, tool_list)
    result = run_LLM_func(resJ)
    if result == 0:
        return 200
    else:
        return 500


@app.get('/read')
async def read(path: str):
    file = f'{PWD}/{path}'
    if os.path.isfile(file):
        ofile = open(file, 'r')
        content = ofile.read()
        ofile.close()
        return {"file Content": content}
    else:
        return 404
    
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
import nltk
import string
import pandas as pd
import nlp_utils as nu
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_distances
from nltk.stem import WordNetLemmatizer
import enchant
print("a")
from pymongo import MongoClient

# Replace with your connection string
connection_string = "mongodb+srv://asmitajainshirley:stnley1@cluster0.f3wcvmt.mongodb.net/?retryWrites=true&w=majority"

# Connect to the MongoDB Atlas cluster
client = MongoClient(connection_string)

# Access a specific database
db = client.Chatbot

# Access a specific collection
collection = db.Chatbot_dataset

# Retrieve all documents from the collection
cursor = collection.find()

# Convert documents to a list and create a DataFrame
data = [{'Question': document['Question'], 'Answer': document['Answer']} for document in cursor]
df = pd.DataFrame(data)

# Optional: Specify the columns you want to keep
#df = df[['Question', 'Answer']]

# Print the DataFrame
#print(df)

import contractions

text = '''I'll be there within 5 min. Shouldn't you be there too?
          I'd love to see u there my dear. It's awesome to meet new friends.
          We've been waiting for this day for so long.
        '''

expanded_words = []
for word in text.split():
    expanded_words.append(contractions.fix(word))

expanded_text = ' '.join(expanded_words)
print('Original text: ' + text)
print('Expanded_text: ' + expanded_text)


def correction(input_text):

    # Define custom shorthand rules
    shorthand_map = {
        "r": "are", "u": "you", "ur": "your", "n": "and", "b": "be", "4": "for", "2": "to", "m": "am",
        "clc": "college level counselling",
        "1st": "first", "2nd": "second", "resv": "reservation", "grt": "great", "cse": "computer science engineering",
        "mech": "mechanical engineering",
        "me": "mechanical engineering", "cs": "computer science engineering", "ce": "civil engineering",
        "ch": "chemical engineering",
        "elec": "electrical engineering", "ee": "electrical engineering",
        "ec": "electronics and communication engineering", "ece": "electronics and communication engineering",
        "el": "electronics and communication engineering", "et": "electronics and telecommunication engineering",
        "ete": "electronics and telecommunication engineering",
        "cm": "chemical engineering", "it": "information technology", "au": "automobile engineering",
        "air": "artificial intelligence and robotics engineering",
        "mac": "mathematics and computing engineering", "eeiot": "internet of things engineering by electrical dept ",
        "(ee)iot": "internet of things engineering by electrical dept",
        "iot(ee)": "internet of things engineering by electrical dept",
        "iot(it)": "internet of things engineering by it dept", "itiot": "internet of things engineering by it dept",
        "dept": "department", "aids": "artificial intelligence and data science",
        "aiml": "artificial intelligence and machine learning",
        "clg": "college", "chnge": "change", "y": "why", "prof": "professor", "profs": "professors", "pkg": "package",
        "hon": "Honors", "credit": "Credit",
        "MATH": "Mathematics", "SCI": "Science", "chem": "chemical engineering", "engg": "engineering",
        "eng": "engineering", "plcmnt": "placement", "engl": " english", "lib": "library",
        "hstl": "hostel", "net": "internet", "+ve": "positive", "-ve": "negative", "tech": "technology",
        "stdnt": "student", "frm": "from", "dsa": "data structures and algorithm",
        "diff": "difference", "cgpa": "cgpa", "doc": "document", "docs": "documents", "dox": "documents",
        "admsn": "admission", "bcs": "because",
        "bcz": "because", "cz": "because", "bc": "because", "sih": "smart india hackathon", "wht": "what",
        "whut": "what", "whr": "where",
        "envrnmnt": "environment", "lst": "last", "rq": "require", "cllg": "college", "collage": "college",
        "councelling": "counselling", "counselling": "counselling", "counseling": "counselling",
        "counceling": "counselling",

    }

    # Tokenize the text input and replace shorthand words with full-forms

    input_tokens = input_text.split()

    for i, token in enumerate(input_tokens):
        if token.lower() in shorthand_map:
            input_tokens[i] = shorthand_map[token.lower()]

    processed_input = " ".join(input_tokens)

    return processed_input

ans =correction("r")
print(ans)

lemmatizer = WordNetLemmatizer()


# answer= chatbot_data.find("answer": {})
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
factors = tfidf.fit_transform(df['Question']).toarray() #make df of questions only!
#here, fetch column 'Question' and vectorise
tfidf.get_feature_names_out()

def chatbot(query):
    query = lemmatizer.lemmatize(query)

    query_vector = tfidf.transform([query]).toarray()

    similar_score = 1 - cosine_distances(factors, query_vector)  # df
    index = similar_score.argmax()

    matching_question = df.loc[index]['Question']  # query to return the question with most matching score
    Answer = df.loc[index]['Answer']  # query to return the answer of that index
    # pos_score = df.loc[index]['pos']
    # neg_score = df.loc[index]['neg']
    confidence = similar_score[index][0]
    chat_dict = {
        'Answer': Answer,  # answer
        'score': confidence,
    }
    return chat_dict


def chatbot_response(user):
    import random

    conversation_starters = [
        "Hi there, how can I help you today?",
        "What brings you here today?",
        "Hello! What can I do for you?",
        "How are you doing today?",
        "Nice to see you! What can I assist you with?"
    ]

    # greeting=['hi','hello','hey','hey there!','hello stnley']
    # give instructions- 1. for exiting 2. ask relevant question and try to be grammatically corect format.
    3  ##print("INSTRUCTIONS:\n 1. write exit for exiting \n2. Please ask relevant questions ONLY.")
    flag = True
    ##print(" StnLey : my name is stnley.lets have some conservation !")
    while (flag == True):
       # user = input("you: ")
        user = user.lower()
        user = correction(user)
        user = contractions.fix(user)

        #
        if user in ["hii", "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "Howdy", "Greetings",
                    "What's up?", "yo", "hi there", "Hey there", "hi stnley",
                    "hola", "bonjour", "buongiorno", "hallo", "guten Tag", "ciao", "namaste", "hello stnley",
                    "hi stnley",
                    "hey stnley", "good morning stnley",
                    "good afternoon stnley", "good evening stnley", "Howdy stnley", "Greetings", "What's up stnley?",
                    "Yo stnley", "Hi there stnley",
                    "Hey there stnley", "hi stnley ", "hola stnley", "bonjour stnley", "buongiorno stnley",
                    "hallo stnley",
                    "ciao stnley", "namaste stnley", "heyy", "hayo"]:
            #flag = True
            return random.choice(conversation_starters)

        elif (user == "bye" or user == "exit"):
            flag = False
            break

        else:
            response = chatbot(user)

            if (response['score'] > 0.3):

                #flag = True  # fetching answer
                ans= response['Answer']
                string="\n for more information you can contact :- Admission related enquirDr. Manish Dixit/Ms. Jyotsana Singh Mob.: 9343250503, 9425460166  \n ims related query --Shri. Atul Chauhan,07512409304  ims@mitsgwalior.in \n visit website -mitsgwalior.in/contactus.php "
                string1=ans + string
                return ans

            else:

                return "StnLey : Please rephrase your Question \n or you can contact :- \n Admission related enquirDr. Manish Dixit/Ms. Jyotsana Singh Mob.: 9343250503, 9425460166  \n ims related query --Shri. Atul Chauhan,07512409304  ims@mitsgwalior.in \n visit website -mitsgwalior.in/contactus.php "
                # change1 260723
                new_question = {'Question': user, 'Answer': 'Sorry, I do not have an answer to this question yet.'}
                collection.insert_one(new_question)
                # collection.insert
                #flag = True


from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if(userText=='exit'):
        return exit()
    else:
        return chatbot_response(userText)
if __name__ == "__main__":
    app.run()


# HomeAutomation
Home Automation using Chatbot to operate devices


import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import speech_recognition as sr
import pyttsx3
import smtplib

r = sr.Recognizer()
engine = pyttsx3.init()

with open("intents.json") as file:
    data = json.load(file)

print(data)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)
    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8);
net = tflearn.fully_connected(net, 8);
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type stop to quit)")
    engine.say("Start talking with the bot speak stop or ruk ja or bye to quit")
    engine.runAndWait()
    while True:
        with sr.Microphone() as source:
            print("say something")
            engine.say("Tell me, what next to do ?")
            engine.runAndWait()
            audio = r.record(source, duration=5)
            print("Audio recorded")
        try:
            print("Text : " + r.recognize_google(audio))
            inp = r.recognize_google(audio)
        except:
            print("failed")
            engine.say("Failed to recognize, come back again")
            continue
        #inp = input("You :")
        if inp.lower() == "stop" or inp.lower() == "bye" or inp.lower() == "ruk ja":
            break

        if inp.lower() == "send a email" or inp.lower() == "send an email" or inp.lower() == "send a mail" or inp.lower() == "send an mail":
            with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()

                smtp.login('manish9183859@gmail.com', '0M@NIShji')
                with sr.Microphone  () as source:
                    while True:
                        print("Enter subject")
                        engine.say("What is the subject?")
                        engine.runAndWait()
                        audio = r.record(source, duration=5)
                        print("Audio recorded")
                        try:
                            subject = r.recognize_google(audio)
                            print(subject)
                            engine.say(subject)
                            engine.runAndWait()
                            break

                        except:
                            engine.say("failed to listen subject")
                            engine.runAndWait()
                            continue
                    while True:
                        print("Enter msg to be send")
                        engine.say(("Speak the message to be send"))
                        engine.runAndWait()
                        audio = r.record(source, duration=5)
                        try:
                            body = r.recognize_google(audio)
                            print(body)
                            engine.say(body)
                            engine.runAndWait()
                            break
                        except:
                            engine.say("Failed to listen body")
                            engine.runAndWait()
                            continue

                    msg = f'subject : {subject}\n\n{body}'

                    smtp.sendmail('manish9183859@gmail.com', 'ntops123@gmail.com', msg)

        #if inp.lower() != "stop" and inp.lower() != "bye" and inp.lower() != "send a email" and inp.lower() != "send an email" and inp.lower() != "send a mail" and  inp.lower() != "send an mail":
        else:
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
                #print(tag)
            if results[results_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                        output = str(random.choice(responses))
                        print(output)
                        engine.say(output)
                        engine.runAndWait()
                #except:
                #   print("Didn't understand")
                #    pass;
            else:
                print("I didn't get that, Try again")
                engine.say("I didn't get that, Try again")
                engine.runAndWait()
chat()





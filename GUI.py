from joblib import load
from tkinter import *
from tkinter import messagebox
from preprocess import preprocessor


def predict():
	user_input = [text.get("1.0", END)]

	doc = preprocessor(user_input)
	Y = vectorizer.transform([doc])
	prediction = loaded_model.predict(Y)
	if prediction[0] == 0:
		answer = "Science"
	elif prediction[0] == 1:
		answer = "Sports"
	else:
		answer = "Business"
	messagebox.showinfo("Prediction", message=answer)


# ____________________ MODELS__________________________ #
vectorizer = load('vectorizer.joblib')
loaded_model = load('cluster_model.joblib')
# _____________________GUI________________________________________ #

window = Tk()
window.title("Article Topic Predictor")
window.geometry("1200x660")


frame = Frame(window)
frame.pack(pady=5)

text_scroll = Scrollbar(frame)
text_scroll.pack(side=RIGHT, fill=Y)

text = Text(frame, width=97, height=25, font=("Helvetica", 16), selectbackground="yellow", selectforeground="black", undo=True, yscrollcommand=text_scroll.set)
text.pack()

text_scroll.config(command=text.yview())

predict_button = Button(text="Predict", command=predict)
predict_button.pack()


window.mainloop()
from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='../templates')

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    ova = float(request.form['ova'])

    value_per_age = ova/age

    prediction = model.predict([[age, ova, value_per_age]])

    euro_value = prediction[0] * 1000000
    rupees = euro_value * 90

    output = int(rupees)

    return render_template("index.html",
                           prediction_text="Predicted Transfer Value : ₹{:,.0f}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
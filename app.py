from flask import Flask, render_template, request
import pickle
import pandas as pd 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
Mcle = pickle.load(open('mcle.pkl', 'rb'))

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/about")
def home():
    return render_template('about.html')

@app.route("/predict")
def home1():
    return render_template('predict.html')

@app.route("/submit")
def home2():
    return render_template('submit.html')

@app.route("/pred", methods=['POST'])
def predict():
    quarter = request.form['quarter'] ## Querter 1, Quarter 2, Quarter 3, Quarter 4, Quarter 5
    department = request.form['department'] ## sweing , finishing
    day = request.form['day'] ## Wednesday,Sunday,Tuesday,Thursday,Monday,Saturday $Friday(Holiday)
    team = request.form['team'] ## 1,2,3,4,5,6,7,8,9,10,11,12
    targeted_productivity = request.form['targeted_productivity'] ## This is numerical Data
    smv = request.form['smv'] ## This is numerical Data | range 0-100
    over_time = request.form['over_time'] ## This is numerical Data  | range 0-26000
    incentive = request.form['incentive'] ## This is numerical Data  | range 0-36000
    idle_time = request.form['idle_time'] ## This is numerical Data  | range 0-300
    idle_men = request.form['idle_men'] ## This is numerical Data   | range 0-100
    no_of_style_change = request.form['no_of_style_change'] ## This is numerical Data | range 0-10
    no_of_workers = request.form['no_of_workers'] ## This is numerical Data | range 0-100
    month = request.form['month'] ## 1,2,3 (January, February, March) only dataset are available for these months

    # Prepare data for prediction
    data = {
        'quarter': [quarter.strip().lower()],
        'department': [department.strip().lower()],
        'day': [day.strip().lower()],
        'team': [team.strip()],
        'targeted_productivity': [float(targeted_productivity.strip())],
        'smv': [float(smv.strip())],
        'over_time': [int(over_time.strip())],
        'incentive': [int(incentive.strip())],
        'idle_time': [float(idle_time.strip())],
        'idle_men': [int(idle_men.strip())],
        'no_of_style_change': [int(no_of_style_change.strip())],
        'no_of_workers': [int(no_of_workers.strip())],
        'month': [int(month.strip())]
    }
    print(data)
    example_df = pd.DataFrame(data)
    # # Encode the data
    example_encoded = Mcle.transform(example_df)
    predicted_productivity_example = model.predict(example_encoded)
    print(predicted_productivity_example)

    # prediction = model.predict(total)
    prediction = predicted_productivity_example[0]
    print(prediction)
    if prediction <= 0.3:
        text = 'The employee is averagely productive.'
    elif 0.3 < prediction <= 0.8:
        text = 'The employee is medium productive'
    else:
        text = 'The employee is highly productive'
    return render_template('submit.html', prediction_text=text,score=prediction)

if __name__ == '__main__':
    app.run(debug=False)

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        age = float(request.form['age'])
        annual_income = float(request.form['annual_income'])
        monthly_inhand_salary = float(request.form['monthly_inhand_salary'])
        num_credit_card = int(request.form['num_credit_card'])
        num_of_loan = int(request.form['num_of_loan'])
        delay_from_due_date = float(request.form['delay_from_due_date'])
        num_of_delayed_payment = int(request.form['num_of_delayed_payment'])
        changed_credit_limit = float(request.form['changed_credit_limit'])
        num_credit_inquiries = int(request.form['num_credit_inquiries'])
        credit_mix = float(request.form['credit_mix'])
        outstanding_debt = float(request.form['outstanding_debt'])
        payment_of_min_amount = float(request.form['payment_of_min_amount'])
        total_emi_per_month = float(request.form['total_emi_per_month'])
        amount_invested_monthly = float(request.form['amount_invested_monthly'])
        payment_behaviour = float(request.form['payment_behaviour'])
        monthly_balance = float(request.form['monthly_balance'])
        
        # Make prediction using the model
        prediction = model.predict([[age, annual_income, monthly_inhand_salary, num_credit_card, num_of_loan, delay_from_due_date, 
                                     num_of_delayed_payment, changed_credit_limit, num_credit_inquiries, credit_mix, 
                                     outstanding_debt, payment_of_min_amount, total_emi_per_month, amount_invested_monthly, 
                                     payment_behaviour, monthly_balance]])[0] 
        
        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

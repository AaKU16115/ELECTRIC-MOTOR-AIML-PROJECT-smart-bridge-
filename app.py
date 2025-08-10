from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import random


app = Flask(__name__)


# Load model and scaler if available
model_path = "best_model.save"
scaler_path = "scaler.save"
model = None
scaler =None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model and scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")
else:
    print("⚠ Model/scaler not found. Using sample output.")
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/manual')
def manual():
    return render_template('Manual_predict.html', prediction=None)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        ambient = float(request.form['ambient'])
        coolant = float(request.form['coolant'])
        u_d = float(request.form['u_d'])
        u_q = float(request.form['u_q'])
        motor_speed = float(request.form['motor_speed'])
        torque = float(request.form['torque'])
        i_d = float(request.form['i_d'])

        features = np.array([[ambient, coolant, u_d, u_q, motor_speed, torque, i_d]])

        if model and scaler:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            result = round(prediction, 2)
        else:    result = round(random.uniform(88, 98), 2)

        return render_template(
            'Manual_predict.html',
            prediction=result,
            ambient=ambient,
            coolant=coolant,
            u_d=u_d,
            u_q=u_q,
            motor_speed=motor_speed,
            torque=torque,
            i_d=i_d
            
        )
    except Exception as e:
        return render_template('Manual_predict.html', prediction=f"Error: {str(e)}")
    
@app.route('/sensor')
def sensor():
    return render_template('sensor_predict.html', prediction=None)


@app.route('/predict_sensor', methods=['POST'])
def predict_sensor():
        # Just grab form values to display, no model involved
        inputs = request.form.to_dict()
        result = "Sensor prediction feature coming soon!"  # placeholder
        
        return render_template('sensor_predict.html', prediction=result, inputs=inputs)


@app.route('/project_info')
def project_info():
    return render_template('project_info.html')

# After all your routes are defined

print("\nAll registered routes:")
for rule in app.url_map.iter_rules():
    print(rule.endpoint, rule)
print()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


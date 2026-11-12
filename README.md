# 🏭 Bayesian Probability for Defect Prediction

A manufacturing defect prediction system that uses Bayesian probability (a Naive Bayes classifier) to predict whether a product is **Defective** or **Non-Defective** from machine readings.

## How it works

```
Temperature, Pressure, Vibration  →  Bayesian model (prior → posterior)  →  Defective / Non-Defective
```

- **Prior** P(Defective): the defect rate in past inspections.
- **Likelihoods** P(reading | class): learned from the training data, with Laplace smoothing.
- **Posterior** P(Defective | readings): calculated with Bayes' theorem:

  P(A|B) = P(B|A)·P(A) / [P(B|A)·P(A) + P(B|¬A)·P(¬A)]

Example: with P(A) = 0.2, P(B|A) = 0.8 and P(B|¬A) = 0.2, the posterior is 0.16 / (0.16 + 0.16) = **50%**.

## Features

- **Prediction:** enter sensor values (°C, bar, mm/s) or categories. The app shows the prior, the posterior, the verdict and a step-by-step calculation.
- **Bayes Theorem calculator:** a single-evidence worked example you can edit.
- **Training Data & Model:** the dataset, the learned likelihoods and a leave-one-out accuracy test.
- **Inspection Log:** records predictions and exports them to CSV.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Sensor thresholds

| Parameter   | Abnormal when |
|-------------|---------------|
| Temperature | > 80 °C       |
| Pressure    | < 4 bar       |
| Vibration   | > 7 mm/s      |

## Future upgrade

Connect an ESP32 with temperature, pressure and vibration sensors to stream live readings into the model.

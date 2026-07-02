# JSW Steel ERS - AI Predictive Maintenance Portal
===================================================
An enterprise-ready AI predictive maintenance system integrated with JSW Steel's Electrical Repair Shop (ERS) application. The portal transitions maintenance workflows from reactive to condition-based diagnostics by predicting motor failure risk categories, highlighting sub-component vulnerabilities, and scheduling proactive inspections.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technical Architecture](#-technical-architecture)
3. [Feature Engineering Pipeline](#-feature-engineering-pipeline)
4. [Machine Learning Model](#-machine-learning-model)
5. [JVM-Native Deployment (ONNX)](#-jvm-native-deployment-onnx)
6. [Knowledge Transfer (KT) & Integration Guide](#-knowledge-transfer-kt--integration-guide)
7. [Installation & Local Execution](#-installation--local-execution)

---

## 🔍 Project Overview
The Electrical Repair Shop (ERS) processes motor repairs across various departments (e.g., Hot Strip Mill, Cold Rolling Mill). This project upgrades the ERS system by adding machine learning capability:
*   **Active Risk Assessment:** Predicts motor health score (0-100) and risk category (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
*   **Component-Level Diagnostics:** Evaluates probability scores for individual failure vectors (Bearing, Winding, Rotor, Insulation).
*   **Dynamic AI Inspection:** Connects operator checklist items in real-time to the XGBoost inference model to forecast failure risks before disassembly.

---

## 🛠 Technical Architecture
The portal operates on a hybrid architecture designed to easily connect to JSW's environment:
*   **Model Training (Python):** Offline model training and validation utilizing XGBoost and Scikit-Learn.
*   **Interoperability Standard (ONNX):** The trained model is exported to Open Neural Network Exchange (ONNX) format, eliminating Python dependency in production.
*   **In-Memory Inference (Java/Tomcat):** JSW's Java web server runs predictions natively via the ONNX Runtime Java API.
*   **UI Representation (Angular/Vanilla JS):** A corporate light-themed dashboard matching the current JSW ERS screens.

---

## 📊 Feature Engineering Pipeline
Raw repair requisitions are transformed into **18 engineered features** before evaluation:

| Feature Group | Feature Name | Description |
| :--- | :--- | :--- |
| **Specifications** | `kw`, `rpm`, `voltage`, `current` | Physical electrical parameters of the motor |
| **History** | `repair_count`, `days_since_last_repair` | Frequency and recency of maintenance events |
| **Fault Vectors** | `bearing_failure_count`, `winding_failure_count`, `rotor_failure_count`, `insulation_failure_count` | Cumulative count of specific past sub-component failures |
| **Aggregates** | `dept_failure_rate`, `make_failure_rate` | Risk coefficients grouped by department and manufacturer |
| **Categories** | `make_encoded`, `equipment_type_encoded`, `duty_cycle_encoded`, `application_use_encoded` | Encoded categorical flags from historical metadata |

---

## 🤖 Machine Learning Model
The system uses an **XGBoost Multiclass Classifier** optimized using soft probabilities:
*   **Accuracy:** ~76.5% on ERS test parameters.
*   **ROC-AUC:** 0.93 (indicating highly stable class separation).
*   **Explainable AI (SHAP):** Feature impact is verified using SHapley Additive exPlanations to ensure predictions align with industrial thermodynamic and mechanical wear laws.

---

## ⚡ JVM-Native Deployment (ONNX)
Rather than hosting a Flask or FastAPI Python sidecar (which introduces network overhead and infrastructure costs), the model is compiled to ONNX. 
*   **Artifact:** `models/xgboost_risk_model.onnx`
*   **Inference Latency:** < 5ms execution time.
*   **Dependency:** Runs completely inside JSW's JVM, eliminating the need to install Python or library packages in production.

---

## 🎓 Knowledge Transfer (KT) & Integration Guide
This section guides ERS engineers on how to deploy this portal live in JSW's actual ERS application.

### 1. Java Backend Integration (Tomcat)
Add the ONNX Runtime dependency to your Java project's Maven configuration (`pom.xml`):

```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.16.3</version>
</dependency>
```

Add the following class to your Java source code to run native, in-memory model inferences:

```java
package com.jsw.ers.service;

import ai.onnxruntime.*;
import java.util.*;
import java.nio.FloatBuffer;

public class PredictiveMaintenanceService {
    private OrtEnvironment env;
    private OrtSession session;

    public PredictiveMaintenanceService(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public float[] predictRisk(float[] featureArray) throws OrtException {
        // featureArray must contain the 18 engineered features in exact order
        String inputName = session.getInputNames().iterator().next();
        long[] shape = new long[]{1, 18};
        
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(featureArray), shape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
            try (OrtSession.Result results = session.run(inputs)) {
                float[][] output = (float[][]) results.get(0).getValue();
                return output[0]; // Returns [Prob_LOW, Prob_MEDIUM, Prob_HIGH, Prob_CRITICAL]
            }
        }
    }
}
```

### 2. Relational Database Mapping (SQL)
When an operator triggers a prediction request:
1.  Query the motor's specs and repair records using JDBC or Hibernate.
2.  Populate a float array of length 18 representing the features.
3.  If on the Inspection Form, increment corresponding counts in the float array (e.g. if checklist items for `R2: Bearing Seating Condition` are checked, increment feature `12` (`bearing_failure_count`) by `1.0f`).
4.  Execute `predictRisk(features)` and return the values as JSON to the UI.

### 3. Angular UI Integration
*   **For the Registry Table:** Add an `(click)="analyzeMotor(motor.serialNo)"` event binding to trigger the diagnostics details modal.
*   **For the Inspection Form:** Replicate the checklist checkboxes from the prototype HTML, map them to an array of codes, and POST them to the java endpoint on click.

---

## 🚀 Installation & Local Execution

### Prerequisites
*   Python 3.8+ (for local training/simulation)
*   Web Browser

### Local Installation
1.  Clone or copy the directory:
    ```bash
    cd c:/Users/atish/jswinternship
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project Locally
1.  **Execute the Full Pipeline (Data Gen ➔ Features ➔ Training ➔ ONNX Export):**
    ```bash
    python src/run_pipeline.py
    ```
2.  **Start the Web Portal Server:**
    ```bash
    python src/api.py
    ```
3.  **Open the Web Dashboard:**
    Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to view the portal.

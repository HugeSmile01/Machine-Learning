# 🚀 Using ML Platform Models in Your Projects

This guide shows you how to use trained models from the ML Learning Platform in your own projects.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Web Applications](#web-applications)
3. [Node.js Applications](#nodejs-applications)
4. [React Applications](#react-applications)
5. [Python Projects](#python-projects)
6. [Mobile Apps](#mobile-apps)
7. [Production Deployment](#production-deployment)

---

## Quick Start

### Step 1: Train and Export Model

1. Train your model on the ML Learning Platform
2. Click "📥 Download Model"
3. Save the model files (model.json + weights)

### Step 2: Use in Your Project

Choose your platform below and follow the integration guide.

---

## Web Applications

### Vanilla JavaScript

**Project Structure:**
```
my-project/
├── index.html
├── app.js
└── model/
    ├── model.json
    └── group1-shard1of1.bin
```

**HTML (index.html):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>My ML App</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0"></script>
</head>
<body>
    <h1>ML Prediction App</h1>
    <input type="text" id="input" placeholder="Enter values">
    <button onclick="predict()">Predict</button>
    <div id="result"></div>
    
    <script src="app.js"></script>
</body>
</html>
```

**JavaScript (app.js):**
```javascript
let model;

// Load model on page load
async function loadModel() {
    try {
        model = await tf.loadLayersModel('model/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Make prediction
async function predict() {
    if (!model) {
        alert('Model not loaded yet!');
        return;
    }
    
    // Get input
    const input = document.getElementById('input').value;
    const values = input.split(',').map(v => parseFloat(v.trim()));
    
    // Prepare tensor
    const inputTensor = tf.tensor2d([values]);
    
    // Make prediction
    const prediction = model.predict(inputTensor);
    const result = await prediction.data();
    
    // Display result
    document.getElementById('result').innerHTML = 
        `Prediction: ${result[0].toFixed(4)}`;
    
    // Clean up
    inputTensor.dispose();
    prediction.dispose();
}

// Load model when page loads
window.onload = loadModel;
```

### Using with Form Data

```javascript
async function predictFromForm() {
    // Get form values
    const feature1 = parseFloat(document.getElementById('feature1').value);
    const feature2 = parseFloat(document.getElementById('feature2').value);
    const feature3 = parseFloat(document.getElementById('feature3').value);
    
    // Create input tensor
    const inputTensor = tf.tensor2d([[feature1, feature2, feature3]]);
    
    // Predict
    const prediction = model.predict(inputTensor);
    const result = await prediction.data();
    
    // Interpret result (for classification)
    const classLabel = result[0] > 0.5 ? 'Class A' : 'Class B';
    const confidence = (result[0] > 0.5 ? result[0] : 1 - result[0]) * 100;
    
    console.log(`Predicted: ${classLabel} (${confidence.toFixed(2)}% confident)`);
    
    // Clean up
    inputTensor.dispose();
    prediction.dispose();
}
```

---

## Node.js Applications

### Installation

```bash
npm install @tensorflow/tfjs-node
# or for GPU support
npm install @tensorflow/tfjs-node-gpu
```

### Basic Usage

**server.js:**
```javascript
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Load model
let model;

async function loadModel() {
    const modelPath = 'file://' + path.join(__dirname, 'model/model.json');
    model = await tf.loadLayersModel(modelPath);
    console.log('Model loaded successfully');
}

// Prediction function
async function makePrediction(inputData) {
    const inputTensor = tf.tensor2d([inputData]);
    const prediction = model.predict(inputTensor);
    const result = await prediction.data();
    
    inputTensor.dispose();
    prediction.dispose();
    
    return Array.from(result);
}

// Example usage
async function main() {
    await loadModel();
    
    const testData = [5.1, 3.5, 1.4, 0.2]; // Example input
    const prediction = await makePrediction(testData);
    
    console.log('Prediction:', prediction);
}

main();
```

### Express API Server

**server.js:**
```javascript
const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const app = express();
app.use(express.json());

let model;

// Load model on startup
async function loadModel() {
    const modelPath = 'file://' + path.join(__dirname, 'model/model.json');
    model = await tf.loadLayersModel(modelPath);
    console.log('Model loaded successfully');
}

// Prediction endpoint
app.post('/predict', async (req, res) => {
    try {
        const { data } = req.body;
        
        if (!data || !Array.isArray(data)) {
            return res.status(400).json({ error: 'Invalid input data' });
        }
        
        const inputTensor = tf.tensor2d([data]);
        const prediction = model.predict(inputTensor);
        const result = await prediction.data();
        
        inputTensor.dispose();
        prediction.dispose();
        
        res.json({
            prediction: Array.from(result),
            input: data
        });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Batch prediction endpoint
app.post('/predict/batch', async (req, res) => {
    try {
        const { data } = req.body;
        
        if (!data || !Array.isArray(data)) {
            return res.status(400).json({ error: 'Invalid input data' });
        }
        
        const inputTensor = tf.tensor2d(data);
        const predictions = model.predict(inputTensor);
        const results = await predictions.data();
        
        inputTensor.dispose();
        predictions.dispose();
        
        res.json({
            predictions: Array.from(results),
            count: data.length
        });
    } catch (error) {
        console.error('Batch prediction error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        modelLoaded: !!model
    });
});

// Start server
const PORT = process.env.PORT || 3000;

loadModel().then(() => {
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });
});
```

**Usage:**
```bash
# Install dependencies
npm install express @tensorflow/tfjs-node

# Start server
node server.js

# Make predictions
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [5.1, 3.5, 1.4, 0.2]}'
```

---

## React Applications

### Installation

```bash
npm install @tensorflow/tfjs
```

### Component Example

**PredictionComponent.jsx:**
```javascript
import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

function PredictionComponent() {
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [input, setInput] = useState('');
    const [result, setResult] = useState(null);

    // Load model on component mount
    useEffect(() => {
        async function loadModel() {
            try {
                const loadedModel = await tf.loadLayersModel('/model/model.json');
                setModel(loadedModel);
                setLoading(false);
                console.log('Model loaded successfully');
            } catch (error) {
                console.error('Error loading model:', error);
                setLoading(false);
            }
        }
        loadModel();
    }, []);

    // Make prediction
    async function predict() {
        if (!model) return;

        try {
            // Parse input
            const values = input.split(',').map(v => parseFloat(v.trim()));
            
            // Create tensor
            const inputTensor = tf.tensor2d([values]);
            
            // Predict
            const prediction = model.predict(inputTensor);
            const data = await prediction.data();
            
            setResult(data[0]);
            
            // Clean up
            inputTensor.dispose();
            prediction.dispose();
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Error making prediction');
        }
    }

    if (loading) {
        return <div>Loading model...</div>;
    }

    return (
        <div className="prediction-component">
            <h2>ML Prediction</h2>
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter values (comma-separated)"
            />
            <button onClick={predict}>Predict</button>
            {result !== null && (
                <div className="result">
                    <h3>Result: {result.toFixed(4)}</h3>
                </div>
            )}
        </div>
    );
}

export default PredictionComponent;
```

### Custom Hook for Model Loading

**useMLModel.js:**
```javascript
import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export function useMLModel(modelPath) {
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadModel() {
            try {
                const loadedModel = await tf.loadLayersModel(modelPath);
                setModel(loadedModel);
                setLoading(false);
            } catch (err) {
                setError(err);
                setLoading(false);
            }
        }
        loadModel();
    }, [modelPath]);

    const predict = async (inputData) => {
        if (!model) throw new Error('Model not loaded');
        
        const inputTensor = tf.tensor2d([inputData]);
        const prediction = model.predict(inputTensor);
        const result = await prediction.data();
        
        inputTensor.dispose();
        prediction.dispose();
        
        return Array.from(result);
    };

    return { model, loading, error, predict };
}
```

**Usage:**
```javascript
function MyComponent() {
    const { model, loading, error, predict } = useMLModel('/model/model.json');
    
    async function handlePredict() {
        const result = await predict([1.2, 3.4, 5.6]);
        console.log('Prediction:', result);
    }
    
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error.message}</div>;
    
    return (
        <div>
            <button onClick={handlePredict}>Predict</button>
        </div>
    );
}
```

---

## Python Projects

### Converting TensorFlow.js to Python

**Installation:**
```bash
pip install tensorflowjs tensorflow
```

**Conversion Script:**
```python
import tensorflowjs as tfjs

# Convert TensorFlow.js model to Keras
model = tfjs.converters.load_keras_model('path/to/model.json')

# Save as Keras model
model.save('my_model.h5')

# Or save as SavedModel format
model.save('my_model/')
```

### Using Converted Model

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('my_model.h5')

# Make prediction
input_data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(input_data)

print(f'Prediction: {prediction[0][0]:.4f}')
```

### Flask API

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('my_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        input_data = np.array([data])
        
        prediction = model.predict(input_data)
        result = prediction[0].tolist()
        
        return jsonify({
            'prediction': result,
            'input': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=3000)
```

---

## Mobile Apps

### React Native

**Installation:**
```bash
npm install @tensorflow/tfjs @tensorflow/tfjs-react-native
npm install @react-native-async-storage/async-storage
npm install react-native-fs
```

**Usage:**
```javascript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

// Load model
async function loadModel() {
    await tf.ready();
    
    const model = await tf.loadLayersModel(
        bundleResourceIO({
            modelJSON: require('./model/model.json'),
            modelWeights: [require('./model/weights.bin')]
        })
    );
    
    return model;
}

// Make prediction
async function predict(model, inputData) {
    const inputTensor = tf.tensor2d([inputData]);
    const prediction = model.predict(inputTensor);
    const result = await prediction.data();
    
    inputTensor.dispose();
    prediction.dispose();
    
    return result[0];
}
```

---

## Production Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM node:18

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --production

# Copy model and application
COPY model/ ./model/
COPY server.js ./

EXPOSE 3000

CMD ["node", "server.js"]
```

**Build and Run:**
```bash
docker build -t ml-api .
docker run -p 3000:3000 ml-api
```

### AWS Lambda

**handler.js:**
```javascript
const tf = require('@tensorflow/tfjs-node');
let model;

exports.handler = async (event) => {
    // Load model (cached after first invocation)
    if (!model) {
        model = await tf.loadLayersModel('file://./model/model.json');
    }
    
    const data = JSON.parse(event.body).data;
    const inputTensor = tf.tensor2d([data]);
    const prediction = model.predict(inputTensor);
    const result = await prediction.data();
    
    inputTensor.dispose();
    prediction.dispose();
    
    return {
        statusCode: 200,
        body: JSON.stringify({
            prediction: Array.from(result)
        })
    };
};
```

### Heroku Deployment

**Procfile:**
```
web: node server.js
```

**Deploy:**
```bash
heroku create my-ml-app
git push heroku main
```

---

## Best Practices

### Performance Optimization

1. **Load Model Once**: Cache the model in memory
2. **Dispose Tensors**: Always dispose tensors after use
3. **Batch Predictions**: Process multiple inputs together
4. **Use WebGL**: Enable GPU acceleration in browsers
5. **Warm Up**: Make a dummy prediction after loading

### Error Handling

```javascript
async function safePrediction(model, inputData) {
    let inputTensor, prediction;
    
    try {
        // Validate input
        if (!Array.isArray(inputData)) {
            throw new Error('Input must be an array');
        }
        
        // Create tensor
        inputTensor = tf.tensor2d([inputData]);
        
        // Predict
        prediction = model.predict(inputTensor);
        const result = await prediction.data();
        
        return {
            success: true,
            result: Array.from(result)
        };
    } catch (error) {
        console.error('Prediction error:', error);
        return {
            success: false,
            error: error.message
        };
    } finally {
        // Clean up
        if (inputTensor) inputTensor.dispose();
        if (prediction) prediction.dispose();
    }
}
```

### Security Considerations

1. **Validate Inputs**: Always validate user input
2. **Rate Limiting**: Implement rate limiting for APIs
3. **CORS**: Configure CORS properly
4. **Authentication**: Add authentication for production
5. **Input Sanitization**: Sanitize all inputs

---

## Troubleshooting

### Common Issues

**Model Not Loading:**
```javascript
// Check CORS settings
// Ensure model files are served with correct content-type
// Verify file paths are correct
```

**Memory Leaks:**
```javascript
// Always dispose tensors
const tensor = tf.tensor([1, 2, 3]);
// ... use tensor ...
tensor.dispose();

// Or use tf.tidy()
const result = tf.tidy(() => {
    const tensor = tf.tensor([1, 2, 3]);
    return tensor.square();
});
```

**Performance Issues:**
```javascript
// Use batch predictions
const batchSize = 32;
for (let i = 0; i < data.length; i += batchSize) {
    const batch = data.slice(i, i + batchSize);
    const predictions = model.predict(tf.tensor2d(batch));
    // Process predictions
    predictions.dispose();
}
```

---

## Examples Repository

Check out complete examples at:
- [GitHub Examples](https://github.com/HugeSmile01/Machine-Learning/tree/main/examples)

---

**Happy Building! 🚀**

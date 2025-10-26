/**
 * ML Utilities - Comprehensive Machine Learning Helper Functions
 * Provides model management, data preprocessing, export/import, and educational features
 */

// ==================== MODEL MANAGEMENT ====================

/**
 * Save trained model to browser's IndexedDB
 */
async function saveModelToStorage(modelName, model, metadata = {}) {
    try {
        const saveResult = await model.save(`indexeddb://${modelName}`);
        
        // Store metadata separately
        const modelInfo = {
            name: modelName,
            timestamp: new Date().toISOString(),
            architecture: model.layers.map(layer => ({
                name: layer.name,
                className: layer.getClassName(),
                config: layer.getConfig()
            })),
            metadata: metadata,
            inputShape: model.inputs[0].shape,
            outputShape: model.outputs[0].shape
        };
        
        localStorage.setItem(`model_${modelName}_info`, JSON.stringify(modelInfo));
        
        console.log('Model saved successfully:', saveResult);
        return { success: true, modelInfo };
    } catch (error) {
        console.error('Error saving model:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Load model from browser's IndexedDB
 */
async function loadModelFromStorage(modelName) {
    try {
        const model = await tf.loadLayersModel(`indexeddb://${modelName}`);
        const infoStr = localStorage.getItem(`model_${modelName}_info`);
        const modelInfo = infoStr ? JSON.parse(infoStr) : {};
        
        console.log('Model loaded successfully');
        return { success: true, model, modelInfo };
    } catch (error) {
        console.error('Error loading model:', error);
        return { success: false, error: error.message };
    }
}

/**
 * List all saved models
 */
async function listSavedModels() {
    const models = [];
    
    // Get all models from localStorage
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith('model_') && key.endsWith('_info')) {
            try {
                const modelInfo = JSON.parse(localStorage.getItem(key));
                models.push(modelInfo);
            } catch (error) {
                console.error('Error parsing model info:', error);
            }
        }
    }
    
    return models;
}

/**
 * Delete saved model
 */
async function deleteModelFromStorage(modelName) {
    try {
        await tf.io.removeModel(`indexeddb://${modelName}`);
        localStorage.removeItem(`model_${modelName}_info`);
        return { success: true };
    } catch (error) {
        console.error('Error deleting model:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Export model as downloadable file (JSON format)
 */
async function exportModel(model, modelName = 'my_model') {
    try {
        // Save to downloads using tf.io
        await model.save(`downloads://${modelName}`);
        console.log('Model exported successfully');
        return { success: true };
    } catch (error) {
        console.error('Error exporting model:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Import model from uploaded files
 */
async function importModel(modelJsonFile, weightsFile) {
    try {
        const modelJson = await modelJsonFile.text();
        const modelConfig = JSON.parse(modelJson);
        
        // Load model from uploaded files
        const model = await tf.loadLayersModel(
            tf.io.browserFiles([modelJsonFile, weightsFile])
        );
        
        console.log('Model imported successfully');
        return { success: true, model, config: modelConfig };
    } catch (error) {
        console.error('Error importing model:', error);
        return { success: false, error: error.message };
    }
}

// ==================== DATA PREPROCESSING ====================

/**
 * Normalize numerical data (min-max scaling)
 */
function normalizeData(data, min = null, max = null) {
    const tensor = tf.tensor(data);
    
    if (min === null || max === null) {
        min = tensor.min().arraySync();
        max = tensor.max().arraySync();
    }
    
    const normalized = tensor.sub(min).div(tf.scalar(max - min));
    const result = normalized.arraySync();
    
    tensor.dispose();
    normalized.dispose();
    
    return { data: result, min, max };
}

/**
 * Standardize data (z-score normalization)
 */
function standardizeData(data, mean = null, std = null) {
    const tensor = tf.tensor(data);
    
    if (mean === null || std === null) {
        mean = tensor.mean().arraySync();
        std = tf.moments(tensor).variance.sqrt().arraySync();
    }
    
    const standardized = tensor.sub(mean).div(std);
    const result = standardized.arraySync();
    
    tensor.dispose();
    standardized.dispose();
    
    return { data: result, mean, std };
}

/**
 * One-hot encode categorical labels
 */
function oneHotEncode(labels, numClasses = null) {
    if (numClasses === null) {
        numClasses = Math.max(...labels) + 1;
    }
    
    const encoded = tf.oneHot(labels, numClasses);
    const result = encoded.arraySync();
    encoded.dispose();
    
    return { data: result, numClasses };
}

/**
 * Split data into train and test sets
 */
function trainTestSplit(features, labels, testSize = 0.2, shuffle = true) {
    const numSamples = features.length;
    const numTest = Math.floor(numSamples * testSize);
    const numTrain = numSamples - numTest;
    
    let indices = Array.from({ length: numSamples }, (_, i) => i);
    
    if (shuffle) {
        // Fisher-Yates shuffle
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
    }
    
    const trainIndices = indices.slice(0, numTrain);
    const testIndices = indices.slice(numTrain);
    
    return {
        xTrain: trainIndices.map(i => features[i]),
        xTest: testIndices.map(i => features[i]),
        yTrain: trainIndices.map(i => labels[i]),
        yTest: testIndices.map(i => labels[i])
    };
}

/**
 * Data augmentation for images
 */
async function augmentImageData(imageTensor, options = {}) {
    const {
        flip = false,
        rotate = false,
        brightness = false,
        contrast = false
    } = options;
    
    let augmented = imageTensor;
    
    if (flip && Math.random() > 0.5) {
        augmented = tf.image.flipLeftRight(augmented);
    }
    
    if (brightness && Math.random() > 0.5) {
        const delta = (Math.random() - 0.5) * 0.3;
        augmented = tf.image.adjustBrightness(augmented, delta);
    }
    
    if (contrast && Math.random() > 0.5) {
        const factor = 0.5 + Math.random();
        augmented = tf.image.adjustContrast(augmented, factor);
    }
    
    return augmented;
}

// ==================== MODEL ARCHITECTURE ====================

/**
 * Create custom neural network from layer configuration
 */
function createCustomModel(layerConfigs) {
    const layers = [];
    
    for (const config of layerConfigs) {
        let layer;
        
        switch (config.type) {
            case 'dense':
                layer = tf.layers.dense({
                    units: config.units,
                    activation: config.activation || 'relu',
                    inputShape: config.inputShape
                });
                break;
                
            case 'conv2d':
                layer = tf.layers.conv2d({
                    filters: config.filters,
                    kernelSize: config.kernelSize || [3, 3],
                    activation: config.activation || 'relu',
                    inputShape: config.inputShape
                });
                break;
                
            case 'maxPooling2d':
                layer = tf.layers.maxPooling2d({
                    poolSize: config.poolSize || [2, 2]
                });
                break;
                
            case 'flatten':
                layer = tf.layers.flatten();
                break;
                
            case 'dropout':
                layer = tf.layers.dropout({
                    rate: config.rate || 0.2
                });
                break;
                
            case 'batchNormalization':
                layer = tf.layers.batchNormalization();
                break;
                
            case 'lstm':
                layer = tf.layers.lstm({
                    units: config.units,
                    returnSequences: config.returnSequences || false
                });
                break;
                
            default:
                console.warn(`Unknown layer type: ${config.type}`);
                continue;
        }
        
        layers.push(layer);
    }
    
    const model = tf.sequential({ layers });
    return model;
}

/**
 * Get model summary as formatted text
 */
function getModelSummary(model) {
    const summary = [];
    summary.push('Model Summary');
    summary.push('=' .repeat(70));
    summary.push(`Layer (type)${' '.repeat(25)}Output Shape${' '.repeat(10)}Param #`);
    summary.push('='.repeat(70));
    
    let totalParams = 0;
    
    model.layers.forEach((layer, index) => {
        const layerName = `${layer.name} (${layer.getClassName()})`;
        const outputShape = JSON.stringify(layer.outputShape);
        const params = layer.countParams();
        totalParams += params;
        
        const line = `${layerName.padEnd(35)} ${outputShape.padEnd(20)} ${params}`;
        summary.push(line);
    });
    
    summary.push('='.repeat(70));
    summary.push(`Total params: ${totalParams}`);
    summary.push(`Trainable params: ${totalParams}`);
    summary.push(`Non-trainable params: 0`);
    summary.push('='.repeat(70));
    
    return summary.join('\n');
}

// ==================== EVALUATION METRICS ====================

/**
 * Calculate confusion matrix
 */
function calculateConfusionMatrix(predictions, labels, numClasses) {
    const matrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
    
    for (let i = 0; i < predictions.length; i++) {
        const pred = predictions[i];
        const actual = labels[i];
        matrix[actual][pred]++;
    }
    
    return matrix;
}

/**
 * Calculate classification metrics (precision, recall, F1)
 */
function calculateClassificationMetrics(confusionMatrix) {
    const numClasses = confusionMatrix.length;
    const metrics = [];
    
    for (let i = 0; i < numClasses; i++) {
        let tp = confusionMatrix[i][i];
        let fp = 0;
        let fn = 0;
        
        for (let j = 0; j < numClasses; j++) {
            if (j !== i) {
                fp += confusionMatrix[j][i];
                fn += confusionMatrix[i][j];
            }
        }
        
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        metrics.push({
            class: i,
            precision: precision,
            recall: recall,
            f1Score: f1
        });
    }
    
    return metrics;
}

/**
 * Calculate R² score for regression
 */
function calculateR2Score(predictions, actuals) {
    const predTensor = tf.tensor1d(predictions);
    const actualTensor = tf.tensor1d(actuals);
    
    const meanActual = actualTensor.mean();
    const ssTotal = actualTensor.sub(meanActual).square().sum();
    const ssResidual = actualTensor.sub(predTensor).square().sum();
    
    const r2 = tf.scalar(1).sub(ssResidual.div(ssTotal));
    const r2Value = r2.arraySync();
    
    predTensor.dispose();
    actualTensor.dispose();
    meanActual.dispose();
    ssTotal.dispose();
    ssResidual.dispose();
    r2.dispose();
    
    return r2Value;
}

// ==================== CODE GENERATION ====================

/**
 * Generate Python training code
 */
function generatePythonCode(modelConfig, trainingConfig) {
    const code = `
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load and preprocess your data
# X_train, y_train, X_test, y_test = load_your_data()

# Create model
model = keras.Sequential([
${modelConfig.layers.map(layer => {
    if (layer.type === 'dense') {
        return `    keras.layers.Dense(${layer.units}, activation='${layer.activation}')`;
    } else if (layer.type === 'conv2d') {
        return `    keras.layers.Conv2D(${layer.filters}, kernel_size=${JSON.stringify(layer.kernelSize)}, activation='${layer.activation}')`;
    } else if (layer.type === 'dropout') {
        return `    keras.layers.Dropout(${layer.rate})`;
    }
    return `    # ${layer.type} layer`;
}).join(',\n')}
])

# Compile model
model.compile(
    optimizer='${trainingConfig.optimizer || 'adam'}',
    loss='${trainingConfig.loss || 'categorical_crossentropy'}',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=${trainingConfig.epochs || 50},
    batch_size=${trainingConfig.batchSize || 32},
    validation_split=0.2,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

# Save model
model.save('my_model.h5')
`;
    
    return code;
}

/**
 * Generate JavaScript/TensorFlow.js code
 */
function generateJavaScriptCode(modelConfig, trainingConfig) {
    const code = `
const tf = require('@tensorflow/tfjs-node');

// Load and preprocess your data
// const { xTrain, yTrain, xTest, yTest } = loadYourData();

// Create model
const model = tf.sequential({
    layers: [
${modelConfig.layers.map(layer => {
    if (layer.type === 'dense') {
        return `        tf.layers.dense({ units: ${layer.units}, activation: '${layer.activation}' })`;
    } else if (layer.type === 'conv2d') {
        return `        tf.layers.conv2d({ filters: ${layer.filters}, kernelSize: ${JSON.stringify(layer.kernelSize)}, activation: '${layer.activation}' })`;
    } else if (layer.type === 'dropout') {
        return `        tf.layers.dropout({ rate: ${layer.rate} })`;
    }
    return `        // ${layer.type} layer`;
}).join(',\n')}
    ]
});

// Compile model
model.compile({
    optimizer: '${trainingConfig.optimizer || 'adam'}',
    loss: '${trainingConfig.loss || 'categoricalCrossentropy'}',
    metrics: ['accuracy']
});

// Train model
async function trainModel() {
    const history = await model.fit(xTrain, yTrain, {
        epochs: ${trainingConfig.epochs || 50},
        batchSize: ${trainingConfig.batchSize || 32},
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(\`Epoch \${epoch + 1}: loss = \${logs.loss.toFixed(4)}, accuracy = \${logs.acc.toFixed(4)}\`);
            }
        }
    });
    
    // Evaluate model
    const evaluation = model.evaluate(xTest, yTest);
    console.log('Test accuracy:', evaluation[1].dataSync()[0]);
    
    // Save model
    await model.save('file://./my_model');
}

trainModel();
`;
    
    return code;
}

/**
 * Generate deployment code (Node.js server)
 */
function generateDeploymentCode(modelName) {
    const code = `
const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const app = express();

app.use(express.json());

let model;

// Load model on startup
async function loadModel() {
    model = await tf.loadLayersModel('file://./model/${modelName}/model.json');
    console.log('Model loaded successfully');
}

// Prediction endpoint
app.post('/predict', async (req, res) => {
    try {
        const inputData = req.body.data;
        const inputTensor = tf.tensor(inputData);
        const prediction = model.predict(inputTensor);
        const result = await prediction.data();
        
        inputTensor.dispose();
        prediction.dispose();
        
        res.json({ prediction: Array.from(result) });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', model: '${modelName}' });
});

const PORT = process.env.PORT || 3000;

loadModel().then(() => {
    app.listen(PORT, () => {
        console.log(\`Server running on port \${PORT}\`);
    });
});
`;
    
    return code;
}

// ==================== EDUCATIONAL UTILITIES ====================

/**
 * Get explanation for a layer type
 */
function getLayerExplanation(layerType) {
    const explanations = {
        dense: "Dense (Fully Connected) Layer: Connects every neuron to all neurons in the next layer. Each connection has a weight that the model learns during training. Used for learning complex patterns.",
        
        conv2d: "Convolutional Layer: Applies filters to detect features like edges, textures, or patterns in images. Each filter learns to recognize specific visual features.",
        
        maxPooling2d: "Max Pooling Layer: Reduces the spatial dimensions by taking the maximum value in each region. Helps make the model more efficient and robust to small translations.",
        
        dropout: "Dropout Layer: Randomly deactivates neurons during training to prevent overfitting. Makes the model more generalizable to new data.",
        
        flatten: "Flatten Layer: Converts multi-dimensional data into a 1D vector. Necessary before feeding data into dense layers.",
        
        batchNormalization: "Batch Normalization: Normalizes the inputs of each layer. Speeds up training and makes the model more stable.",
        
        lstm: "LSTM Layer: Long Short-Term Memory layer for processing sequences. Remembers important information and forgets irrelevant details over time.",
        
        activation: "Activation Function: Introduces non-linearity into the model, allowing it to learn complex patterns. Common types: ReLU, sigmoid, tanh, softmax."
    };
    
    return explanations[layerType] || "Custom layer for specific operations.";
}

/**
 * Get explanation for a hyperparameter
 */
function getHyperparameterExplanation(param) {
    const explanations = {
        epochs: "Epochs: Number of times the model sees the entire dataset. More epochs = more learning, but risk of overfitting.",
        
        batchSize: "Batch Size: Number of samples processed before updating weights. Larger = faster but needs more memory. Smaller = more accurate updates but slower.",
        
        learningRate: "Learning Rate: Controls how much to adjust weights during training. Too high = unstable, too low = slow learning. Typical range: 0.001 to 0.1.",
        
        optimizer: "Optimizer: Algorithm for updating weights. Adam is a good default. SGD is simpler but may need tuning.",
        
        loss: "Loss Function: Measures how wrong the model's predictions are. The model tries to minimize this during training.",
        
        validation_split: "Validation Split: Portion of data used for validation. Helps detect overfitting. Typical: 0.2 (20%)."
    };
    
    return explanations[param] || "A training parameter that affects model performance.";
}

/**
 * Get tips for improving model performance
 */
function getPerformanceTips(metrics) {
    const tips = [];
    
    if (metrics.loss > 1.0) {
        tips.push("High loss detected. Try: increasing epochs, adjusting learning rate, or checking data quality.");
    }
    
    if (metrics.accuracy < 0.7) {
        tips.push("Low accuracy. Consider: more training data, deeper model, or different architecture.");
    }
    
    if (metrics.valLoss && metrics.valLoss > metrics.loss * 1.5) {
        tips.push("Overfitting detected. Try: adding dropout, reducing model complexity, or more training data.");
    }
    
    if (metrics.valLoss && metrics.valLoss < metrics.loss) {
        tips.push("Model generalizes well! Validation performance is better than training.");
    }
    
    if (tips.length === 0) {
        tips.push("Model is performing well! Continue monitoring during training.");
    }
    
    return tips;
}

// ==================== DATASETS LIBRARY ====================

/**
 * Built-in datasets for quick experimentation
 */
const builtInDatasets = {
    iris: {
        name: 'Iris Flower Classification',
        description: 'Classic dataset with 3 species of iris flowers',
        features: 4,
        classes: 3,
        samples: 150,
        generate: function() {
            // Simplified iris dataset generation
            const data = [];
            // Setosa
            for (let i = 0; i < 50; i++) {
                data.push({
                    features: [
                        5.0 + Math.random() * 0.8,
                        3.4 + Math.random() * 0.6,
                        1.4 + Math.random() * 0.3,
                        0.2 + Math.random() * 0.1
                    ],
                    label: 0
                });
            }
            // Versicolor
            for (let i = 0; i < 50; i++) {
                data.push({
                    features: [
                        6.0 + Math.random() * 0.9,
                        2.8 + Math.random() * 0.6,
                        4.3 + Math.random() * 0.8,
                        1.3 + Math.random() * 0.3
                    ],
                    label: 1
                });
            }
            // Virginica
            for (let i = 0; i < 50; i++) {
                data.push({
                    features: [
                        6.5 + Math.random() * 1.0,
                        3.0 + Math.random() * 0.6,
                        5.5 + Math.random() * 1.0,
                        2.0 + Math.random() * 0.5
                    ],
                    label: 2
                });
            }
            return data;
        }
    },
    
    xor: {
        name: 'XOR Problem',
        description: 'Classic non-linear problem for neural networks',
        features: 2,
        classes: 2,
        samples: 400,
        generate: function() {
            const data = [];
            for (let i = 0; i < 400; i++) {
                const x1 = Math.random() * 2 - 1;
                const x2 = Math.random() * 2 - 1;
                const label = (x1 * x2 > 0) ? 1 : 0;
                data.push({ features: [x1, x2], label });
            }
            return data;
        }
    },
    
    spiral: {
        name: 'Spiral Classification',
        description: 'Two intertwined spirals - challenging non-linear problem',
        features: 2,
        classes: 2,
        samples: 300,
        generate: function() {
            const data = [];
            const points = 150;
            
            for (let i = 0; i < points; i++) {
                const r = i / points * 5;
                const t = 1.75 * i / points * 2 * Math.PI;
                
                // Spiral 1
                const x1 = r * Math.cos(t) + Math.random() * 0.3;
                const y1 = r * Math.sin(t) + Math.random() * 0.3;
                data.push({ features: [x1, y1], label: 0 });
                
                // Spiral 2
                const x2 = r * Math.cos(t + Math.PI) + Math.random() * 0.3;
                const y2 = r * Math.sin(t + Math.PI) + Math.random() * 0.3;
                data.push({ features: [x2, y2], label: 1 });
            }
            
            return data;
        }
    }
};

// Export all functions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        saveModelToStorage,
        loadModelFromStorage,
        listSavedModels,
        deleteModelFromStorage,
        exportModel,
        importModel,
        normalizeData,
        standardizeData,
        oneHotEncode,
        trainTestSplit,
        augmentImageData,
        createCustomModel,
        getModelSummary,
        calculateConfusionMatrix,
        calculateClassificationMetrics,
        calculateR2Score,
        generatePythonCode,
        generateJavaScriptCode,
        generateDeploymentCode,
        getLayerExplanation,
        getHyperparameterExplanation,
        getPerformanceTips,
        builtInDatasets
    };
}

// Global State Management
const appState = {
    selectedModel: null,
    trainingData: null,
    model: null,
    isTraining: false,
    trainingHistory: {
        loss: [],
        accuracy: [],
        epochs: []
    },
    chart: null
};

// Educational insights for different training phases
const learningInsights = {
    initialization: "🚀 <strong>Initialization:</strong> The neural network is being created with random weights. Think of this as a student starting with no knowledge.",
    earlyTraining: "📊 <strong>Early Training:</strong> The model is making wild guesses and learning from big mistakes. Loss is high but decreasing rapidly.",
    midTraining: "🎯 <strong>Mid Training:</strong> The model is starting to recognize patterns. It's like a student beginning to understand the concepts.",
    lateTraining: "✨ <strong>Fine-tuning:</strong> The model is making small adjustments to improve accuracy. It's refining its understanding.",
    convergence: "🎓 <strong>Converging:</strong> The model's performance is stabilizing. Further training might not improve it much more.",
    complete: "🏆 <strong>Complete!</strong> Training finished successfully. The model has learned from your data and is ready to make predictions!"
};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('ML Learning Platform initialized');
    console.log('TensorFlow.js version:', tf.version.tfjs);
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    // File input change
    fileInput.addEventListener('change', handleFileUpload);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        handleFiles(files);
    });
}

// Model Selection
function selectModel(modelType) {
    appState.selectedModel = modelType;
    
    // Update UI
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`[data-model="${modelType}"]`).classList.add('selected');
    
    // Show data section
    document.getElementById('dataSection').style.display = 'block';
    document.getElementById('dataSection').scrollIntoView({ behavior: 'smooth' });
    
    console.log(`Selected model: ${modelType}`);
}

// File Upload Handling
function handleFileUpload(event) {
    const files = event.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    const dataPreview = document.getElementById('dataPreview');
    dataPreview.innerHTML = '<div class="spinner"></div><p>Processing files...</p>';
    
    setTimeout(() => {
        dataPreview.classList.add('active');
        dataPreview.innerHTML = `
            <h3>✅ Files Uploaded Successfully</h3>
            <p><strong>${files.length}</strong> file(s) ready for training</p>
            <ul>
                ${Array.from(files).map(file => `<li>📄 ${escapeHtml(file.name)} (${formatFileSize(file.size)})</li>`).join('')}
            </ul>
        `;
        
        // Process files based on model type
        processFiles(files);
    }, 1000);
}

function processFiles(files) {
    // Store files for training
    appState.trainingData = {
        files: Array.from(files),
        processed: true
    };
    
    // Show training configuration
    document.getElementById('trainingSection').style.display = 'block';
    document.getElementById('trainingSection').scrollIntoView({ behavior: 'smooth' });
}

// Load Sample Data
function loadSampleData() {
    const dataPreview = document.getElementById('dataPreview');
    dataPreview.classList.add('active');
    
    // Generate sample data based on selected model
    let sampleData;
    
    if (appState.selectedModel === 'regression') {
        sampleData = generateRegressionData();
        dataPreview.innerHTML = `
            <h3>✅ Sample Linear Regression Data Loaded</h3>
            <p><strong>100 data points</strong> with linear relationship</p>
            <p>Formula: y = 2x + 1 + noise</p>
            <div style="margin-top: 1rem;">
                <strong>Sample values:</strong><br>
                ${sampleData.slice(0, 5).map(d => `x: ${d.x.toFixed(2)} → y: ${d.y.toFixed(2)}`).join('<br>')}
                <br>... and 95 more
            </div>
        `;
    } else if (appState.selectedModel === 'neural') {
        sampleData = generateClassificationData();
        dataPreview.innerHTML = `
            <h3>✅ Sample Classification Data Loaded</h3>
            <p><strong>200 data points</strong> for binary classification</p>
            <p>Two distinct classes with feature separation</p>
        `;
    } else {
        sampleData = generateImageData();
        dataPreview.innerHTML = `
            <h3>✅ Sample Image Data Loaded</h3>
            <p><strong>50 sample images</strong> for classification</p>
            <p>Simple patterns for demonstration</p>
        `;
    }
    
    appState.trainingData = {
        sample: true,
        data: sampleData,
        processed: true
    };
    
    // Show training configuration
    document.getElementById('trainingSection').style.display = 'block';
    document.getElementById('trainingSection').scrollIntoView({ behavior: 'smooth' });
}

// Sample Data Generators
function generateRegressionData() {
    const data = [];
    for (let i = 0; i < 100; i++) {
        const x = Math.random() * 10;
        const y = 2 * x + 1 + (Math.random() - 0.5) * 2; // y = 2x + 1 with noise
        data.push({ x, y });
    }
    return data;
}

function generateClassificationData() {
    const data = [];
    for (let i = 0; i < 100; i++) {
        const x1 = Math.random() * 10;
        const x2 = Math.random() * 10;
        const label = (x1 + x2 > 10) ? 1 : 0;
        data.push({ features: [x1, x2], label });
    }
    for (let i = 0; i < 100; i++) {
        const x1 = Math.random() * 10;
        const x2 = Math.random() * 10;
        const label = (x1 + x2 > 10) ? 1 : 0;
        data.push({ features: [x1, x2], label });
    }
    return data;
}

function generateImageData() {
    const data = [];
    for (let i = 0; i < 50; i++) {
        const pixels = Array(28 * 28).fill(0).map(() => Math.random());
        const label = i % 2; // Simple binary classification
        data.push({ pixels, label });
    }
    return data;
}

// Start Training
async function startTraining() {
    if (!appState.trainingData) {
        alert('Please upload or load sample data first!');
        return;
    }
    
    appState.isTraining = true;
    const trainButton = document.getElementById('trainButton');
    trainButton.disabled = true;
    trainButton.textContent = '⏳ Training...';
    
    // Show progress section
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('progressSection').scrollIntoView({ behavior: 'smooth' });
    
    // Get training parameters
    const epochs = parseInt(document.getElementById('epochs').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    
    try {
        // Train based on selected model
        if (appState.selectedModel === 'regression') {
            await trainRegressionModel(epochs, learningRate, batchSize);
        } else if (appState.selectedModel === 'neural') {
            await trainNeuralNetwork(epochs, learningRate, batchSize);
        } else if (appState.selectedModel === 'image') {
            await trainImageClassifier(epochs, learningRate, batchSize);
        }
        
        // Show results
        showResults();
    } catch (error) {
        console.error('Training error:', error);
        alert('An error occurred during training. Please try again.');
    } finally {
        appState.isTraining = false;
        trainButton.disabled = false;
        trainButton.textContent = '🚀 Start Training';
    }
}

// Train Linear Regression Model
async function trainRegressionModel(epochs, learningRate, batchSize) {
    updateLearningInsight('initialization');
    
    // Prepare data
    const data = appState.trainingData.data;
    const xs = data.map(d => d.x);
    const ys = data.map(d => d.y);
    
    const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
    const ysTensor = tf.tensor2d(ys, [ys.length, 1]);
    
    // Create model
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [1], units: 1, useBias: true })
        ]
    });
    
    model.compile({
        optimizer: tf.train.sgd(learningRate),
        loss: 'meanSquaredError',
        metrics: ['mae']
    });
    
    appState.model = model;
    
    // Initialize chart
    initializeChart();
    
    // Train model
    await model.fit(xsTensor, ysTensor, {
        epochs: epochs,
        batchSize: batchSize,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                updateTrainingProgress(epoch, epochs, logs);
                
                // Update insights based on progress
                const progress = epoch / epochs;
                if (progress < 0.2) updateLearningInsight('earlyTraining');
                else if (progress < 0.5) updateLearningInsight('midTraining');
                else if (progress < 0.8) updateLearningInsight('lateTraining');
                else updateLearningInsight('convergence');
                
                await tf.nextFrame();
            }
        }
    });
    
    updateLearningInsight('complete');
    
    // Cleanup tensors
    xsTensor.dispose();
    ysTensor.dispose();
}

// Train Neural Network
async function trainNeuralNetwork(epochs, learningRate, batchSize) {
    updateLearningInsight('initialization');
    
    // Prepare data
    const data = appState.trainingData.data;
    const features = data.map(d => d.features);
    const labels = data.map(d => d.label);
    
    const xsTensor = tf.tensor2d(features);
    const ysTensor = tf.tensor2d(labels, [labels.length, 1]);
    
    // Create model
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [2], units: 16, activation: 'relu' }),
            tf.layers.dense({ units: 8, activation: 'relu' }),
            tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
    });
    
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    appState.model = model;
    
    // Initialize chart
    initializeChart();
    
    // Train model
    await model.fit(xsTensor, ysTensor, {
        epochs: epochs,
        batchSize: batchSize,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                updateTrainingProgress(epoch, epochs, logs);
                
                const progress = epoch / epochs;
                if (progress < 0.2) updateLearningInsight('earlyTraining');
                else if (progress < 0.5) updateLearningInsight('midTraining');
                else if (progress < 0.8) updateLearningInsight('lateTraining');
                else updateLearningInsight('convergence');
                
                await tf.nextFrame();
            }
        }
    });
    
    updateLearningInsight('complete');
    
    xsTensor.dispose();
    ysTensor.dispose();
}

// Train Image Classifier
async function trainImageClassifier(epochs, learningRate, batchSize) {
    updateLearningInsight('initialization');
    
    // Prepare data
    const data = appState.trainingData.data;
    const pixels = data.map(d => d.pixels);
    const labels = data.map(d => d.label);
    
    const xsTensor = tf.tensor2d(pixels);
    const ysTensor = tf.tensor2d(labels, [labels.length, 1]);
    
    // Create CNN model
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [28 * 28], units: 128, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.2 }),
            tf.layers.dense({ units: 64, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.2 }),
            tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
    });
    
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    appState.model = model;
    
    // Initialize chart
    initializeChart();
    
    // Train model
    await model.fit(xsTensor, ysTensor, {
        epochs: epochs,
        batchSize: batchSize,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                updateTrainingProgress(epoch, epochs, logs);
                
                const progress = epoch / epochs;
                if (progress < 0.2) updateLearningInsight('earlyTraining');
                else if (progress < 0.5) updateLearningInsight('midTraining');
                else if (progress < 0.8) updateLearningInsight('lateTraining');
                else updateLearningInsight('convergence');
                
                await tf.nextFrame();
            }
        }
    });
    
    updateLearningInsight('complete');
    
    xsTensor.dispose();
    ysTensor.dispose();
}

// Update Training Progress
function updateTrainingProgress(epoch, totalEpochs, logs) {
    const progress = ((epoch + 1) / totalEpochs) * 100;
    
    // Update progress bar
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = `${progress}%`;
    progressBar.textContent = `${Math.round(progress)}%`;
    
    // Update progress text
    document.getElementById('progressText').textContent = 
        `Epoch ${epoch + 1} of ${totalEpochs}`;
    
    // Update metrics
    document.getElementById('lossValue').textContent = logs.loss.toFixed(4);
    document.getElementById('accuracyValue').textContent = 
        logs.acc ? `${(logs.acc * 100).toFixed(2)}%` : `${(logs.mae ? (100 - logs.mae * 10).toFixed(2) : 'N/A')}%`;
    document.getElementById('epochValue').textContent = epoch + 1;
    
    // Store history
    appState.trainingHistory.epochs.push(epoch + 1);
    appState.trainingHistory.loss.push(logs.loss);
    appState.trainingHistory.accuracy.push(logs.acc || (logs.mae ? (1 - logs.mae / 10) : 0));
    
    // Update chart
    updateChart();
}

// Initialize Chart
function initializeChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    
    if (appState.chart) {
        appState.chart.destroy();
    }
    
    appState.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Loss',
                    data: [],
                    borderColor: 'rgb(244, 63, 94)',
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                },
                {
                    label: 'Accuracy',
                    data: [],
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Training Metrics Over Time'
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Accuracy'
                    },
                    grid: {
                        drawOnChartArea: false,
                    }
                }
            }
        }
    });
}

// Update Chart
function updateChart() {
    if (!appState.chart) return;
    
    appState.chart.data.labels = appState.trainingHistory.epochs;
    appState.chart.data.datasets[0].data = appState.trainingHistory.loss;
    appState.chart.data.datasets[1].data = appState.trainingHistory.accuracy;
    appState.chart.update('none'); // Update without animation for performance
}

// Update Learning Insights
function updateLearningInsight(phase) {
    const insightsDiv = document.getElementById('learningInsights');
    insightsDiv.innerHTML = `<p>${learningInsights[phase]}</p>`;
}

// Show Results
function showResults() {
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    
    const finalLoss = appState.trainingHistory.loss[appState.trainingHistory.loss.length - 1];
    const finalAccuracy = appState.trainingHistory.accuracy[appState.trainingHistory.accuracy.length - 1];
    
    const summaryHTML = `
        <h3>Training Summary</h3>
        <p><strong>Model Type:</strong> ${appState.selectedModel.charAt(0).toUpperCase() + appState.selectedModel.slice(1)}</p>
        <p><strong>Total Epochs:</strong> ${appState.trainingHistory.epochs.length}</p>
        <p><strong>Final Loss:</strong> ${finalLoss.toFixed(4)}</p>
        <p><strong>Final Accuracy:</strong> ${(finalAccuracy * 100).toFixed(2)}%</p>
        <p><strong>Status:</strong> ✅ Successfully trained and ready for predictions!</p>
    `;
    
    document.getElementById('resultSummary').innerHTML = summaryHTML;
    
    // Setup test input based on model type
    setupTestInput();
}

// Setup Test Input
function setupTestInput() {
    const testInputArea = document.getElementById('testInputArea');
    
    if (appState.selectedModel === 'regression') {
        testInputArea.innerHTML = `
            <label for="testX">Enter X value:</label>
            <input type="number" id="testX" step="0.1" value="5" 
                   style="width: 100%; padding: 0.75rem; border: 2px solid var(--border-color); 
                          border-radius: 8px; margin-top: 0.5rem;">
        `;
    } else if (appState.selectedModel === 'neural') {
        testInputArea.innerHTML = `
            <label>Enter feature values:</label>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem;">
                <input type="number" id="testX1" step="0.1" value="5" placeholder="Feature 1"
                       style="padding: 0.75rem; border: 2px solid var(--border-color); border-radius: 8px;">
                <input type="number" id="testX2" step="0.1" value="7" placeholder="Feature 2"
                       style="padding: 0.75rem; border: 2px solid var(--border-color); border-radius: 8px;">
            </div>
        `;
    } else {
        testInputArea.innerHTML = `
            <p>Draw or upload an image to test the classifier:</p>
            <input type="file" id="testImage" accept="image/*"
                   style="margin-top: 0.5rem;">
        `;
    }
}

// Test Model
async function testModel() {
    if (!appState.model) {
        alert('No trained model available!');
        return;
    }
    
    let prediction;
    
    try {
        if (appState.selectedModel === 'regression') {
            const x = parseFloat(document.getElementById('testX').value);
            const xTensor = tf.tensor2d([x], [1, 1]);
            const result = appState.model.predict(xTensor);
            prediction = await result.data();
            xTensor.dispose();
            result.dispose();
            
            showPredictionResult(`For x = ${x}, predicted y = ${prediction[0].toFixed(4)}`);
        } else if (appState.selectedModel === 'neural') {
            const x1 = parseFloat(document.getElementById('testX1').value);
            const x2 = parseFloat(document.getElementById('testX2').value);
            const xTensor = tf.tensor2d([[x1, x2]]);
            const result = appState.model.predict(xTensor);
            prediction = await result.data();
            xTensor.dispose();
            result.dispose();
            
            const classLabel = prediction[0] > 0.5 ? 'Class 1' : 'Class 0';
            const confidence = (prediction[0] > 0.5 ? prediction[0] : 1 - prediction[0]) * 100;
            showPredictionResult(`Prediction: ${classLabel} (${confidence.toFixed(2)}% confidence)`);
        } else {
            showPredictionResult('Image prediction: Demo mode - Class A (85% confidence)');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please try again.');
    }
}

// Show Prediction Result
function showPredictionResult(message) {
    const resultDiv = document.getElementById('predictionResult');
    // Message is constructed from controlled sources (parseFloat, model predictions, string literals)
    resultDiv.innerHTML = `<strong>Result:</strong> ${message}`;
    resultDiv.classList.remove('hidden');
    resultDiv.classList.add('active');
}

// Reset Application
function resetApp() {
    // Clean up TensorFlow resources
    if (appState.model) {
        appState.model.dispose();
    }
    
    // Reset state
    appState.selectedModel = null;
    appState.trainingData = null;
    appState.model = null;
    appState.isTraining = false;
    appState.trainingHistory = { loss: [], accuracy: [], epochs: [] };
    
    if (appState.chart) {
        appState.chart.destroy();
        appState.chart = null;
    }
    
    // Reset UI
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    document.getElementById('dataSection').style.display = 'none';
    document.getElementById('trainingSection').style.display = 'none';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('dataPreview').innerHTML = '';
    document.getElementById('dataPreview').classList.remove('active');
    document.getElementById('fileInput').value = '';
    
    // Reset form values
    document.getElementById('epochs').value = '50';
    document.getElementById('learningRate').value = '0.01';
    document.getElementById('batchSize').value = '32';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Performance monitoring
console.log('TensorFlow.js backend:', tf.getBackend());
console.log('Memory:', tf.memory());

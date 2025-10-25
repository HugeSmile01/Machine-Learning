// Global State Management
const appState = {
    selectedModel: null,
    trainingData: null,
    model: null,
    isTraining: false,
    trainingHistory: {
        loss: [],
        accuracy: [],
        epochs: [],
        valLoss: [],
        valAccuracy: []
    },
    chart: null,
    modelArchitecture: [],
    savedModels: [],
    currentDataset: null,
    preprocessingSteps: []
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
    resultDiv.classList.add('block', 'active');
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

// ==================== NEW COMPREHENSIVE FEATURES ====================

// Model Management Functions

async function saveModel() {
    if (!appState.model) {
        alert('No trained model to save!');
        return;
    }
    
    const modelName = prompt('Enter a name for your model:', `model_${Date.now()}`);
    if (!modelName) return;
    
    const metadata = {
        modelType: appState.selectedModel,
        trainingDate: new Date().toISOString(),
        finalLoss: appState.trainingHistory.loss[appState.trainingHistory.loss.length - 1],
        finalAccuracy: appState.trainingHistory.accuracy[appState.trainingHistory.accuracy.length - 1],
        epochs: appState.trainingHistory.epochs.length,
        architecture: appState.model.layers.map(l => ({
            type: l.getClassName(),
            config: l.getConfig()
        }))
    };
    
    const result = await saveModelToStorage(modelName, appState.model, metadata);
    
    if (result.success) {
        alert(`Model "${modelName}" saved successfully!`);
        await refreshSavedModels();
    } else {
        alert(`Error saving model: ${result.error}`);
    }
}

async function downloadModel() {
    if (!appState.model) {
        alert('No trained model to download!');
        return;
    }
    
    const modelName = prompt('Enter a name for your model:', 'my_model');
    if (!modelName) return;
    
    try {
        await exportModel(appState.model, modelName);
        alert(`Model "${modelName}" downloaded successfully! Check your Downloads folder.`);
    } catch (error) {
        console.error('Download error:', error);
        alert(`Error downloading model: ${error.message}`);
    }
}

async function refreshSavedModels() {
    const modelsList = document.getElementById('savedModelsList');
    const models = await listSavedModels();
    
    if (models.length === 0) {
        modelsList.innerHTML = '<p class="text-gray-500 italic">No saved models yet</p>';
        appState.savedModels = [];
        return;
    }
    
    appState.savedModels = models;
    
    modelsList.innerHTML = models.map((model, index) => `
        <div class="bg-white p-4 rounded-lg border-2 border-gray-200 hover:border-primary transition-all">
            <div class="flex justify-between items-start mb-2">
                <div class="flex-1">
                    <h4 class="font-bold text-primary">${escapeHtml(model.name)}</h4>
                    <p class="text-xs text-gray-500">${new Date(model.timestamp).toLocaleString()}</p>
                </div>
                <div class="flex gap-2">
                    <button class="text-blue-500 hover:text-blue-700 font-semibold text-sm" onclick="loadSavedModel('${escapeHtml(model.name)}')">Load</button>
                    <button class="text-red-500 hover:text-red-700 font-semibold text-sm" onclick="deleteSavedModel('${escapeHtml(model.name)}')">Delete</button>
                </div>
            </div>
            ${model.metadata ? `
                <p class="text-xs text-gray-600">
                    Type: ${model.metadata.modelType || 'N/A'} | 
                    Accuracy: ${model.metadata.finalAccuracy ? (model.metadata.finalAccuracy * 100).toFixed(2) + '%' : 'N/A'}
                </p>
            ` : ''}
        </div>
    `).join('');
}

async function loadSavedModel(modelName) {
    const result = await loadModelFromStorage(modelName);
    
    if (result.success) {
        appState.model = result.model;
        appState.selectedModel = result.modelInfo.metadata?.modelType || 'custom';
        alert(`Model "${modelName}" loaded successfully! You can now use it for predictions.`);
        
        // Show results section for testing
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
        
        const summaryHTML = `
            <h3>Loaded Model Summary</h3>
            <p><strong>Model Name:</strong> ${escapeHtml(modelName)}</p>
            <p><strong>Model Type:</strong> ${result.modelInfo.metadata?.modelType || 'Custom'}</p>
            <p><strong>Saved:</strong> ${new Date(result.modelInfo.timestamp).toLocaleString()}</p>
            <p><strong>Status:</strong> ✅ Ready for predictions!</p>
        `;
        
        document.getElementById('resultSummary').innerHTML = summaryHTML;
        setupTestInput();
    } else {
        alert(`Error loading model: ${result.error}`);
    }
}

async function deleteSavedModel(modelName) {
    if (!confirm(`Are you sure you want to delete model "${modelName}"?`)) {
        return;
    }
    
    const result = await deleteModelFromStorage(modelName);
    
    if (result.success) {
        alert(`Model "${modelName}" deleted successfully!`);
        await refreshSavedModels();
    } else {
        alert(`Error deleting model: ${result.error}`);
    }
}

async function importModelFiles() {
    const jsonInput = document.getElementById('modelJsonInput');
    const weightsInput = document.getElementById('modelWeightsInput');
    
    if (!jsonInput.files.length || !weightsInput.files.length) {
        alert('Please select both model.json and weights file!');
        return;
    }
    
    try {
        const result = await importModel(jsonInput.files[0], weightsInput.files[0]);
        
        if (result.success) {
            appState.model = result.model;
            alert('Model imported successfully! You can now use it for predictions.');
            
            // Show results section
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
            setupTestInput();
        } else {
            alert(`Error importing model: ${result.error}`);
        }
    } catch (error) {
        console.error('Import error:', error);
        alert(`Error importing model: ${error.message}`);
    }
}

// Built-in Datasets

function loadBuiltInDataset(datasetName) {
    if (!builtInDatasets[datasetName]) {
        alert('Dataset not found!');
        return;
    }
    
    const dataset = builtInDatasets[datasetName];
    const data = dataset.generate();
    
    appState.trainingData = {
        builtin: true,
        data: data,
        processed: true,
        dataset: datasetName
    };
    
    // Auto-select appropriate model type
    if (dataset.classes === 2) {
        appState.selectedModel = 'neural';
    } else if (dataset.classes > 2) {
        appState.selectedModel = 'neural';
    }
    
    const dataPreview = document.getElementById('dataPreview');
    dataPreview.classList.add('active');
    dataPreview.innerHTML = `
        <h3>✅ ${dataset.name} Loaded</h3>
        <p>${dataset.description}</p>
        <p><strong>Features:</strong> ${dataset.features} | <strong>Classes:</strong> ${dataset.classes} | <strong>Samples:</strong> ${dataset.samples}</p>
        <div class="mt-4 text-sm">
            <strong>Sample data:</strong><br>
            ${data.slice(0, 3).map(d => 
                `Features: [${d.features.map(f => f.toFixed(2)).join(', ')}] → Label: ${d.label}`
            ).join('<br>')}
            <br>... and ${data.length - 3} more samples
        </div>
    `;
    
    // Show data and training sections
    document.getElementById('dataSection').style.display = 'block';
    document.getElementById('trainingSection').style.display = 'block';
    document.getElementById('trainingSection').scrollIntoView({ behavior: 'smooth' });
}

// Custom Architecture Builder

function addLayerToArchitecture(layerType) {
    let config = { type: layerType };
    
    switch (layerType) {
        case 'dense':
            const units = prompt('Enter number of units (neurons):', '64');
            if (!units) return;
            config.units = parseInt(units);
            config.activation = prompt('Enter activation function (relu, sigmoid, tanh, softmax):', 'relu');
            break;
            
        case 'conv2d':
            const filters = prompt('Enter number of filters:', '32');
            if (!filters) return;
            config.filters = parseInt(filters);
            config.kernelSize = [3, 3];
            config.activation = 'relu';
            break;
            
        case 'dropout':
            const rate = prompt('Enter dropout rate (0.0 to 1.0):', '0.2');
            if (!rate) return;
            config.rate = parseFloat(rate);
            break;
            
        case 'flatten':
            // No additional config needed
            break;
    }
    
    if (!appState.modelArchitecture) {
        appState.modelArchitecture = [];
    }
    
    appState.modelArchitecture.push(config);
    displayArchitecture();
}

function displayArchitecture() {
    const display = document.getElementById('architectureDisplay');
    
    if (appState.modelArchitecture.length === 0) {
        display.innerHTML = '<p class="text-gray-400 italic text-center">No layers added yet. Click buttons above to build your model.</p>';
        return;
    }
    
    display.innerHTML = appState.modelArchitecture.map((layer, index) => {
        let layerInfo = `<strong>${layer.type}</strong>`;
        
        if (layer.type === 'dense') {
            layerInfo += ` (${layer.units} units, ${layer.activation})`;
        } else if (layer.type === 'conv2d') {
            layerInfo += ` (${layer.filters} filters, ${layer.activation})`;
        } else if (layer.type === 'dropout') {
            layerInfo += ` (rate: ${layer.rate})`;
        }
        
        return `
            <div class="flex justify-between items-center bg-gray-50 p-3 rounded-lg mb-2">
                <span>Layer ${index + 1}: ${layerInfo}</span>
                <button class="text-red-500 hover:text-red-700 font-bold" onclick="removeLayer(${index})">✕</button>
            </div>
        `;
    }).join('');
}

function removeLayer(index) {
    appState.modelArchitecture.splice(index, 1);
    displayArchitecture();
}

function clearArchitecture() {
    appState.modelArchitecture = [];
    displayArchitecture();
}

async function buildCustomModel() {
    if (!appState.modelArchitecture || appState.modelArchitecture.length === 0) {
        alert('Please add at least one layer to your architecture!');
        return;
    }
    
    if (!appState.trainingData) {
        alert('Please load or upload training data first!');
        return;
    }
    
    try {
        // Add input shape to first layer
        if (!appState.modelArchitecture[0].inputShape) {
            const inputDim = appState.trainingData.data[0].features.length;
            appState.modelArchitecture[0].inputShape = [inputDim];
        }
        
        const model = createCustomModel(appState.modelArchitecture);
        appState.model = model;
        appState.selectedModel = 'custom';
        
        alert('Custom model created successfully! Configure training parameters and start training.');
        
        // Show training section
        document.getElementById('trainingSection').style.display = 'block';
        document.getElementById('trainingSection').scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Error building model:', error);
        alert(`Error building model: ${error.message}`);
    }
}

// Code Generation

function showCodeGeneration() {
    if (!appState.model) {
        alert('No trained model available! Train a model first.');
        return;
    }
    
    const modal = document.getElementById('codeModal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    
    // Show Python code by default
    showCodeTab('python');
}

function showCodeTab(language) {
    const tabs = ['pythonTab', 'javascriptTab', 'deployTab'];
    tabs.forEach(tab => {
        const element = document.getElementById(tab);
        element.classList.remove('bg-primary', 'text-white');
        element.classList.add('bg-gray-200', 'text-gray-700');
    });
    
    const activeTab = document.getElementById(`${language}Tab`);
    activeTab.classList.remove('bg-gray-200', 'text-gray-700');
    activeTab.classList.add('bg-primary', 'text-white');
    
    const modelConfig = {
        layers: appState.model.layers.map(layer => ({
            type: layer.getClassName().toLowerCase().replace('layers.', ''),
            ...layer.getConfig()
        }))
    };
    
    const trainingConfig = {
        epochs: parseInt(document.getElementById('epochs').value),
        batchSize: parseInt(document.getElementById('batchSize').value),
        optimizer: 'adam',
        loss: appState.selectedModel === 'regression' ? 'mean_squared_error' : 'binary_crossentropy'
    };
    
    let code;
    
    switch (language) {
        case 'python':
            code = generatePythonCode(modelConfig, trainingConfig);
            break;
        case 'javascript':
            code = generateJavaScriptCode(modelConfig, trainingConfig);
            break;
        case 'deploy':
            code = generateDeploymentCode('my_model');
            break;
    }
    
    document.getElementById('generatedCode').querySelector('code').textContent = code;
}

function copyCode() {
    const code = document.getElementById('generatedCode').querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        alert('Code copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy code. Please copy manually.');
    });
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('flex');
    modal.classList.add('hidden');
}

// Model Details

function showModelDetails() {
    if (!appState.model) {
        alert('No trained model available!');
        return;
    }
    
    const modal = document.getElementById('detailsModal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    
    const summary = getModelSummary(appState.model);
    const totalParams = appState.model.countParams();
    
    const detailsHTML = `
        <div class="bg-gray-50 p-6 rounded-lg">
            <h3 class="text-xl font-bold text-gray-800 mb-4">Model Architecture</h3>
            <pre class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">${summary}</pre>
        </div>
        
        <div class="bg-gradient-to-br from-blue-50 to-white p-6 rounded-lg border-2 border-blue-200">
            <h3 class="text-xl font-bold text-gray-800 mb-4">Model Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p class="text-sm text-gray-600">Total Parameters</p>
                    <p class="text-2xl font-bold text-primary">${totalParams.toLocaleString()}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-600">Number of Layers</p>
                    <p class="text-2xl font-bold text-primary">${appState.model.layers.length}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-600">Input Shape</p>
                    <p class="text-lg font-bold text-gray-700">${JSON.stringify(appState.model.inputs[0].shape)}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-600">Output Shape</p>
                    <p class="text-lg font-bold text-gray-700">${JSON.stringify(appState.model.outputs[0].shape)}</p>
                </div>
            </div>
        </div>
        
        <div class="bg-gradient-to-br from-green-50 to-white p-6 rounded-lg border-2 border-green-200">
            <h3 class="text-xl font-bold text-gray-800 mb-4">Training Performance</h3>
            ${appState.trainingHistory.loss.length > 0 ? `
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="text-sm text-gray-600">Final Loss</p>
                        <p class="text-2xl font-bold text-red-600">${appState.trainingHistory.loss[appState.trainingHistory.loss.length - 1].toFixed(4)}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-600">Final Accuracy</p>
                        <p class="text-2xl font-bold text-green-600">${(appState.trainingHistory.accuracy[appState.trainingHistory.accuracy.length - 1] * 100).toFixed(2)}%</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-600">Total Epochs</p>
                        <p class="text-lg font-bold text-gray-700">${appState.trainingHistory.epochs.length}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-600">Model Type</p>
                        <p class="text-lg font-bold text-gray-700">${appState.selectedModel}</p>
                    </div>
                </div>
            ` : '<p class="text-gray-500 italic">No training history available</p>'}
        </div>
        
        <div class="bg-gradient-to-br from-purple-50 to-white p-6 rounded-lg border-2 border-purple-200">
            <h3 class="text-xl font-bold text-gray-800 mb-4">Layer Details</h3>
            <div class="space-y-3">
                ${appState.model.layers.map((layer, index) => `
                    <div class="bg-white p-4 rounded-lg border-l-4 border-purple-500">
                        <p class="font-bold text-purple-600">Layer ${index + 1}: ${layer.name} (${layer.getClassName()})</p>
                        <p class="text-sm text-gray-600 mt-2">${getLayerExplanation(layer.getClassName().toLowerCase().replace('layers.', ''))}</p>
                        <p class="text-xs text-gray-500 mt-2">Output Shape: ${JSON.stringify(layer.outputShape)}</p>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    document.getElementById('modelDetailsContent').innerHTML = detailsHTML;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await refreshSavedModels();
});
}

// Performance monitoring
console.log('TensorFlow.js backend:', tf.getBackend());
console.log('Memory:', tf.memory());

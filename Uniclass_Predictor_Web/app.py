"""
Flask Backend for Uniclass Predictor Web Application
"""
import os
from feature_merger import FeatureMerger

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import io
import numpy as np
import joblib
import dill

from custom_encoder import BaseEstimator, TransformerMixin, SentenceTransformerEncoder
from sentence_transformers import SentenceTransformer
from file_prediction_systems import FilePredictionSystem 
# --- END: Necessary Imports/Definitions ---
import __main__
__main__.FeatureMerger = FeatureMerger


app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
SAVED_MODEL_FOLDER = 'save_model' 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Model Loading ---
print("\n" + "="*60)
print("Loading Model Components...")
print("="*60)

MODEL = None
COLUMNS_TRAINING = None
LABEL_ENCODER = None
ID_TO_LABEL = {}
LABEL_TO_ID = {}

try:
# --- Use dill for the pipeline ---
    MODEL = joblib.load(os.path.join(SAVED_MODEL_FOLDER, 'best_pipeline.joblib'))
    print("✓ Model loaded: best_pipeline.joblib")
    
    # --- Other components stay with joblib ---
    COLUMNS_TRAINING = joblib.load(os.path.join(SAVED_MODEL_FOLDER, 'columns_training.joblib'))
    print("✓ Training columns loaded: columns_training.joblib")
    
    LABEL_ENCODER = joblib.load(os.path.join(SAVED_MODEL_FOLDER, 'label_encoder.joblib'))
    print("✓ Label encoder loaded: label_encoder.joblib")
    
    ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_ENCODER.classes_)}
    LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
    
    print(f"✓ Found {len(ID_TO_LABEL)} unique classes")
    
    # Ensure the SentenceTransformer models in the pipeline are loaded
    print("\n🔧 Initializing transformer models in pipeline...")
    if hasattr(MODEL, 'named_steps'):
        for step_name, step in MODEL.named_steps.items():
            if isinstance(step, SentenceTransformerEncoder):
                print(f"   Loading SentenceTransformer for step: {step_name}")
                if step.model is None:
                    step.model = SentenceTransformer(step.model_name, device=step.device)
                print(f"   ✓ {step_name} model loaded")
    
    print("="*60)
    print(f"✓ All model components loaded successfully!")
    print("="*60 + "\n")
    
except FileNotFoundError as e:
    print(f"✗ Error loading model components: {e}")
    print("\nEnsure the following files are in the 'saved_model/' directory:")
    print("  - best_pipeline.joblib")
    print("  - columns_training.joblib")
    print("  - label_encoder.joblib")
    print("="*60 + "\n")
except Exception as e:
    print(f"✗ An unexpected error occurred during model loading: {e}")
    import traceback
    traceback.print_exc()
    print("="*60 + "\n")


# HTML template (keeping your existing beautiful design)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Uniclass Predictor - AI-Powered Classification</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes blob {
            0%, 100% { transform: translate(0, 0) scale(1); }
            30% { transform: translate(20px, 50px) scale(1.1); }
            60% { transform: translate(-20px, 20px) scale(0.9); }
            90% { transform: translate(50px, -50px) scale(1.05); }
        }
        .animate-blob {
            animation: blob 8s infinite;
        }
        .animation-delay-2000 {
            animation-delay: 2s;
        }
        .animation-delay-4000 {
            animation-delay: 4s;
        }
        .gradient-bg {
            background:linear-gradient(135deg, #ffeedb 30%, #9fd8f3 100%);
        }
    </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-sky-50 via-pink-50 to-yellow-50">
    <!-- Animated Background -->
    <div class="absolute inset-0 overflow-hidden pointer-events-none">
        <div class="absolute top-20 left-10 w-72 h-72 bg-sky-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div class="absolute top-40 right-10 w-72 h-72 bg-yellow-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div class="absolute bottom-20 left-1/2 w-72 h-72 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
    </div>

    <div class="relative max-w-4xl mx-auto p-8">
        <!-- Header -->
        <div class="text-center mb-12 space-y-4">
            <div class="flex items-center justify-center gap-3 mb-4">
                <div class="p-3 gradient-bg rounded-2xl shadow-lg">
                    <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                    </svg>
                </div>
                <h1 class="text-5xl font-bold bg-gradient-to-r from-sky-400 via-pink-300 to-yellow-400 bg-clip-text text-transparent">
                    Uniclass Predictor
                </h1>
            </div>
            <p class="text-xl text-orange-300">AI-Powered Classification System</p>
            <p class="text-gray-400">Upload your data and let our machine learning model predict Uniclass codes .-.</p>
        </div>

        <!-- Main Card -->
        <div class="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl p-8 md:p-12 border border-white/30">
            <div id="uploadSection">
                <div class="text-center mb-6">
                    <h2 class="text-2xl font-semibold text-gray-600 mb-2">Upload Your File</h2>
                    <p class="text-gray-400">Support for CSV and Excel files (.csv, .xlsx, .xls)</p>
                </div>

                <!-- Upload Area -->
                <label id="uploadArea" class="relative block w-full p-12 border-4 border-dashed border-gray-400 rounded-3xl cursor-pointer hover:border-yellow-400 hover:bg-pink-50/30 transition-all duration-300 group">
                    <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" class="hidden">
                    <div class="flex flex-col items-center space-y-4">
                        <svg class="w-16 h-16 text-sky-300 group-hover:text-sky-300 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                        </svg>
                        <div class="text-center">
                            <p class="text-lg font-semibold text-gray-700 group-hover:text-sky-600">Click to upload or drag and drop</p>
                            <p class="text-sm text-gray-500 mt-1">CSV or Excel files only</p>
                        </div>
                    </div>
                </label>

                <div id="fileInfo" class="hidden mt-4 p-4 bg-green-50 border border-green-200 rounded-xl">
                    <div class="flex items-center gap-3">
                        <svg class="w-6 h-6 text-pink-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <div>
                            <p id="fileName" class="font-semibold text-gray-700"></p>
                            <p id="fileSize" class="text-sm text-gray-500"></p>
                        </div>
                    </div>
                </div>

                <div id="errorMessage" class="hidden mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
                    <div class="flex items-center gap-2">
                        <svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <p id="errorText" class="text-red-700"></p>
                    </div>
                </div>

                <button id="predictBtn" disabled class="w-full mt-6 py-4 px-8 bg-gray-200 text-gray-400 rounded-2xl font-semibold text-lg cursor-not-allowed transition-all duration-300">
                    <span class="flex items-center justify-center gap-3">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"></path>
                        </svg>
                        Predict Uniclass Codes
                    </span>
                </button>

                <div id="progressSection" class="hidden mt-6 space-y-2">
                    <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div id="progressBar" class="h-full bg-gradient-to-r from-sky-300 to-pink-300 transition-all duration-300 rounded-full" style="width: 0%"></div>
                    </div>
                    <div class="flex items-center justify-center gap-2 text-sm text-gray-600">
                        <svg class="w-4 h-4 text-yellow-500 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z"></path>
                        </svg>
                        <span id="progressText">AI model analyzing your data...</span>
                    </div>
                </div>

                <!-- Features -->
                <div class="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="text-center p-4 rounded-xl bg-gradient-to-br from-white/50 to-white/30 border border-white/50">
                        <div class="inline-flex p-3 bg-gradient-to-br from-sky-100 to-pink-100 rounded-xl mb-3">
                            <svg class="w-6 h-6 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                            </svg>
                        </div>
                        <h3 class="font-semibold text-gray-700 mb-1">AI-Powered</h3>
                        <p class="text-sm text-gray-500">Advanced ML model</p>
                    </div>
                    <div class="text-center p-4 rounded-xl bg-gradient-to-br from-white/50 to-white/30 border border-white/50">
                        <div class="inline-flex p-3 bg-gradient-to-br from-sky-100 to-pink-100 rounded-xl mb-3">
                            <svg class="w-6 h-6 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z"></path>
                            </svg>
                        </div>
                        <h3 class="font-semibold text-gray-700 mb-1">Fast Processing</h3>
                        <p class="text-sm text-gray-500">Results in seconds</p>
                    </div>
                    <div class="text-center p-4 rounded-xl bg-gradient-to-br from-white/30 to-white/30 border border-white/30">
                        <div class="inline-flex p-3 bg-gradient-to-br from-sky-100 to-pink-100 rounded-xl mb-3">
                            <svg class="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                        </div>
                        <h3 class="font-semibold text-gray-700 mb-1">Excel Export</h3>
                        <p class="text-sm text-gray-500">Download instantly</p>
                    </div>
                </div>
            </div>

            <!-- Results Section (Hidden by default) -->
            <div id="resultsSection" class="hidden space-y-6">
                <div class="text-center">
                    <div class="inline-flex p-4 bg-green-100 rounded-full mb-4">
                        <svg class="w-12 h-12 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <h2 class="text-3xl font-bold text-gray-700 mb-2">Predictions Complete!</h2>
                    <p class="text-gray-500">Your file has been successfully processed</p>
                </div>

                <!-- Stats Grid -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div class="p-4 bg-gradient-to-br from-sky-50 to-pink-50 rounded-xl text-center border border-sky-100">
                        <p class="text-sm text-gray-500 mb-1">Samples</p>
                        <p id="statSamples" class="text-2xl font-bold text-yellow-400">-</p>
                    </div>
                    <div class="p-4 bg-gradient-to-br from-sky-50 to-pink-50 rounded-xl text-center border border-sky-100">
                        <p class="text-sm text-gray-600 mb-1">Predictions</p>
                        <p id="statPredictions" class="text-2xl font-bold text-yellow-400">-</p>
                    </div>
                    <div class="p-4 bg-gradient-to-br from-sky-50 to-pink-50 rounded-xl text-center border border-sky-100">
                        <p class="text-sm text-gray-500 mb-1">Confidence</p>
                        <p id="statConfidence" class="text-2xl font-bold text-yellow-400">-</p>
                    </div>
                    <div class="p-4 bg-gradient-to-br from-sky-50 to-pink-50 rounded-xl text-center border border-sky-100">
                        <p class="text-sm text-gray-500 mb-1">Time</p>
                        <p id="statTime" class="text-2xl font-bold text-yellow-400">-</p>
                    </div>
                </div>

                <!-- Download Section -->
                <div class="p-6 bg-gradient-to-r from-sky-50 to-pink-50 rounded-2xl border border-sky-100">
                    <div class="flex items-center justify-between flex-wrap gap-4">
                        <div class="flex items-center gap-4">
                            <div class="p-3 bg-white rounded-xl shadow-sm">
                                <svg class="w-8 h-8 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                                </svg>
                            </div>
                            <div>
                                <p id="resultFileName" class="font-semibold text-gray-800">predictions.xlsx</p>
                                <p class="text-sm text-gray-600">Excel file with predictions</p>
                            </div>
                        </div>
                        <button id="downloadBtn" class="px-6 py-3 bg-gradient-to-r from-sky-400 to-yellow-400 hover:from-sky-400 hover:to-yellow-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all flex items-center gap-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                            </svg>
                            Download
                        </button>
                    </div>
                </div>

                <!-- Action Button -->
                <button id="resetBtn" class="w-full py-3 px-6 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold hover:bg-gray-50 transition-colors">
                    Upload Another File
                </button>
            </div>
        </div>

        <!-- Footer -->
        <div class="mt-8 text-center text-sm text-gray-500">
            <p>Powered by Caffeine from * Matcha * Coffee * Bubble Tea</p>
        </div>
    </div>

    <script>
        let selectedFile = null;
        let resultFileId = null;

        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const predictBtn = document.getElementById('predictBtn');
        const errorMessage = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        const progressSection = document.getElementById('progressSection');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const uploadSection = document.getElementById('uploadSection');
        const resultsSection = document.getElementById('resultsSection');
        const downloadBtn = document.getElementById('downloadBtn');
        const resetBtn = document.getElementById('resetBtn');

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleFileSelect(file);
            }
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('border-sky-400', 'bg-yellow-50');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('border-sky-400', 'bg-yellow-50');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('border-sky-400', 'bg-yellow-50');
            const file = e.dataTransfer.files[0];
            if (file) {
                handleFileSelect(file);
            }
        });

        function handleFileSelect(file) {
            const validExtensions = ['csv', 'xlsx', 'xls'];
            const fileExtension = file.name.split('.').pop().toLowerCase();

            if (!validExtensions.includes(fileExtension)) {
                showError('Please upload a CSV or Excel file');
                selectedFile = null;
                predictBtn.disabled = true;
                predictBtn.classList.remove('gradient-bg', 'text-white', 'hover:scale-105', 'shadow-lg');
                predictBtn.classList.add('bg-gray-200', 'text-gray-400', 'cursor-not-allowed');
                return;
            }

            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = `${(file.size / 1024).toFixed(2)} KB`;
            fileInfo.classList.remove('hidden');
            errorMessage.classList.add('hidden');
            
            uploadArea.classList.remove('border-gray-400');
            uploadArea.classList.add('border-green-400', 'bg-green-50');

            predictBtn.disabled = false;
            predictBtn.classList.remove('bg-gray-200', 'text-gray-400', 'cursor-not-allowed');
            predictBtn.classList.add('gradient-bg', 'text-white', 'hover:scale-105', 'shadow-lg');
        }

        function showError(message) {
            errorText.textContent = message;
            errorMessage.classList.remove('hidden');
            fileInfo.classList.add('hidden');
        }

        predictBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            predictBtn.disabled = true;
            progressSection.classList.remove('hidden');
            
            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += 10;
                    if (progress <= 90) {
                        progressBar.style.width = progress + '%';
                    }
                }, 200);

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                clearInterval(progressInterval);
                progressBar.style.width = '100%';

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Prediction failed');
                }

                const result = await response.json();
                
                setTimeout(() => {
                    displayResults(result);
                }, 500);

            } catch (error) {
                clearInterval(progressInterval);
                showError(error.message);
                predictBtn.disabled = false;
                progressSection.classList.add('hidden');
                progressBar.style.width = '0%';
            }
        });

        function displayResults(result) {
            uploadSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');

            document.getElementById('statSamples').textContent = result.samples.toLocaleString();
            document.getElementById('statPredictions').textContent = result.predictions.toLocaleString();
            document.getElementById('statConfidence').textContent = result.avg_confidence;
            document.getElementById('statTime').textContent = result.processing_time;
            document.getElementById('resultFileName').textContent = result.filename;
            
            resultFileId = result.file_id;
        }

        downloadBtn.addEventListener('click', () => {
            if (resultFileId) {
                window.location.href = `/api/download/${resultFileId}`;
            }
        });

        resetBtn.addEventListener('click', () => {
            resultsSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            selectedFile = null;
            resultFileId = null;
            fileInput.value = '';
            fileInfo.classList.add('hidden');
            errorMessage.classList.add('hidden');
            progressSection.classList.add('hidden');
            progressBar.style.width = '0%';
            uploadArea.classList.remove('border-green-400', 'bg-green-50');
            uploadArea.classList.add('border-gray-400');
            predictBtn.disabled = true;
            predictBtn.classList.remove('gradient-bg', 'text-white', 'hover:scale-105', 'shadow-lg');
            predictBtn.classList.add('bg-gray-200', 'text-gray-400', 'cursor-not-allowed');
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main page"""
    if MODEL is None:
        return render_template_string("""
            <div style="padding: 50px; text-align: center; font-family: sans-serif;">
                <h1 style="color: red;">FATAL ERROR: Model Not Loaded</h1>
                <p>The backend server could not load the required machine learning artifacts (best_pipeline.joblib, etc.). Please check the console output for file path errors.</p>
            </div>
        """)
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    start_time = time.time()
    
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Server configuration error.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file extension
    allowed_extensions = {'csv', 'xlsx', 'xls'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload CSV or Excel file.'}), 400
    
    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_filename = f"upload_{timestamp}.{file_ext}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)
    
    try:
        # Initialize prediction system
        pred_system = FilePredictionSystem(
            model=MODEL,
            id_to_label=ID_TO_LABEL,
            label_to_id=LABEL_TO_ID,
            target_column='X_AST Uniclass', 
            text_column='merged_text',
            columns_to_merge=COLUMNS_TRAINING,
            auto_merge=False
        )
        
        # Make predictions
        results_df = pred_system.predict_file(
            file_path=input_path,
            has_actual_labels=False,  
            top_n_predictions=1
        )
        
        # Save output file
        output_filename = f"predictions_{timestamp}.xlsx"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        output_path_obj = Path(output_path)
        
        # Export results
        pred_system.export_results(
            results_df=results_df,
            output_path=output_path_obj,
            include_metrics=False,
            include_visualization=False,
            create_summary_sheet=False
        )
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_confidence = results_df['Prediction_Confidence'].mean() if 'Prediction_Confidence' in results_df.columns else 0.0
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'file_id': timestamp,
            'samples': len(results_df),
            'predictions': len(results_df),
            'avg_confidence': f"{avg_confidence:.1%}",
            'processing_time': f"{processing_time:.1f}s"
        })
        
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        print(f"Prediction Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Processing Error: {str(e)}"}), 500

@app.route('/api/download/<file_id>')
def download(file_id):
    """Download prediction results"""
    try:
        output_filename = f"predictions_{file_id}.xlsx"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        if not os.path.exists(output_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'timestamp': datetime.now().isoformat()
    })
    

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Uniclass Predictor Server Starting...")
    print("="*60)
    print(f"✓ Model loaded: {MODEL is not None}")
    print(f"✓ Labels loaded: {len(ID_TO_LABEL)} classes")
    print(f"✓ Server running on: http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
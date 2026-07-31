document.addEventListener('DOMContentLoaded', () => {
    // ==========================================================================
    // 0. ONBOARDING POPUP
    // ==========================================================================
    const onboardingOverlay = document.getElementById('onboarding-overlay');
    const onboardingProceedBtn = document.getElementById('onboarding-proceed-btn');
    const onboardingSkip = document.getElementById('onboarding-skip');

    function dismissOnboarding() {
        onboardingOverlay.classList.add('hiding');
        setTimeout(() => {
            onboardingOverlay.style.display = 'none';
        }, 420);
    }

    if (onboardingProceedBtn) {
        onboardingProceedBtn.addEventListener('click', dismissOnboarding);
    }
    if (onboardingSkip) {
        onboardingSkip.addEventListener('click', dismissOnboarding);
    }

    // State management
    let activeFile = null;
    let cameraStream = null;
    let currentPredictionId = null;
    let diseaseInfoCache = {}; // Cache to retrieve info from history clicks
    let diseaseDatabase = {};  // Stores local disease dataset loaded from api
    let uniquePlants = new Set();

    // Navigation and Tab Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // AI Scanner Elements
    const tabUpload = document.getElementById('tab-upload');
    const tabCamera = document.getElementById('tab-camera');
    const uploadZone = document.getElementById('upload-zone');
    const cameraZone = document.getElementById('camera-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    
    const videoStream = document.getElementById('video-stream');
    const captureCanvas = document.getElementById('capture-canvas');
    const btnSnap = document.getElementById('btn-snap');
    const btnStopCamera = document.getElementById('btn-stop-camera');
    const btnReset = document.getElementById('btn-reset');
    const btnAnalyze = document.getElementById('btn-analyze');
    const cropSelect = document.getElementById('crop-select');

    const resultsPanel = document.getElementById('results-panel');
    const resultsEmpty = document.getElementById('results-empty');
    const resultsLoading = document.getElementById('results-loading');
    const resultsContent = document.getElementById('results-content');

    const resPlant = document.getElementById('res-plant');
    const resDisease = document.getElementById('res-disease');
    const resConfidence = document.getElementById('res-confidence');
    const resConfidenceBar = document.getElementById('res-confidence-bar');
    
    const descText = document.getElementById('desc-text');
    const symptomsText = document.getElementById('symptoms-text');
    const treatmentText = document.getElementById('treatment-text');
    const infoTabs = document.querySelectorAll('.info-tab');
    const infoPanes = document.querySelectorAll('.info-pane');

    const btnFeedbackYes = document.getElementById('btn-feedback-yes');
    const btnFeedbackNo = document.getElementById('btn-feedback-no');
    const feedbackBtns = document.getElementById('feedback-btns');
    const feedbackThanks = document.getElementById('feedback-thanks');
    const feedbackPrompt = document.getElementById('feedback-prompt');

    // Correction Form Elements
    const correctionFormPanel = document.getElementById('correction-form-panel');
    const correctionSelect = document.getElementById('correction-select');
    const correctionComments = document.getElementById('correction-comments');
    const btnCancelCorrection = document.getElementById('btn-cancel-correction');
    const btnSubmitCorrection = document.getElementById('btn-submit-correction');

    // History Elements
    const historyGrid = document.getElementById('history-grid');
    const historyCount = document.getElementById('history-count');
    const noHistoryText = document.getElementById('no-history-text');

    // Encyclopedia Elements
    const encyclopediaGrid = document.getElementById('encyclopedia-grid');
    const encyclopediaSearch = document.getElementById('encyclopedia-search');
    const encyclopediaFilter = document.getElementById('encyclopedia-filter');
    const encyclopediaCount = document.getElementById('encyclopedia-count');

    // Stats Elements
    const statTotalScans = document.getElementById('stat-total-scans');
    const statAccuracy = document.getElementById('stat-accuracy');
    const statCorrect = document.getElementById('stat-correct');
    const statIncorrect = document.getElementById('stat-incorrect');
    const statsDistributionChart = document.getElementById('stats-distribution-chart');
    const noStatsText = document.getElementById('no-stats-text');

    // ==========================================================================
    // 1. NAVIGATION & THEME SYSTEM
    // ==========================================================================
    
    // Tab switching
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.add('hidden'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.remove('hidden');

            if (targetTab === 'encyclopedia') {
                ensureDiseaseDatabaseLoaded().then(() => renderEncyclopedia());
            } else if (targetTab === 'analytics') {
                loadStats();
            }
        });
    });

    // ==========================================================================
    // 2. FILE UPLOAD & CAMERA SYSTEM
    // ==========================================================================
    
    tabUpload.addEventListener('click', () => {
        switchMediaTab('upload');
    });

    tabCamera.addEventListener('click', () => {
        switchMediaTab('camera');
    });

    function switchMediaTab(mode) {
        if (mode === 'upload') {
            tabUpload.classList.add('active');
            tabCamera.classList.remove('active');
            uploadZone.classList.remove('hidden');
            cameraZone.classList.add('hidden');
            stopCamera();
        } else {
            tabUpload.classList.remove('active');
            tabCamera.classList.add('active');
            uploadZone.classList.add('hidden');
            cameraZone.classList.remove('hidden');
            previewContainer.classList.remove('scanning');
            previewContainer.classList.add('hidden');
            activeFile = null;
            startCamera();
        }
    }

    async function startCamera() {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' },
                audio: false
            });
            videoStream.srcObject = cameraStream;
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Unable to access camera. Please upload an image instead.');
            switchMediaTab('upload');
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        videoStream.srcObject = null;
    }

    btnStopCamera.addEventListener('click', () => {
        switchMediaTab('upload');
    });

    btnSnap.addEventListener('click', () => {
        if (!videoStream.srcObject) return;

        captureCanvas.width = videoStream.videoWidth;
        captureCanvas.height = videoStream.videoHeight;
        
        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(videoStream, 0, 0, captureCanvas.width, captureCanvas.height);
        
        captureCanvas.toBlob((blob) => {
            const file = new File([blob], 'captured_leaf.jpg', { type: 'image/jpeg' });
            handleSelectedFile(file);
            stopCamera();
            uploadZone.classList.add('hidden');
            cameraZone.classList.add('hidden');
        }, 'image/jpeg', 0.95);
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (PNG/JPEG).');
            return;
        }
        activeFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            uploadZone.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    btnReset.addEventListener('click', (e) => {
        e.stopPropagation();
        activeFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        previewContainer.classList.remove('scanning');
        if (tabUpload.classList.contains('active')) {
            uploadZone.classList.remove('hidden');
        } else {
            startCamera();
            cameraZone.classList.remove('hidden');
        }
    });

    // ==========================================================================
    // 3. AI DIAGNOSTICS & RESULTS
    // ==========================================================================
    
    btnAnalyze.addEventListener('click', async () => {
        if (!activeFile) return;

        // UI scanning states
        previewContainer.classList.add('scanning');
        resultsPanel.classList.remove('empty');
        resultsEmpty.classList.add('hidden');
        resultsLoading.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        correctionFormPanel.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', activeFile);
        formData.append('crop', cropSelect.value);
        console.log('[LEAF-LENZ] Sending crop:', cropSelect.value);

        try {
            const response = await fetch('/api/predict/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('API Diagnosis Error');

            const data = await response.json();
            previewContainer.classList.remove('scanning');
            displayResults(data);
            loadHistory();
        } catch (err) {
            console.error('Diagnosis error:', err);
            previewContainer.classList.remove('scanning');
            alert('Failed to connect to leaf diagnostic engine. Please ensure the server is running.');
            resetDiagnosticUI();
        }
    });

    function resetDiagnosticUI() {
        resultsPanel.classList.add('empty');
        resultsEmpty.classList.remove('hidden');
        resultsLoading.classList.add('hidden');
        resultsContent.classList.add('hidden');
        previewContainer.classList.remove('hidden');
    }

    function displayResults(data) {
        resultsLoading.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        currentPredictionId = data.prediction_id;
        
        feedbackBtns.classList.remove('hidden');
        feedbackThanks.classList.add('hidden');
        feedbackPrompt.textContent = "Was this prediction accurate?";
        correctionFormPanel.classList.add('hidden');

        const plantName = data.prediction.plant;
        const diseaseName = data.prediction.disease;
        const isHealthy = diseaseName.toLowerCase() === 'healthy';
        const isInvalid = diseaseName.toLowerCase() === 'invalid image';

        resPlant.textContent = plantName;
        if (isInvalid) {
            resDisease.textContent = '❌ Invalid Image';
        } else {
            resDisease.textContent = isHealthy ? '✅ Healthy!' : diseaseName;
        }

        // Style the header card
        const headerCard = document.getElementById('prediction-header-card');
        const iconEl = document.getElementById('disease-icon-el');
        const subtitleEl = document.getElementById('disease-subtitle');
        headerCard.classList.remove('result-healthy', 'result-disease', 'result-invalid');

        if (isInvalid) {
            headerCard.classList.add('result-invalid');
            iconEl.className = 'fa-solid fa-circle-question';
            subtitleEl.textContent = 'Upload a leaf';
        } else if (isHealthy) {
            headerCard.classList.add('result-healthy');
            iconEl.className = 'fa-solid fa-heart-pulse';
            subtitleEl.textContent = 'No Disease Detected';
        } else {
            headerCard.classList.add('result-disease');
            iconEl.className = 'fa-solid fa-virus';
            subtitleEl.textContent = 'Disease Detected';
        }

        const confidencePct = (data.prediction.confidence * 100).toFixed(1);
        resConfidence.textContent = `${confidencePct}%`;
        resConfidenceBar.style.width = '0%';
        setTimeout(() => {
            resConfidenceBar.style.width = `${confidencePct}%`;
        }, 150);

        // Crop mismatch warning
        const mismatchWarning = document.getElementById('crop-mismatch-warning');
        const mismatchText = document.getElementById('crop-mismatch-text');
        if (data.crop_mismatch) {
            mismatchText.textContent = data.crop_mismatch;
            mismatchWarning.classList.remove('hidden');
        } else {
            mismatchWarning.classList.add('hidden');
        }

        descText.textContent = data.disease_info.description || "N/A";
        symptomsText.textContent = data.disease_info.symptoms || "N/A";
        treatmentText.textContent = data.disease_info.treatment_and_prevention || "N/A";
    }

    // Results panel tab navigation
    infoTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            infoTabs.forEach(t => t.classList.remove('active'));
            infoPanes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const target = tab.getAttribute('data-target');
            document.getElementById(target).classList.add('active');
        });
    });

    // ==========================================================================
    // 4. FEEDBACK & CORRECTION SYSTEM
    // ==========================================================================
    
    async function sendFeedback(isCorrect) {
        if (!currentPredictionId) return;

        feedbackBtns.classList.add('hidden');
        feedbackPrompt.textContent = "Saving...";

        try {
            await fetch('/api/feedback/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_id: currentPredictionId,
                    is_correct: isCorrect
                })
            });
            feedbackPrompt.textContent = "";
            feedbackThanks.classList.remove('hidden');
        } catch (err) {
            console.error('Error submitting feedback:', err);
            feedbackBtns.classList.remove('hidden');
            feedbackPrompt.textContent = "Was this prediction accurate?";
        }
    }

    btnFeedbackYes.addEventListener('click', () => sendFeedback(true));
    
    btnFeedbackNo.addEventListener('click', () => {
        // Show advanced correction form
        feedbackBtns.classList.add('hidden');
        feedbackPrompt.textContent = "";
        
        ensureDiseaseDatabaseLoaded().then(() => {
            populateCorrectionDropdown();
            correctionFormPanel.classList.remove('hidden');
            correctionComments.value = '';
        });
    });

    btnCancelCorrection.addEventListener('click', () => {
        correctionFormPanel.classList.add('hidden');
        feedbackBtns.classList.remove('hidden');
        feedbackPrompt.textContent = "Was this prediction accurate?";
    });

    btnSubmitCorrection.addEventListener('click', async () => {
        if (!currentPredictionId) return;

        const selectedLabel = correctionSelect.value;
        const comment = correctionComments.value.trim();

        btnSubmitCorrection.disabled = true;
        btnSubmitCorrection.textContent = "Submitting...";

        try {
            const response = await fetch('/api/feedback/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_id: currentPredictionId,
                    is_correct: false,
                    corrected_label: selectedLabel,
                    user_feedback: comment
                })
            });

            if (!response.ok) throw new Error('Failed to submit correction');

            correctionFormPanel.classList.add('hidden');
            feedbackThanks.classList.remove('hidden');
            
            // Reload logs and stats
            loadHistory();
        } catch (err) {
            console.error(err);
            alert('Unable to submit correction. Please try again.');
        } finally {
            btnSubmitCorrection.disabled = false;
            btnSubmitCorrection.textContent = "Submit Correction";
        }
    });

    function populateCorrectionDropdown() {
        correctionSelect.innerHTML = '';
        
        // Populate group keys (except fallback)
        Object.keys(diseaseDatabase).forEach(key => {
            if (key === 'fallback') return;
            const info = diseaseDatabase[key];
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = `${info.plant_name} - ${info.disease_name}`;
            correctionSelect.appendChild(opt);
        });
    }

    // ==========================================================================
    // 5. ENCYCLOPEDIA SYSTEM
    // ==========================================================================
    
    async function ensureDiseaseDatabaseLoaded() {
        if (Object.keys(diseaseDatabase).length > 0) return;

        try {
            const response = await fetch('/api/diseases/');
            if (!response.ok) return;

            diseaseDatabase = await response.json();
            
            // Extract unique plants
            uniquePlants.clear();
            Object.keys(diseaseDatabase).forEach(key => {
                if (key === 'fallback') return;
                uniquePlants.add(diseaseDatabase[key].plant_name);
            });

            // Populate filter select options
            encyclopediaFilter.innerHTML = '<option value="all">All Plants</option>';
            Array.from(uniquePlants).sort().forEach(plant => {
                const opt = document.createElement('option');
                opt.value = plant;
                opt.textContent = plant;
                encyclopediaFilter.appendChild(opt);
            });
        } catch (err) {
            console.error('Error fetching disease data:', err);
        }
    }

    function renderEncyclopedia() {
        encyclopediaGrid.innerHTML = '';
        
        const filterVal = encyclopediaFilter.value.toLowerCase();
        const searchVal = encyclopediaSearch.value.trim().toLowerCase();

        let visibleCount = 0;

        Object.keys(diseaseDatabase).forEach(key => {
            if (key === 'fallback') return;
            const info = diseaseDatabase[key];
            
            // Filters validation
            const plantMatches = filterVal === 'all' || info.plant_name.toLowerCase() === filterVal;
            const searchMatches = !searchVal || 
                info.plant_name.toLowerCase().includes(searchVal) ||
                info.disease_name.toLowerCase().includes(searchVal) ||
                info.description.toLowerCase().includes(searchVal) ||
                info.symptoms.toLowerCase().includes(searchVal) ||
                info.treatment_prevention.toLowerCase().includes(searchVal);

            if (plantMatches && searchMatches) {
                visibleCount++;
                const card = document.createElement('div');
                card.className = 'encyco-card';
                card.innerHTML = `
                    <div class="encyco-card-header">
                        <div class="encyco-card-title">
                            <span>${info.plant_name}</span>
                            <h3>${info.disease_name}</h3>
                        </div>
                        <i class="fa-solid fa-chevron-down chevron-icon"></i>
                    </div>
                    <div class="encyco-card-body collapsed">
                        <div class="encyco-section">
                            <h4><i class="fa-solid fa-circle-info"></i> About the Condition</h4>
                            <p>${info.description}</p>
                        </div>
                        <div class="encyco-section">
                            <h4><i class="fa-solid fa-list-check"></i> Symptoms</h4>
                            <p>${info.symptoms}</p>
                        </div>
                        <div class="encyco-section">
                            <h4><i class="fa-solid fa-hand-holding-medical"></i> Treatment & Prevention</h4>
                            <p>${info.treatment_prevention}</p>
                        </div>
                    </div>
                `;

                const header = card.querySelector('.encyco-card-header');
                const body = card.querySelector('.encyco-card-body');
                
                header.addEventListener('click', () => {
                    const isExpanded = card.classList.toggle('expanded');
                    if (isExpanded) {
                        body.classList.remove('collapsed');
                        // Calculate smooth height transition
                        body.style.maxHeight = 'none';
                        const height = body.scrollHeight + 'px';
                        body.style.maxHeight = '0px';
                        setTimeout(() => {
                            body.style.maxHeight = height;
                        }, 10);
                        setTimeout(() => {
                            body.style.maxHeight = 'none';
                        }, 310);
                    } else {
                        body.style.maxHeight = body.scrollHeight + 'px';
                        setTimeout(() => {
                            body.style.maxHeight = '0px';
                        }, 10);
                        setTimeout(() => {
                            body.classList.add('collapsed');
                        }, 300);
                    }
                });

                encyclopediaGrid.appendChild(card);
            }
        });

        encyclopediaCount.textContent = `${visibleCount} condition${visibleCount !== 1 ? 's' : ''}`;
    }

    encyclopediaSearch.addEventListener('input', renderEncyclopedia);
    encyclopediaFilter.addEventListener('change', renderEncyclopedia);

    // ==========================================================================
    // 6. STATISTICS & ANALYTICS DASHBOARD
    // ==========================================================================
    
    async function loadStats() {
        try {
            const response = await fetch('/api/stats/');
            if (!response.ok) return;

            const data = await response.json();
            
            // Fill general metrics
            statTotalScans.textContent = data.total_scans;
            statAccuracy.textContent = `${(data.accuracy_rate * 100).toFixed(0)}%`;
            statCorrect.textContent = data.correct_scans;
            statIncorrect.textContent = data.incorrect_scans;

            // Fill distribution chart
            statsDistributionChart.innerHTML = '';
            
            if (data.plant_distribution && data.plant_distribution.length > 0) {
                if (noStatsText) noStatsText.classList.add('hidden');
                
                const maxCount = Math.max(...data.plant_distribution.map(d => d.count), 1);
                
                data.plant_distribution.forEach(item => {
                    const percentage = ((item.count / maxCount) * 100).toFixed(1);
                    const row = document.createElement('div');
                    row.className = 'distribution-row';
                    row.innerHTML = `
                        <div class="distribution-label-bar">
                            <span>${item.plant_name}</span>
                            <span>${item.count} scan${item.count !== 1 ? 's' : ''}</span>
                        </div>
                        <div class="distribution-bar-bg">
                            <div class="distribution-bar-fill" style="width: 0%"></div>
                        </div>
                    `;
                    statsDistributionChart.appendChild(row);
                    
                    // Animate the fill
                    setTimeout(() => {
                        const bar = row.querySelector('.distribution-bar-fill');
                        if (bar) bar.style.width = `${percentage}%`;
                    }, 50);
                });
            } else {
                if (noStatsText) {
                    noStatsText.classList.remove('hidden');
                    statsDistributionChart.appendChild(noStatsText);
                } else {
                    statsDistributionChart.innerHTML = `
                        <div class="no-stats-state">
                            <p><i class="fa-solid fa-chart-pie"></i> No scan data recorded yet.</p>
                        </div>
                    `;
                }
            }
        } catch (err) {
            console.error('Error loading analytics:', err);
        }
    }

    // ==========================================================================
    // 7. DIAGNOSTIC SCAN HISTORY
    // ==========================================================================
    
    async function loadHistory() {
        try {
            const response = await fetch('/api/history/');
            if (!response.ok) return;

            const data = await response.json();
            renderHistory(data);
        } catch (err) {
            console.error('Error loading history:', err);
        }
    }

    function renderHistory(logs) {
        historyGrid.innerHTML = '';
        historyCount.textContent = `${logs.length} item${logs.length !== 1 ? 's' : ''}`;

        if (logs.length === 0) {
            noHistoryText.classList.remove('hidden');
            historyGrid.appendChild(noHistoryText);
            return;
        }

        noHistoryText.classList.add('hidden');
        diseaseInfoCache = {}; // Clear cache

        logs.forEach(log => {
            diseaseInfoCache[log.id] = log;

            const card = document.createElement('div');
            card.className = 'history-card';
            card.dataset.id = log.id;

            const date = new Date(log.created_at);
            const dateStr = date.toLocaleDateString(undefined, { 
                month: 'short', 
                day: 'numeric', 
                hour: '2-digit', 
                minute: '2-digit' 
            });

            const confidencePct = (log.confidence * 100).toFixed(0);

            card.innerHTML = `
                <img src="${log.image}" class="history-thumb" alt="Thumbnail">
                <div class="history-info">
                    <p>${log.plant_name}</p>
                    <h4>${log.disease_name}</h4>
                    <span>${dateStr}</span>
                </div>
            `;

            card.addEventListener('click', () => {
                loadHistoryItemIntoResults(log.id);
            });

            historyGrid.appendChild(card);
        });
    }

    async function loadHistoryItemIntoResults(id) {
        const log = diseaseInfoCache[id];
        if (!log) return;

        resultsPanel.classList.remove('empty');
        resultsEmpty.classList.add('hidden');
        resultsLoading.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        currentPredictionId = log.id;
        correctionFormPanel.classList.add('hidden');

        if (log.is_correct !== null) {
            feedbackBtns.classList.add('hidden');
            feedbackThanks.classList.remove('hidden');
            feedbackPrompt.textContent = "";
        } else {
            feedbackBtns.classList.remove('hidden');
            feedbackThanks.classList.add('hidden');
            feedbackPrompt.textContent = "Was this prediction accurate?";
        }

        const isHealthy = log.disease_name.toLowerCase() === 'healthy';
        const isInvalid = log.disease_name.toLowerCase() === 'invalid image';
        
        resPlant.textContent = log.plant_name;
        if (isInvalid) {
            resDisease.textContent = '❌ Invalid Image';
        } else {
            resDisease.textContent = isHealthy ? '✅ Healthy!' : log.disease_name;
        }

        // Apply health/disease styling
        const headerCard = document.getElementById('prediction-header-card');
        const iconEl = document.getElementById('disease-icon-el');
        const subtitleEl = document.getElementById('disease-subtitle');
        headerCard.classList.remove('result-healthy', 'result-disease', 'result-invalid');
        
        if (isInvalid) {
            headerCard.classList.add('result-invalid');
            iconEl.className = 'fa-solid fa-circle-question';
            subtitleEl.textContent = 'Upload a leaf';
        } else if (isHealthy) {
            headerCard.classList.add('result-healthy');
            iconEl.className = 'fa-solid fa-heart-pulse';
            subtitleEl.textContent = 'No Disease Detected';
        } else {
            headerCard.classList.add('result-disease');
            iconEl.className = 'fa-solid fa-virus';
            subtitleEl.textContent = 'Disease Detected';
        }

        const confidencePct = (log.confidence * 100).toFixed(1);
        resConfidence.textContent = `${confidencePct}%`;
        resConfidenceBar.style.width = `${confidencePct}%`;

        descText.textContent = log.description || "N/A";
        symptomsText.textContent = log.symptoms || "N/A";
        treatmentText.textContent = log.treatment_prevention || "N/A";
    }

    // Bootstrap
    loadHistory();
});

// Global Chart Instances
let ltChart = null;
let pendingChart = null;
let pendingUserChart = null;
let pendingErsChart = null;
let weekChart = null;
let categoryChart = null;
let repairChart = null;
let modalChart = null;

// Registry data storage
let registryData = [];

// Initialize Page
document.addEventListener('DOMContentLoaded', () => {
    loadFleetData();
    loadMotorRegistry();
    populateInspectionDropdown();
});

// View Navigation
function switchView(viewId) {
    // Hide all view sections
    document.querySelectorAll('.view-section').forEach(section => {
        section.style.display = 'none';
        section.classList.remove('active');
    });
    
    // Deactivate all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected view
    const activeSection = document.getElementById(`view-${viewId}`);
    if (activeSection) {
        activeSection.style.display = 'block';
        activeSection.classList.add('active');
    }
    
    // Activate clicked tab button
    const activeTab = document.getElementById(`tab-${viewId}`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
}

// ---------------------------------------------------------------- //
// API ACTIONS & DATA LOADING
// ---------------------------------------------------------------- //

async function loadFleetData() {
    try {
        const response = await fetch('/api/dashboard/stats');
        const data = await response.json();
        
        if (data.error) {
            console.error("Dashboard stats loading failed:", data.error);
            return;
        }

        renderDashboardCharts(data.dashboard);
        renderCategoryChart(data.categories);
        populateTurnaroundTables(data.turnaround);
        
        // Generate baseline weekly metrics
        renderWeekComparisonChart();
        renderRepairAnalysisChart();
    } catch (err) {
        console.error("Failed to fetch dashboard statistics:", err);
    }
}

async function loadMotorRegistry() {
    try {
        const response = await fetch('/api/motors/registry');
        registryData = await response.json();
        
        if (registryData.error) {
            console.error("Registry loading failed:", registryData.error);
            return;
        }

        populateRegistryTable(registryData);
    } catch (err) {
        console.error("Failed to load equipment registry:", err);
    }
}

async function populateInspectionDropdown() {
    try {
        const response = await fetch('/api/motors');
        const data = await response.json();
        
        const select = document.getElementById('inspect-motor-select');
        select.innerHTML = '<option value="">Select Motor Serial...</option>';
        data.motors.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to populate inspection dropdown:", err);
    }
}

// ---------------------------------------------------------------- //
// TABLE BUILDERS
// ---------------------------------------------------------------- //

function populateRegistryTable(data) {
    const tbody = document.getElementById('registry-tbody');
    tbody.innerHTML = '';
    
    data.forEach(motor => {
        const tr = document.createElement('tr');
        
        // Risk highlighting class
        let riskClass = '';
        if (motor.predicted_risk === 'CRITICAL') riskClass = 'bold text-danger';
        else if (motor.predicted_risk === 'HIGH') riskClass = 'bold text-warning';
        
        tr.innerHTML = `
            <td>${motor.equipment_id}</td>
            <td>${motor.allotment || ''}</td>
            <td>${motor.main_equipment_name}</td>
            <td class="bold">${motor.equipment_serial_no}</td>
            <td>${motor.motor_manufacturing_year}</td>
            <td>${motor.make}</td>
            <td>${motor.model}</td>
            <td>${motor.kw}</td>
            <td>${motor.frame}</td>
            <td>${motor.rpm}</td>
            <td>${motor.voltage}</td>
            <td>${motor.installed_location}</td>
            <td><button class="btn-analyze" onclick="openDiagnosticsModal('${motor.equipment_serial_no}')">Analyze</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function filterRegistryTable() {
    const searchVal = document.getElementById('registry-search').value.toLowerCase();
    const filtered = registryData.filter(m => 
        m.equipment_serial_no.toLowerCase().includes(searchVal) ||
        m.make.toLowerCase().includes(searchVal) ||
        m.model.toLowerCase().includes(searchVal) ||
        m.main_equipment_name.toLowerCase().includes(searchVal) ||
        m.installed_location.toLowerCase().includes(searchVal)
    );
    populateRegistryTable(filtered);
}

function populateTurnaroundTables(turnaround) {
    // Rewinding elements
    document.getElementById('rew-0-7').textContent = turnaround.rewinding["0_7"];
    document.getElementById('rew-8-15').textContent = turnaround.rewinding["8_15"];
    document.getElementById('rew-15-30').textContent = turnaround.rewinding["15_30"];
    document.getElementById('rew-30-plus').textContent = turnaround.rewinding["30_plus"];
    document.getElementById('rew-total').textContent = turnaround.rewinding["total"];
    
    // Overhauling elements
    document.getElementById('ovh-0-3').textContent = turnaround.overhauling["0_3"];
    document.getElementById('ovh-4-7').textContent = turnaround.overhauling["4_7"];
    document.getElementById('ovh-7-15').textContent = turnaround.overhauling["7_15"];
    document.getElementById('ovh-15-plus').textContent = turnaround.overhauling["15_plus"];
    document.getElementById('ovh-total').textContent = turnaround.overhauling["total"];
}

// ---------------------------------------------------------------- //
// CHART RENDERING (CHART.JS)
// ---------------------------------------------------------------- //

function renderDashboardCharts(stats) {
    // 1. LT Motors Data (Bar Chart)
    const ctxLt = document.getElementById('ltMotorsChart').getContext('2d');
    if (ltChart) ltChart.destroy();
    
    ltChart = new Chart(ctxLt, {
        type: 'bar',
        data: {
            labels: ['IN REPAIR', 'REPAIRED'],
            datasets: [
                {
                    label: 'Received',
                    data: [stats.lt_motors.received, 0],
                    backgroundColor: '#d32f2f', // Brand Alert Color
                    borderRadius: 4
                },
                {
                    label: 'Repaired',
                    data: [0, stats.lt_motors.repaired],
                    backgroundColor: '#2e7d32', // Brand Success Color
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // 2. Pending Motors Status (Pie Chart)
    const ctxPending = document.getElementById('pendingMotorsChart').getContext('2d');
    if (pendingChart) pendingChart.destroy();
    
    pendingChart = new Chart(ctxPending, {
        type: 'pie',
        data: {
            labels: ['ERS', 'User Department'],
            datasets: [{
                data: [stats.pending_status.ers, stats.pending_status.user_dept],
                backgroundColor: ['#154284', '#c62828'] // Brand Navy, Brand Red
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });

    // 3. Pending at User Department (Horizontal Bar Chart)
    const ctxUser = document.getElementById('pendingUserChart').getContext('2d');
    if (pendingUserChart) pendingUserChart.destroy();
    
    pendingUserChart = new Chart(ctxUser, {
        type: 'bar',
        data: {
            labels: ['Bearings Required', 'Mechanical Jobs', 'Spares Required'],
            datasets: [{
                data: [
                    stats.pending_user.bearings_required, 
                    stats.pending_user.mechanical_jobs, 
                    stats.pending_user.spares_required
                ],
                backgroundColor: '#c62828', // Primary alert highlight
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });

    // 4. Pending at ERS (Horizontal Bar Chart)
    const ctxErs = document.getElementById('pendingErsChart').getContext('2d');
    if (pendingErsChart) pendingErsChart.destroy();
    
    pendingErsChart = new Chart(ctxErs, {
        type: 'bar',
        data: {
            labels: ['Work U/P', 'Yet to start'],
            datasets: [{
                data: [stats.pending_ers.work_up, stats.pending_ers.yet_to_start],
                backgroundColor: '#154284', // Primary brand color
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });
}

function renderCategoryChart(categories) {
    const ctxCat = document.getElementById('categoryAnalysisChart').getContext('2d');
    if (categoryChart) categoryChart.destroy();
    
    const labels = categories.map(c => c.category);
    const received = categories.map(c => c.received_outstanding);
    const repaired = categories.map(c => c.repaired);
    const atErs = categories.map(c => c.at_ers);
    
    categoryChart = new Chart(ctxCat, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Received + Outstanding',
                    data: received,
                    backgroundColor: '#3b82f6',
                    borderRadius: 4
                },
                {
                    label: 'Repaired',
                    data: repaired,
                    backgroundColor: '#ef4444',
                    borderRadius: 4
                },
                {
                    label: 'At ERS',
                    data: atErs,
                    backgroundColor: '#1e1b4b',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function renderWeekComparisonChart() {
    const ctxWeek = document.getElementById('weekComparisonChart').getContext('2d');
    if (weekChart) weekChart.destroy();
    
    weekChart = new Chart(ctxWeek, {
        type: 'line',
        data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8'],
            datasets: [
                {
                    label: 'Weekly Received',
                    data: [15, 22, 18, 30, 25, 28, 35, 42],
                    borderColor: '#154284',
                    backgroundColor: 'rgba(21, 66, 132, 0.1)',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Weekly Repaired',
                    data: [10, 18, 20, 24, 22, 30, 32, 38],
                    borderColor: '#c62828',
                    backgroundColor: 'rgba(198, 40, 40, 0.1)',
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function renderRepairAnalysisChart() {
    const ctxRepair = document.getElementById('repairAnalysisChart').getContext('2d');
    if (repairChart) repairChart.destroy();
    
    repairChart = new Chart(ctxRepair, {
        type: 'bar',
        data: {
            labels: ['Bearing Failure', 'Winding Failure', 'Rotor Failure', 'Insulation Failure', 'Overheating', 'Vibration Damage'],
            datasets: [{
                label: 'Failure Frequency in Database',
                data: [42, 28, 16, 14, 8, 5],
                backgroundColor: ['#154284', '#3b82f6', '#10b981', '#f59e0b', '#dc3545', '#6c757d'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// ---------------------------------------------------------------- //
// AI INSPECTION CHECKLIST ACTIONS
// ---------------------------------------------------------------- //

function resetChecklistForMotor() {
    const serial = document.getElementById('inspect-motor-select').value;
    if (!serial) {
        document.getElementById('no-inspection-selected').style.display = 'block';
        document.getElementById('live-ai-results').style.display = 'none';
        return;
    }
    
    // Clear all checklists
    const inputs = document.querySelectorAll('.checklist-column input[type="checkbox"]');
    inputs.forEach(input => input.checked = false);
    
    document.getElementById('inspect-reason').value = 'None';
    document.getElementById('inspect-remarks').value = '';
    
    // Hide results panel
    document.getElementById('no-inspection-selected').style.display = 'block';
    document.getElementById('live-ai-results').style.display = 'none';
}

async function runChecklistAI() {
    const serial = document.getElementById('inspect-motor-select').value;
    if (!serial) {
        alert("Please select a motor serial first.");
        return;
    }
    
    // Collect checklist selections
    const checklist = [];
    const codes = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5'];
    
    codes.forEach(code => {
        if (document.getElementById(`chk-${code}`).checked) {
            checklist.push(code);
        }
    });

    try {
        const response = await fetch('/api/predict/custom', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ serial_no: serial, checklist: checklist })
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }

        renderAIInspectionResults(data);
    } catch (err) {
        console.error("AI live predictor failed:", err);
    }
}

function renderAIInspectionResults(data) {
    const pred = data.prediction;
    
    // Hide instructions, show results
    document.getElementById('no-inspection-selected').style.display = 'none';
    document.getElementById('live-ai-results').style.display = 'block';
    
    // Update health circular gauge
    document.getElementById('inspect-health-val').textContent = pred.health_score.toFixed(1);
    
    const circle = document.getElementById('inspect-health-circle');
    const pill = document.getElementById('inspect-risk-pill');
    
    // Color mapping
    let color = '#28a745'; 
    if (pred.risk_category === 'MEDIUM') color = '#ffc107'; 
    else if (pred.risk_category === 'HIGH') color = '#fd7e14'; 
    else if (pred.risk_category === 'CRITICAL') color = '#dc3545';
    
    // Dynamic circular progress conic background
    const scoreVal = pred.health_score;
    circle.style.background = `radial-gradient(closest-side, white 79%, transparent 80% 100%), conic-gradient(${color} ${scoreVal}%, #dee2e6 ${scoreVal}% 100%)`;
    
    // Update risk pill
    pill.textContent = `${pred.risk_category} RISK`;
    pill.style.backgroundColor = color;
    
    // Update recommendations
    document.getElementById('inspect-recommendation').textContent = pred.recommended_action;
    document.getElementById('inspect-target-date').textContent = pred.suggested_inspection_date;
    
    // Update component bars
    const probsContainer = document.getElementById('inspect-probabilities-bars');
    probsContainer.innerHTML = '';
    
    function getAIExplanation(compName, probPct) {
        if (compName === 'Bearing Failure') {
            if (probPct > 50) return "High bearing failure risk: Seizure or clearance damage expected. Check bearing noise and greasing immediately.";
            if (probPct > 15) return "Moderate bearing wear: Monitor temperature rise and vibration levels during continuous load.";
            return "Low bearing wear: Parameters are within nominal operating limits.";
        }
        if (compName === 'Rotor Failure') {
            if (probPct > 50) return "High rotor risk: Suspected broken rotor bars or eccentricity. Perform dynamic balancing check.";
            if (probPct > 15) return "Moderate rotor wear: Minor air gap variance. Keep under watch for abnormal hum.";
            return "Rotor nominal: No significant electrical or mechanical balance offset.";
        }
        if (compName === 'Winding Failure') {
            if (probPct > 50) return "Critical winding risk: High phase imbalance or short-circuit suspected. Run insulation tests.";
            if (probPct > 15) return "Moderate winding stress: Possible localized overheating. Schedule regular winding check.";
            return "Winding nominal: Phase resistances and insulation parameters are balanced.";
        }
        if (compName === 'Insulation Failure') {
            if (probPct > 50) return "Severe insulation degradation: Moisture ingress or age-induced breakdown. Run megger test before operation.";
            if (probPct > 15) return "Moderate insulation drop: Moisture contamination likely. Baking or cleaning recommended.";
            return "Insulation nominal: Resistance values are well within standard limits.";
        }
        return "Component status nominal.";
    }

    Object.keys(pred.failure_probabilities).forEach(comp => {
        const val = parseFloat((pred.failure_probabilities[comp] * 100).toFixed(1));
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.style.marginBottom = '14px';
        row.innerHTML = `
            <div class="prob-meta">
                <span class="bold">${comp}</span>
                <span>${val}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${val}%; background-color: ${color};"></div>
            </div>
            <div class="prob-desc" style="font-size: 0.78rem; color: #555; margin-top: 4px; line-height: 1.3; font-style: italic;">
                ${getAIExplanation(comp, val)}
            </div>
        `;
        probsContainer.appendChild(row);
    });
}

// ---------------------------------------------------------------- //
// DETAILS MODAL RENDERING
// ---------------------------------------------------------------- //

async function openDiagnosticsModal(serialNo) {
    const modal = document.getElementById('diagnostics-modal');
    modal.classList.add('active');
    
    // Clear old details
    document.getElementById('modal-serial').textContent = serialNo;
    document.getElementById('modal-eq-name').textContent = 'Loading...';
    document.getElementById('modal-make').textContent = 'Loading...';
    document.getElementById('modal-model').textContent = 'Loading...';
    document.getElementById('modal-kw').textContent = 'Loading...';
    document.getElementById('modal-rpm-v').textContent = 'Loading...';
    document.getElementById('modal-year').textContent = 'Loading...';
    document.getElementById('modal-loc').textContent = 'Loading...';
    
    try {
        const response = await fetch(`/api/predict/${serialNo}`);
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            closeModal();
            return;
        }
        
        renderModalData(data);
    } catch (err) {
        console.error("Failed to load details modal:", err);
    }
}

function renderModalData(data) {
    const meta = data.metadata;
    const pred = data.prediction;
    
    // Details
    document.getElementById('modal-eq-name').textContent = meta.equipment_type;
    document.getElementById('modal-make').textContent = meta.manufacturer;
    document.getElementById('modal-model').textContent = meta.model || 'N/A';
    document.getElementById('modal-kw').textContent = `${meta.kw_rating} KW`;
    document.getElementById('modal-rpm-v').textContent = `${meta.rpm} RPM / ${pred.voltage || 415} V`;
    document.getElementById('modal-year').textContent = meta.age_years ? (2026 - meta.age_years) : 'N/A';
    document.getElementById('modal-loc').textContent = meta.installed_location;
    
    // Circular Progress
    document.getElementById('modal-health-val').textContent = pred.health_score.toFixed(1);
    
    const circle = document.getElementById('modal-health-circle');
    const pill = document.getElementById('modal-risk-pill');
    const recCard = document.getElementById('modal-rec-card');
    
    let color = '#28a745';
    if (pred.risk_category === 'MEDIUM') color = '#ffc107';
    else if (pred.risk_category === 'HIGH') color = '#fd7e14';
    else if (pred.risk_category === 'CRITICAL') color = '#dc3545';
    
    const scoreVal = pred.health_score;
    circle.style.background = `radial-gradient(closest-side, white 79%, transparent 80% 100%), conic-gradient(${color} ${scoreVal}%, #dee2e6 ${scoreVal}% 100%)`;
    
    pill.textContent = `${pred.risk_category} RISK`;
    pill.style.backgroundColor = color;
    
    recCard.style.borderLeftColor = color;
    document.getElementById('modal-recommendation').textContent = pred.recommended_action;
    document.getElementById('modal-target-date').textContent = pred.suggested_inspection_date;
    
    // Modal chart rendering
    renderModalChart(pred.failure_probabilities, color);
}

function renderModalChart(probs, color) {
    const ctx = document.getElementById('modalProbabilitiesChart').getContext('2d');
    if (modalChart) modalChart.destroy();
    
    const labels = Object.keys(probs);
    const data = Object.values(probs).map(v => (v * 100).toFixed(1));
    
    modalChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Failure Probability (%)',
                data: data,
                backgroundColor: color,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 100 }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function closeModal() {
    const modal = document.getElementById('diagnostics-modal');
    modal.classList.remove('active');
}

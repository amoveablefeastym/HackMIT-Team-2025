// Configuration
const USE_MOCK_API = true;  // Set to false to use real backend API
const API_URL = 'http://localhost:5000/api';

// Sample transcript for demo
const SAMPLE_TRANSCRIPT = `Example Telehealth Appointment Transcript (≈30 minutes)
Clinician (Dr. Lee): Hi Alex, it's good to see you today. Can you hear and see me okay?
Patient (Alex): Yes, loud and clear.
Dr. Lee: Great. Before we dive in, I'll remind you this is a secure telehealth session, and everything we discuss today will be part of your medical record. How are you doing today?
Alex: Honestly, a bit tired. It's been a rough couple of weeks.
Dr. Lee: I'm sorry to hear that. Let's start by checking in on what's been hardest for you lately.
Alex: My sleep has been really bad. I'm waking up multiple times a night, and during the day I feel sluggish.
Dr. Lee: Thank you for sharing that. We'll talk more about your sleep. First, I'd like to review some background. When we last met a month ago, you mentioned stress at work and occasional headaches. Are those still going on?
Alex: The headaches are better, but the stress is worse.
Dr. Lee: Okay, noted. Let's go step by step. Can you tell me about your current sleep routine?
Alex: I usually get into bed around 11, but I scroll on my phone for a while. I probably fall asleep by midnight. Then I wake up around 3 or 4 a.m. and can't get back to sleep for an hour or two.
Dr. Lee: How many total hours do you think you're getting?
Alex: Maybe 5 or 6 at most.
Dr. Lee: That's definitely on the low end for restorative sleep. Have you noticed if caffeine, alcohol, or stress makes it worse?
Alex: I drink coffee in the morning, sometimes an energy drink in the afternoon. I don't drink alcohol often. Stress is probably the main thing.`;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const transcriptText = document.getElementById('transcriptText');
const generateBtn = document.getElementById('generateBtn');
const clearBtn = document.getElementById('clearBtn');
const loadSampleBtn = document.getElementById('loadSampleBtn');
const resultCard = document.getElementById('resultCard');
const resultContent = document.getElementById('resultContent');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const tabBtns = document.querySelectorAll('.tab-btn');

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.dataset.tab;
        
        // Update active tab button
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(targetTab).classList.add('active');
    });
});

// File upload handling
uploadArea.addEventListener('click', () => fileInput.click());

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
    
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'text/plain') {
        handleFile(file);
    } else {
        alert('Please upload a .txt file');
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        transcriptText.value = e.target.result;
    };
    reader.readAsText(file);
}

// Load sample transcript
loadSampleBtn.addEventListener('click', () => {
    // Switch to upload tab
    tabBtns[0].click();
    transcriptText.value = SAMPLE_TRANSCRIPT;
});

// Clear button
clearBtn.addEventListener('click', () => {
    transcriptText.value = '';
    fileInput.value = '';
    resultCard.style.display = 'none';
});

// Generate SOAP note
generateBtn.addEventListener('click', async () => {
    const text = transcriptText.value.trim();
    
    if (!text) {
        alert('Please enter or upload a transcript first.');
        return;
    }
    
    // Show loading state
    const btnText = generateBtn.querySelector('.btn-text');
    const loader = generateBtn.querySelector('.loader');
    btnText.textContent = 'Generating...';
    loader.style.display = 'block';
    generateBtn.disabled = true;
    
    try {
        // Simulate API call (in production, this would call your backend)
        const result = await generateSOAPNote(text);
        
        // Display result
        resultContent.textContent = result;
        resultCard.style.display = 'block';
        
        // Scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (error) {
        alert('Error generating SOAP note: ' + error.message);
    } finally {
        // Reset button state
        btnText.textContent = 'Generate SOAP Note';
        loader.style.display = 'none';
        generateBtn.disabled = false;
    }
});

// Generate SOAP note - calls backend API or uses mock data
async function generateSOAPNote(transcript) {
    if (USE_MOCK_API) {
        // Mock mode for demo without backend
        await new Promise(resolve => setTimeout(resolve, 2000));
        return getMockSOAPNote();
    } else {
        // Real API mode
        const response = await fetch(`${API_URL}/generate-soap`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transcript })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to generate SOAP note');
        }
        
        return data.soap_note;
    }
}

// Mock SOAP note for demo
function getMockSOAPNote() {
    return `**Subjective:**
Patient reports difficulty sleeping, waking multiple times at night, and daytime sluggishness over the past few weeks. Patient attributes sleep disturbance to increased work stress. Headaches have improved since last visit. Patient denies alcohol use but consumes coffee and energy drinks. Sleep routine involves phone use before bed, falling asleep around midnight, and waking at 3-4 AM with difficulty returning to sleep. Estimates 5-6 hours of sleep per night.

**Objective:**
Virtual telehealth appointment conducted. Patient appears fatigued but alert and oriented. No physical examination performed.

**Assessment:**
1. Insomnia secondary to stress and poor sleep hygiene
2. Possible mild anxiety related to work stress
3. Caffeine consumption contributing to sleep disturbance

**Plan:**
1. Sleep hygiene education:
   - Limit screen time 30-60 minutes before bed
   - Establish consistent sleep/wake schedule
   - Reduce caffeine intake, especially after noon
   - Create relaxing bedtime routine
2. Stress management techniques:
   - Consider mindfulness or meditation apps
   - Recommend therapy consultation if stress persists
3. Follow-up in 4 weeks to assess sleep improvement
4. Consider sleep study if symptoms persist despite interventions
5. Patient education materials sent via patient portal`;
}

// Copy to clipboard
copyBtn.addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(resultContent.textContent);
        
        // Visual feedback
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        copyBtn.style.color = 'var(--success-color)';
        
        setTimeout(() => {
            copyBtn.innerHTML = originalHTML;
            copyBtn.style.color = '';
        }, 2000);
    } catch (error) {
        alert('Failed to copy to clipboard');
    }
});

// Download as text file
downloadBtn.addEventListener('click', () => {
    const text = resultContent.textContent;
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soap-note-${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

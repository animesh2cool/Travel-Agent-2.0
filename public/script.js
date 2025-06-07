document.getElementById('planForm').addEventListener('submit', async (e) => {
    e.preventDefault();
   
    const formData = {
        from_city: document.getElementById('fromCity').value,
        destination: document.getElementById('destination').value,
        start_date: document.getElementById('startDate').value,
        end_date: document.getElementById('endDate').value,
        interests: document.getElementById('interests').value
    };

    showLoading();
   
    try {
        const response = await fetch('/plan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
       
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError('Failed to generate plan. Please try again.');
    } finally {
        hideLoading();
    }
});

function displayResults(data) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = `
        <div class="result-section highlight">
            <h2 class="text-2xl font-bold mb-4 text-blue-600">🗺️ ${data.language.toUpperCase()} Travel Plan</h2>
            <div class="prose max-w-none">${marked.parse(data.itinerary.content)}</div>
        </div>
        <div class="result-section">
            <h2 class="text-2xl font-bold mb-4 text-purple-600">📚 Travel Guide</h2>
            <div class="prose max-w-none">${marked.parse(data.guide.content)}</div>
        </div>
        <div class="result-section">
            <h2 class="text-2xl font-bold mb-4 text-green-600">⚙️ Travel Logistics</h2>
            <div class="prose max-w-none">${marked.parse(data.logistics.content)}</div>
        </div>
    `;
    resultsContainer.classList.remove('hidden');
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4';
    errorDiv.textContent = message;
    document.getElementById('results').prepend(errorDiv);
}
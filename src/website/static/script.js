async function getPrediction() {
    const resultsDiv = document.getElementById('results');
    resultsDiv.textContent = 'Loading...';

    const scalarFeaturesInput = document.getElementById('scalar_features').value;
    let scalar_features_list = [];
    if (scalarFeaturesInput.trim() !== '') {
        scalar_features_list = scalarFeaturesInput.split(',').map(Number).filter(n => !isNaN(n));
    }

    // --- Field Data Handling (Example for flattened data) ---
    const fieldDataInput = document.getElementById('field_data').value;
    let field_data_flat_list = [];
    if (fieldDataInput.trim() !== '') {
        field_data_flat_list = fieldDataInput.split(',').map(Number).filter(n => !isNaN(n));
    }
    // --- End Field Data ---


    const payload = {};
    if (scalar_features_list.length > 0) {
        payload.scalar_features = scalar_features_list;
    }
    if (field_data_flat_list.length > 0) {
        payload.field_data_flat = field_data_flat_list;
    }


    try {
        // Assuming your FastAPI backend is running on port 8000
        const response = await fetch('http://localhost:8000/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(`HTTP error! status: ${response.status}, Message: ${errorData.detail || JSON.stringify(errorData)}`);
        }

        const data = await response.json();

        if (data.error) {
            resultsDiv.textContent = `Error: ${data.error}`;
        } else {
            resultsDiv.textContent = JSON.stringify(data, null, 2);
        }

    } catch (error) {
        console.error('Error fetching prediction:', error);
        resultsDiv.textContent = `Error: ${error.message}`;
    }
}

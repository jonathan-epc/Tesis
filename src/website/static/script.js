// src/website/static/script.js

// --- Helper function to generate a flat bed for easy testing ---
function generateFlatBed() {
    const fieldDataInput = document.getElementById('field_data_flat');
    // Create an array of 4411 zeros and join them into a string
    const flatBedArray = new Array(4411).fill(0.0);
    fieldDataInput.value = flatBedArray.join(',');
}

// --- Main prediction function ---
async function sendPredictionRequest() {
    const scalarInput = document.getElementById('scalar_features').value;
    const fieldInput = document.getElementById('field_data_flat').value;
    const jsonOutput = document.getElementById('json-output');
    const plotContainer = document.getElementById('plot-container');
    const loadingIndicator = document.getElementById('loading');

    // --- Reset UI ---
    jsonOutput.textContent = '';
    plotContainer.innerHTML = ''; // Clear previous plots
    loadingIndicator.style.display = 'block'; // Show loading indicator

    try {
        // --- Prepare data for API ---
        const scalar_features = scalarInput.split(',').map(Number);
        const field_data_flat = fieldInput.split(',').map(Number);

        if (scalar_features.length !== 4) {
            throw new Error("Please provide exactly 4 scalar features.");
        }
        if (field_data_flat.length !== 4411) {
            throw new Error(`Field data must have 4411 elements, but found ${field_data_flat.length}. Use 'Generate Flat Bed' for a quick test.`);
        }

        const response = await fetch('http://127.0.0.1:8000/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                scalar_features: scalar_features,
                field_data_flat: field_data_flat,
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            // Handle HTTP errors (like 400 or 500)
            throw new Error(data.detail || 'An unknown error occurred.');
        }

        // Display the raw JSON response for debugging
        const summaryData = { ...data }; // Create a copy of the response data
        delete summaryData.field_predictions_flat; // Remove the large data key from the copy
        
        // Display the cleaned-up summary JSON
        jsonOutput.textContent = JSON.stringify(summaryData, null, 2);
        
        // --- Plotting Logic ---
        if (data.field_predictions_flat) {
            const gridHeight = 11;
            const gridWidth = 401;

            for (const fieldName in data.field_predictions_flat) {
                const flatData = data.field_predictions_flat[fieldName];
                
                // Un-flatten the 1D array into a 2D array for the heatmap
                const zData = unflatten(flatData, gridHeight, gridWidth);

                // Create a new div for this plot
                const plotDiv = document.createElement('div');
                plotDiv.id = `plot-${fieldName}`;
                plotContainer.appendChild(plotDiv);

                const layout = {
                    title: `Predicted Field: ${fieldName}`,
                    xaxis: { title: 'Channel Length (Grid Points)' },
                    yaxis: { title: 'Channel Width (Grid Points)' }
                };
                
                const plotData = [{
                    z: zData,
                    type: 'heatmap',
                    colorscale: 'Viridis' // A nice colorscale for H and U
                }];

                Plotly.newPlot(plotDiv.id, plotData, layout);
            }
        }

    } catch (error) {
        jsonOutput.textContent = `Error: ${error.message}`;
        console.error('Prediction failed:', error);
    } finally {
        loadingIndicator.style.display = 'none'; // Hide loading indicator
    }
}

// --- Helper function to reshape a flat array into a 2D array ---
function unflatten(arr, rows, cols) {
    if (arr.length !== rows * cols) {
        console.error("Cannot unflatten array, dimensions do not match.");
        return [];
    }
    const newArr = [];
    while (arr.length) {
        newArr.push(arr.splice(0, cols));
    }
    return newArr;
}

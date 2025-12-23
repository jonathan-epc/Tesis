// src/website/static/script.js

let availableModels = {};

document.addEventListener('DOMContentLoaded', async () => {
    await fetchAvailableModels();
    updateUIForSelectedModel();
});

async function fetchAvailableModels() {
    try {
        const response = await fetch('/models');
        availableModels = await response.json();
        const selector = document.getElementById('model-selector');
        selector.innerHTML = '';
        for (const key in availableModels) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = `${key.toUpperCase()}: ${availableModels[key].description}`;
            selector.appendChild(option);
        }
    } catch (error) {
        console.error("Error fetching models:", error);
    }
}

function updateUIForSelectedModel() {
    const selectedKey = document.getElementById('model-selector').value;
    if (!selectedKey || !availableModels[selectedKey]) return;

    const config = availableModels[selectedKey];

    // Show/hide scalar section
    const scalarWrapper = document.getElementById('scalar-inputs-wrapper');
    if (config.input_scalars.length > 0) {
        scalarWrapper.classList.remove('hidden');
        createScalarInputs(config.input_scalars);
    } else {
        scalarWrapper.classList.add('hidden');
    }

    // Show/hide field section
    const fieldWrapper = document.getElementById('field-inputs-wrapper');
    if (config.input_fields.length > 0) {
        fieldWrapper.classList.remove('hidden');
        document.getElementById('field-label').textContent = `Field Data (${config.input_fields.join(', ') || 'None'}):`;
        const numFields = config.input_fields.length;
        document.getElementById('field_data_flat').placeholder = `Paste ${4411 * numFields} comma-separated values for ${numFields} field(s).`;
    } else {
        fieldWrapper.classList.add('hidden');
    }
}

function createScalarInputs(scalars) {
    const container = document.getElementById('scalar-inputs-container');
    container.innerHTML = ''; // Clear old sliders

    scalars.forEach(scalar => {
        const row = document.createElement('div');
        row.className = 'scalar-input-row';
        row.innerHTML = `
            <label for="slider-${scalar.name}">${scalar.name}:</label>
            <input type="range" id="slider-${scalar.name}" name="${scalar.name}" min="${scalar.min}" max="${scalar.max}" step="${scalar.step}" value="${scalar.default}">
            <input type="number" id="number-${scalar.name}" min="${scalar.min}" max="${scalar.max}" step="${scalar.step}" value="${scalar.default}">
        `;
        container.appendChild(row);

        const slider = row.querySelector(`#slider-${scalar.name}`);
        const numberInput = row.querySelector(`#number-${scalar.name}`);

        slider.oninput = () => numberInput.value = slider.value;
        numberInput.oninput = () => slider.value = numberInput.value;
    });
}

function generatePlaceholderFieldData() {
    const selectedKey = document.getElementById('model-selector').value;
    const config = availableModels[selectedKey];
    const numFields = config.input_fields.length;
    if (numFields === 0) {
        document.getElementById('field_data_flat').value = '';
        return;
    }
    const totalPoints = 4411 * numFields;
    document.getElementById('field_data_flat').value = new Array(totalPoints).fill(0.0).join(',');
}

async function sendPredictionRequest() {
    const selectedKey = document.getElementById('model-selector').value;
    const config = availableModels[selectedKey];

    // We REMOVED the reference to jsonOutput here
    const plotContainer = document.getElementById('plot-container');
    const loadingIndicator = document.getElementById('loading');

    // We REMOVED jsonOutput.textContent = '' here
    plotContainer.innerHTML = '';
    loadingIndicator.classList.remove('hidden');

    try {
        // Gather scalar inputs dynamically
        const scalar_features = config.input_scalars.map(scalar => {
            return parseFloat(document.getElementById(`slider-${scalar.name}`).value);
        });

        // Gather field inputs
        const fieldInput = document.getElementById('field_data_flat').value;
        const field_data_flat = config.input_fields.length > 0 ? fieldInput.split(',').map(Number) : [];

        const response = await fetch(`/predict/${selectedKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scalar_features, field_data_flat }),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'API request failed.');

        // --- THIS BLOCK IS THE MAIN CHANGE ---
        // Log the summary to the developer console for debugging, not the webpage
        const summaryData = { ...data };
        delete summaryData.field_predictions_flat;
        console.log("API Response Summary:", summaryData);

        plotResults(data, config);

    } catch (error) {
        // --- THIS IS THE OTHER CHANGE ---
        // Use an alert to show errors, since our display box is gone
        alert(`Error: ${error.message}`);
        console.error('Prediction failed:', error);
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

function plotResults(data, config) {
    const plotContainer = document.getElementById('plot-container');

    if (data.scalar_predictions && data.scalar_predictions.length > 0) {
        // Create a container for scalars if it doesn't exist
        let scalarDiv = document.getElementById('plot-scalars');
        if (!scalarDiv) {
            scalarDiv = document.createElement('div');
            scalarDiv.id = 'plot-scalars';
            plotContainer.appendChild(scalarDiv);
        }
        Plotly.newPlot(scalarDiv.id, [{
            x: config.output_scalars,
            y: data.scalar_predictions,
            type: 'bar'
        }], { title: 'Predicted Scalar Values' });
    }

    if (data.field_predictions_flat) {
        const gridHeight = 11;
        const gridWidth = 401;

        for (const fieldName in data.field_predictions_flat) {
            const zData = unflatten(data.field_predictions_flat[fieldName], gridHeight, gridWidth);

            const plotDiv = document.createElement('div');
            plotContainer.appendChild(plotDiv);

            // --- THIS IS THE FIX FOR ASPECT RATIO ---
            const layout = {
                title: `Predicted Field: ${fieldName}`,
                width: plotContainer.clientWidth, // Use the container's width
                height: 250, // A fixed, shorter height
                xaxis: { title: 'Channel Length (Grid Points)' },
                yaxis: {
                    title: 'Width',
                    autorange: 'reversed',
                    // This forces the y-axis scaling to respect the x-axis
                    scaleanchor: 'x',
                    scaleratio: 25 / (gridWidth / gridHeight) // Adjust this ratio to tune the appearance
                }
            };

            Plotly.newPlot(plotDiv, [{ z: zData, type: 'heatmap', colorscale: 'Viridis' }], layout);
        }
    }
}

function unflatten(arr, rows, cols) {
    const newArr = [];
    for (let r = 0; r < rows; r++) {
        newArr.push(arr.slice(r * cols, (r + 1) * cols));
    }
    return newArr;
}

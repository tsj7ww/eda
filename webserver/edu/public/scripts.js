// Initialize when everything is ready
function init() {
    if (typeof Plotly === 'undefined') {
        console.error('Plotly not loaded');
        return;
    }
    if (typeof Papa === 'undefined') {
        console.error('Papaparse not loaded');
        return;
    }
    loadData();
}

// Data loading and parsing function
async function loadData() {
    try {
        const response = await fetch('/data/processed/edu.csv');
        const csvText = await response.text();
        
        Papa.parse(csvText, {
            header: true,           // Use first row as headers
            dynamicTyping: true,    // Automatically convert numbers
            skipEmptyLines: true,   // Skip empty lines
            complete: function(results) {
                // Check for parsing errors
                if (results.errors.length > 0) {
                    console.warn('CSV parsing had errors:', results.errors);
                }

                const data = results.data;

                // Log the first few rows to verify parsing
                console.log('First few rows:', data.slice(0, 3));
                console.log('Total rows:', data.length);
                console.log('Fields:', results.meta.fields);

                // Verify data types
                if (data.length > 0) {
                    console.log('Sample data types:', {
                        Year: typeof data[0].Year,
                        Segment: typeof data[0].Segment,
                        Metric: typeof data[0].Metric
                    });
                }

                // Create visualizations with the parsed data
                createPlot(data);
                createTable(data, results.meta.fields);
            },
            error: function(error) {
                console.error('Error parsing CSV:', error);
                document.getElementById('tableContainer').innerHTML = 
                    `<p style="color: red;">Error parsing CSV: ${error.message}</p>`;
            }
        });
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('tableContainer').innerHTML = 
            `<p style="color: red;">Error loading data: ${error.message}</p>`;
    }
}
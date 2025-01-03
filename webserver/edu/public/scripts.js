// Data loading and parsing functions
async function loadData() {
    try {
        const response = await fetch('/data/processed/edu.csv');
        const csvText = await response.text();
        
        // Parse CSV into array of objects
        const rows = csvText.split('\n');
        const headers = rows[0].split(',');
        const data = rows.slice(1).map(row => {
            const values = row.split(',');
            return headers.reduce((obj, header, index) => {
                obj[header.trim()] = values[index];
                return obj;
            }, {});
        });

        // Create visualizations
        createPlot(data);
        createTable(data, headers);
        
        console.log('Data loaded successfully:', data);
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('tableContainer').innerHTML = 
            `<p style="color: red;">Error loading data: ${error.message}</p>`;
    }
}

function createTable(data, headers) {
    const table = document.createElement('table');
    
    // Create header row
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header.trim();
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create data rows
    const tbody = document.createElement('tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = row[header.trim()];
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    const container = document.getElementById('tableContainer');
    container.innerHTML = '';
    container.appendChild(table);
}
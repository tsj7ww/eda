function createTable(data, headers) {
    const container = document.getElementById('tableContainer');
    const rowsPerPage = 10;
    let currentPage = 1;
    
    container.innerHTML = `
        <div class="data-table-container">
            <div class="data-table-controls">
                <input type="text" class="data-table-search" placeholder="Search table...">
                <div class="data-table-info">
                    Showing <span class="data-table-count">0</span> entries
                </div>
            </div>
            <table class="data-table">
                <thead>
                    <tr>
                        ${headers.map(header => 
                            `<th data-column="${header.trim()}">${header.trim()}</th>`
                        ).join('')}
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
            <div class="data-table-pagination">
                <button class="pagination-prev">Previous</button>
                <span class="pagination-info">Page <span class="current-page">1</span></span>
                <button class="pagination-next">Next</button>
            </div>
        </div>
    `;

    // Add event listeners for sorting and filtering
    setupTableInteractivity(container, data, headers, rowsPerPage);
}

function setupTableInteractivity(container, data, headers, rowsPerPage) {
    const searchInput = container.querySelector('.data-table-search');
    const tableHeaders = container.querySelectorAll('th');
    const prevButton = container.querySelector('.pagination-prev');
    const nextButton = container.querySelector('.pagination-next');
    let sortColumn = '';
    let sortDirection = 'asc';
    let currentPage = 1;
    
    // Sorting functionality
    tableHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const column = header.dataset.column;
            
            // Remove sort classes from all headers
            tableHeaders.forEach(h => {
                h.classList.remove('sort-asc', 'sort-desc');
            });
            
            // Toggle sort direction
            if (sortColumn === column) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortColumn = column;
                sortDirection = 'asc';
            }
            
            // Add sort class to current header
            header.classList.add(`sort-${sortDirection}`);
            
            // Reset to first page when sorting
            currentPage = 1;
            
            // Update table
            updateTable(data, headers, container, searchInput.value, sortColumn, sortDirection, currentPage, rowsPerPage);
        });
    });
    
    // Search functionality
    searchInput.addEventListener('input', (e) => {
        currentPage = 1; // Reset to first page when searching
        updateTable(data, headers, container, e.target.value, sortColumn, sortDirection, currentPage, rowsPerPage);
    });

    // Pagination functionality
    prevButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            updateTable(data, headers, container, searchInput.value, sortColumn, sortDirection, currentPage, rowsPerPage);
        }
    });

    nextButton.addEventListener('click', () => {
        const filteredData = filterData(data, searchInput.value);
        const maxPage = Math.ceil(filteredData.length / rowsPerPage);
        if (currentPage < maxPage) {
            currentPage++;
            updateTable(data, headers, container, searchInput.value, sortColumn, sortDirection, currentPage, rowsPerPage);
        }
    });

    // Initial table update
    updateTable(data, headers, container, '', sortColumn, sortDirection, currentPage, rowsPerPage);
}

function filterData(data, searchTerm) {
    if (!searchTerm) return data;
    
    const term = searchTerm.toLowerCase();
    return data.filter(row => 
        Object.values(row).some(value => 
            String(value).toLowerCase().includes(term)
        )
    );
}

function updateTable(data, headers, container, searchTerm, sortColumn, sortDirection, currentPage, rowsPerPage) {
    // Filter data
    let filteredData = filterData(data, searchTerm);
    
    // Sort data
    if (sortColumn) {
        filteredData.sort((a, b) => {
            const aVal = String(a[sortColumn]).toLowerCase();
            const bVal = String(b[sortColumn]).toLowerCase();
            
            // Try to compare as numbers if possible
            const aNum = parseFloat(aVal);
            const bNum = parseFloat(bVal);
            
            if (!isNaN(aNum) && !isNaN(bNum)) {
                return sortDirection === 'asc' ? aNum - bNum : bNum - aNum;
            }
            
            // Fall back to string comparison
            return sortDirection === 'asc' 
                ? aVal.localeCompare(bVal) 
                : bVal.localeCompare(aVal);
        });
    }

    // Calculate pagination
    const totalPages = Math.ceil(filteredData.length / rowsPerPage);
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    const paginatedData = filteredData.slice(startIndex, endIndex);
    
    // Update table body
    const tbody = container.querySelector('tbody');
    tbody.innerHTML = paginatedData.map(row => `
        <tr>
            ${headers.map(header => 
                `<td>${row[header.trim()] || ''}</td>`
            ).join('')}
        </tr>
    `).join('');
    
    // Update count and pagination info
    container.querySelector('.data-table-count').textContent = 
        `${startIndex + 1}-${Math.min(endIndex, filteredData.length)} of ${filteredData.length}`;
    container.querySelector('.current-page').textContent = `${currentPage} of ${totalPages}`;
    
    // Update pagination buttons
    const prevButton = container.querySelector('.pagination-prev');
    const nextButton = container.querySelector('.pagination-next');
    prevButton.disabled = currentPage === 1;
    nextButton.disabled = currentPage === totalPages;
}
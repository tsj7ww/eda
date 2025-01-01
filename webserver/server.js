const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());

// Root route - redirect to project1 by default
app.get('/', (req, res) => {
    res.redirect('/init');
});

// Serve static files for project1
app.use('/init', express.static(path.join(__dirname, 'init')));

// Serve static files for project2
app.use('/project2', express.static(path.join(__dirname, 'project2')));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
});

// Start server
app.listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://localhost:${port}`);
});
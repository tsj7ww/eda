const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());

// Root route - redirect to init by default
app.get('/', (req, res) => {
    res.redirect('/init');
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
});

// Create endpoint to access data files
app.get('/data/processed/:filename', (req, res) => {
    const filename = req.params.filename;
    res.sendFile(`/data/processed/${filename}`);
});

// Static files
app.use('/edu/public', express.static(path.join(__dirname, 'edu/public')));
app.use('/edu/charts', express.static(path.join(__dirname, 'edu/charts')));
app.use('/js', express.static(path.join(__dirname, 'node_modules/plotly.js-dist')));

// Project routes
app.use('/init', express.static(path.join(__dirname, 'init')));
app.use('/edu', express.static(path.join(__dirname, 'edu')));

// Start server
app.listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://localhost:${port}`);
});
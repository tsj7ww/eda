const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Basic middleware
app.use(cors());
app.use(helmet({
    contentSecurityPolicy: false  // For external CDNs
}));
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
});

// Root route - redirect to init by default
app.get('/', (req, res) => {
    res.redirect('/init');
});

// Create a libraries object mapping routes to node_modules files
const libraries = {
    '/js/plotly.js': 'plotly.js-dist/plotly.js',
    '/js/papaparse.js': 'papaparse/papaparse.min.js',
    '/js/d3.js': 'd3/dist/d3.min.js',
    '/js/echarts.js': 'echarts/dist/echarts.min.js'
};
// Serve library files with proper MIME types
Object.entries(libraries).forEach(([route, modulePath]) => {
    app.use(route, (req, res) => {
        res.set('Content-Type', 'application/javascript');
        res.sendFile(path.join(__dirname, 'node_modules', modulePath));
    });
});
// function setupModuleFile(moduleDirectory) {
//     app.use('/js', express.static(path.join(__dirname, `node_modules/${moduleDirectory}`), {
//         setHeaders: (res, path) => {
//             if (path.endsWith('.js')) {
//                 res.setHeader('Content-Type', 'application/javascript');
//             }
//         }
//     }))
// };
// setupModuleFile('plotly.js-dist');
// setupModuleFile('papaparse');

// Function to set up static serving for a visualization directory
function setupVisualizationRoutes(directoryName) {
    // Serve the charts directory
    app.use(`/${directoryName}/charts`, express.static(path.join(__dirname, directoryName, 'charts'), {
        setHeaders: (res, path) => {
            if (path.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            }
        }
    }));

    // Serve the public directory
    app.use(`/${directoryName}/public`, express.static(path.join(__dirname, directoryName, 'public'), {
        setHeaders: (res, path) => {
            if (path.endsWith('.css')) {
                res.setHeader('Content-Type', 'text/css');
            }
            if (path.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            }
        }
    }));

    // Serve the main directory for static files
    app.use(`/${directoryName}`, express.static(path.join(__dirname, directoryName)));
}

// Set up routes for each visualization directory
setupVisualizationRoutes('edu');
setupVisualizationRoutes('init');

// Create endpoint to access data files
app.get('/data/processed/:filename', (req, res) => {
    const filename = req.params.filename;
    res.sendFile(`/data/processed/${filename}`);
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Start server
app.listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://localhost:${port}`);
    console.log(`Education visualization available at: http://localhost:${port}/edu`);
    console.log(`Initial visualization available at: http://localhost:${port}/init`);
});
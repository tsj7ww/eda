function createPlot(data) {
    const traces = [];
    
    // Get unique segments
    const segments = [...new Set(data.map(row => row.Segment))];
    
    // Create a trace for each segment
    segments.forEach(segment => {
        const segmentData = data.filter(row => row.Segment === segment);
        
        traces.push({
            type: 'scatter',
            x: segmentData.map(row => row.Year),
            y: segmentData.map(row => row.Metric),
            name: segment,
            mode: 'lines+markers',
            hovertemplate: '%{x}<br>' +
                         segment + ': %{y:.1f}%' +
                         '<extra></extra>'
        });
    });

    const layout = {
        title: {
            text: 'Metrics by Segment Over Time',
            y: 0.95,
            x: 0.5,
            xanchor: 'center',
            yanchor: 'top'
        },
        xaxis: {
            title: 'Year'
        },
        yaxis: {
            title: 'Metric',
            tickformat: '.0f',
            ticksuffix: '%'
        },
        height: 700,
        margin: {
            t: 200
        },
        legend: {
            yanchor: "bottom",
            y: 1.02,
            xanchor: "left",
            x: 0.01,
            orientation: "h",
            font: {
                size: 10
            }
        }
    };

    Plotly.newPlot('myPlot', traces, layout);
}
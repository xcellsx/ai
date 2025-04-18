// src/JournalReport.js
import React, { useMemo, useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
// 1. Import ApexCharts component (ensure installed: npm install react-apexcharts apexcharts)
import Chart from 'react-apexcharts';
// If using Next.js and encountering SSR issues:
// import dynamic from 'next/dynamic';
// const Chart = dynamic(() => import('react-apexcharts'), { ssr: false });

import AnimatedBackgroundContainer from './AnimatedBackgroundContainer';
// Assuming StyledNavButton defined here or imported (ensure definition matches other files)
const StyledNavButton = styled.button`
  background-color: transparent;
  color: #333;
  border: 1px solid black;
  padding: 8px 18px;
  margin: 0 5px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  text-decoration: none;
  transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  font-family: inherit;
  &:hover { background-color: black; color: #FFF; }
`;

// --- Constants ---
const emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];
// Colors used for bar chart and potentially legend/tooltips
const emotionColors = {
  sadness: '#BCD8EC', joy: '#F9E1A8', love: '#FFCBE1',
  anger: '#FFDAB4', fear: '#DCCCEC', surprise: '#D6E5BD',
};

// Mapping emotions to inferred depression levels for heatmap grouping
const emotionToDepressionLevel = {
    sadness: 'High', anger: 'High',
    fear: 'Mid',
    joy: 'Low', love: 'Low', surprise: 'Low',
};
const depressionLevelOrder = ['High', 'Mid', 'Low']; // Order for heatmap rows
// --- End Constants ---


function JournalReport() {
    const [allStoredEntries, setAllStoredEntries] = useState({}); // State for fetched data from Local Storage

    // --- Fetch ALL Data from Local Storage ONCE on Mount ---
    useEffect(() => {
        // console.log("Fetching all entries from Local Storage for Report.");
        try {
            // Assumes data is stored like: { "YYYY-MM-DD": [entry1, entry2], ... }
            const entries = JSON.parse(localStorage.getItem('journalEntries') || '{}');
            setAllStoredEntries(entries);
        } catch (error) {
            console.error("Failed to load entries from Local Storage:", error);
            setAllStoredEntries({}); // Reset on error
        }
    }, []); // Empty dependency array = runs only on mount
    // --- End Fetch Data ---

    // --- Calculate TOTAL Emotion Counts from Stored Data ---
    const emotionCounts = useMemo(() => {
        const counts = {};
        emotions.forEach(emotion => { counts[emotion] = 0; }); // Initialize counts

        // Iterate through each day's array of entries
        Object.values(allStoredEntries).forEach(dayEntriesArray => {
             // Check if the data for the day is actually an array
             if (Array.isArray(dayEntriesArray)) {
                 dayEntriesArray.forEach(entry => {
                     // Check if the entry has a valid emotion and increment its count
                     if (entry && entry.emotion && counts.hasOwnProperty(entry.emotion)) {
                         counts[entry.emotion]++;
                     }
                 });
             }
        });
        // console.log("Calculated TOTAL Emotion Counts:", counts);
        return counts;
    }, [allStoredEntries]); // Recalculate when fetched data changes
    // --- End Calculation ---

    const showHelpMessage = useMemo(() => {
      if (!emotionCounts) return false;
      return (emotionCounts.sadness > 0 || emotionCounts.anger > 0);
  }, [emotionCounts]);

    // --- Process data specifically for ApexCharts Heatmap ---
    const heatmapSeries = useMemo(() => {
         // Defensive check
         if (!emotionCounts || typeof emotionCounts !== 'object') {
             return []; // Return empty series if counts aren't ready
         }
        // Initialize series structure (one object per row/level)
        const series = depressionLevelOrder.map(level => ({
            name: level, data: []
        }));
        // Populate the data arrays with { x: emotionName, y: count }
        Object.entries(emotionCounts).forEach(([emotion, count]) => {
            if (count > 0 && emotions.includes(emotion)) {
                const level = emotionToDepressionLevel[emotion];
                if (level) {
                    const levelIndex = series.findIndex(item => item.name === level);
                    if (levelIndex !== -1) {
                        series[levelIndex].data.push({
                            x: emotion.charAt(0).toUpperCase() + emotion.slice(1), // Capitalize
                            y: count // Count is the value
                        });
                    }
                }
            }
        });
        return series.filter(s => s.data.length > 0); // Filter empty rows
    }, [emotionCounts]);

    // --- Configure ApexCharts Heatmap Options ---
    const heatmapOptions = useMemo(() => ({
        chart: { type: 'heatmap', toolbar: { show: true }, fontFamily: 'inherit', background: 'transparent' },
        plotOptions: { heatmap: { shadeIntensity: 0.7, enableShades: true, radius: 4, useFillColorAsStroke: false, colorScale: { ranges: [ { from: 1, to: 5, name: '1-5', color: '#C8E6C9' }, { from: 6, to: 10, name: '6-10', color: '#81C784' }, { from: 11, to: 20, name: '11-20', color: '#4CAF50' }, { from: 21, to: 50, name: '21-50', color: '#388E3C' }, { from: 51, to: 1000, name: '>50', color: '#1B5E20' } ] } } },
        dataLabels: { enabled: true, style: { fontSize: '12px', colors: ['#333'] } },
        xaxis: { type: 'category', categories: emotions.map(e => e.charAt(0).toUpperCase() + e.slice(1)), title: { text: 'Emotion Category', style: { color: '#555', fontSize: '13px' } }, tickPlacement: 'on', labels: { style: { colors: '#555', fontSize: '12px' } } },
        yaxis: { title: { text: 'Inferred Depression Level', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' } } },
        stroke: { width: 1, colors: ['#fff'] },
        tooltip: { enabled: true, y: { formatter: function (val) { return val + " entries" } }, marker: { show: false } }
    }), []); // Empty deps array is fine for static options

    // --- Process data for Bar Chart ---
    const barChartData = useMemo(() => {
        if (!emotionCounts || typeof emotionCounts !== 'object') {
            return { series: [], categories: [], chartColors: [] };
        }
        const categories = emotions.map(e => e.charAt(0).toUpperCase() + e.slice(1));
        const data = emotions.map(emotion => emotionCounts[emotion] || 0);
        const series = [{ name: 'Total Count', data: data }];
        const chartColors = emotions.map(emotion => emotionColors[emotion] || '#888888');
        return { series, categories, chartColors };
    }, [emotionCounts]);

    // --- Configure ApexCharts Bar Chart Options ---
    const barChartOptions = useMemo(() => ({
        chart: { type: 'bar', height: 350, toolbar: { show: true }, fontFamily: 'inherit', background: 'transparent' },
        plotOptions: { bar: { borderRadius: 4, horizontal: false, distributed: true, dataLabels: { position: 'top' } } },
        colors: barChartData.chartColors, // Use emotion colors
        dataLabels: { enabled: true, offsetY: -20, style: { fontSize: '12px', colors: ["#333"] }, formatter: function (val) { return val > 0 ? val : ""; } },
        xaxis: { categories: barChartData.categories, title: { text: 'Emotion', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' } }, tooltip: { enabled: false } },
        yaxis: { title: { text: 'Total Entries Count', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' }, formatter: function (val) { return Math.floor(val); } }, tickAmount: 5 },
        legend: { show: false }, // Hide legend as colors match categories
        tooltip: { enabled: true, y: { formatter: function (val) { return val + " entries" } } },
        grid: { borderColor: '#f0f0f0', yaxis: { lines: { show: true } }, xaxis: { lines: { show: false } } }
    }), [barChartData.categories, barChartData.chartColors]);

    // --- Styles ---
    const fixedHeaderStyle = { position: 'fixed', top: 0, left: 0, width: '100%', backdropFilter: 'blur(8px)', padding: '15px 0', textAlign: 'center', zIndex: 10, fontSize: '20px', fontWeight: 'bold', color: '#333',};
    const headerWordStyle = { fontWeight: 'normal', fontSize: '1.2em', fontFamily: 'serif' };
    const contentAreaStyle = { paddingTop: '80px', paddingBottom: '40px', width: '100%', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', textAlign: 'center', flex: 1, minHeight: 'calc(100vh - 80px)' }; // Wider max-width
    const topNavContainerStyle = { display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '40px', width: '100%' };
    const mainHeadingStyle = { fontSize: '2.8em', fontWeight: 'bold', color: '#000000', textAlign: 'center', marginBottom: '30px'};
    const journalWordStyle = { fontWeight: 'normal', fontFamily: 'serif',};
    const chartContainerBaseStyle = { width: '95%', maxWidth: '800px', height: '380px', marginBottom: '15px', marginTop: '10px', border: '1px solid #eee', borderRadius: '8px', background: '#fdfdfd', boxShadow: '0 1px 4px rgba(0,0,0,0.08)', padding: '15px 10px 10px 10px' }; // Base style for chart containers
    const heatmapContainerStyle = { ...chartContainerBaseStyle, height: '350px' }; // Specific height for heatmap
    const barChartContainerStyle = { ...chartContainerBaseStyle, height: '380px', marginTop: '30px' }; // Specific height/margin for bar chart
    const subtitleStyle = { fontSize: '1em', color: '#444', textAlign: 'center', marginBottom: '30px', marginTop: '5px', fontStyle: 'italic'}; // Reduced bottom margin
    const chartTitleStyle = { marginTop: '30px', marginBottom: '0px', color: '#333', fontWeight: '600', fontSize: '1.4em' }; // Style for chart titles
    // --- End Styles ---

    const helpMessageContainerStyle = {
      width: '90%',
      maxWidth: '800px',
      marginTop: '10px', // Space after bar chart subtitle
      padding: '20px',
      border: '1px solid #F5C6CB', // Soft red border
      backgroundColor: '#F8D7DA', // Soft pink background
      borderRadius: '8px',
      textAlign: 'left', // Align text left within the box
      color: '#721C24',  // Dark red text for readability
      fontSize: '14px',
      lineHeight: '1.6',
  };
  const disclaimerStyle = {
      fontWeight: 'bold',
      marginBottom: '10px',
      fontSize: '15px',
  };
  const resourceListStyle = {
      listStyle: 'disc',
      paddingLeft: '25px',
      margin: '10px 0 0 0',
  };
   const resourceLinkStyle = { // Basic link styling
       color: '#721C24', // Match text color
       textDecoration: 'underline',
   };


    return (
        <>
            {/* Fixed Header */}
            <div style={fixedHeaderStyle}>
                emotional <span style={headerWordStyle}>journal</span>
            </div>

            <AnimatedBackgroundContainer>
                 {/* Content Area Wrapper */}
                <div style={contentAreaStyle}>
                    {/* Top Navigation */}
                    <div style={topNavContainerStyle}>
                         <Link to="/" style={{ textDecoration: 'none' }}><StyledNavButton>Home</StyledNavButton></Link>
                         <Link to="/journal-entry" style={{ textDecoration: 'none' }}><StyledNavButton>Journal Entry</StyledNavButton></Link>
                         <Link to="/journal-history" style={{ textDecoration: 'none' }}><StyledNavButton>View History</StyledNavButton></Link>
                    </div>

                    {/* Main Heading */}
                    <h1 style={mainHeadingStyle}>
                        Welcome to your <span style={journalWordStyle}>journal report</span>
                    </h1>

                    {/* ApexCharts Heatmap */}
                    <h2 style={chartTitleStyle}>Emotion Intensity by Depression Level</h2>
                    <div style={heatmapContainerStyle}>
                        {heatmapSeries && heatmapSeries.length > 0 ? (
                             <Chart options={heatmapOptions} series={heatmapSeries} type="heatmap" width="100%" height="100%" />
                        ) : (
                            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#888', fontStyle: 'italic', padding: '20px'}}>
                                 {Object.keys(allStoredEntries).length > 0 ? 'No entries with recognized emotions found.' : 'Enter some journal entries to see a report!'}
                            </div>
                        )}
                    </div>
                    <p style={subtitleStyle}>
                        Heatmap showing total counts of each emotion grouped by inferred depression level.
                    </p>

                    {/* ApexCharts Bar Chart */}
                    <h2 style={chartTitleStyle}>Total Emotion Breakdown</h2>
                    <div style={barChartContainerStyle}>
                        {barChartData.series.length > 0 && barChartData.series[0].data.some(d => d > 0) ? (
                             <Chart options={barChartOptions} series={barChartData.series} type="bar" width="100%" height="100%" />
                        ) : (
                            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#888', fontStyle: 'italic'}}>
                                 No emotion counts to display.
                            </div>
                        )}
                    </div>
                     <p style={subtitleStyle}>
                        Bar chart showing the total number of entries logged for each primary emotion.
                    </p>

 {/* == NEW: Conditional Help Message == */}
 {showHelpMessage && (
                        <div style={helpMessageContainerStyle}>
                            <p style={disclaimerStyle}>
                                Important Disclaimer:
                            </p>
                            <p>
                                This journal app analyzes text patterns and infers emotions; it cannot provide a medical diagnosis. The information presented here is not a substitute for professional medical advice, diagnosis, or treatment.
                            </p>
                            <p>
                                If you're consistently feeling down, overwhelmed, or finding it hard to cope, please know that support is available. Talking to a trusted friend, family member, or a mental health professional can make a difference.
                            </p>
                            <p>
                                Mental Health Resources in Singapore:
                                <ul style={resourceListStyle}>
                                    <li>
                                        Samaritans of Singapore (SOS) 24/7 Hotline: <strong>1-767</strong> or <strong>1-SOS</strong>
                                    </li>
                                    <li>
                                        Institute of Mental Health (IMH) Mental Health Helpline (24/7): <strong>6389 2222</strong>
                                    </li>
                                    <li>
                                        Singapore Association for Mental Health (SAMH): <strong>1800 283 7019</strong>
                                    </li>
                                    <li>
                                        CHAT (Youth mental health): <a href="https://www.chat.mentalhealth.sg" target="_blank" rel="noopener noreferrer" style={resourceLinkStyle}>chat.mentalhealth.sg</a>
                                    </li>
                                    {/* Add other relevant resources if desired */}
                                </ul>
                            </p>
                             <p>
                                Please reach out if you need help. You are not alone.
                             </p>
                        </div>
                    )}
                    {/* == End Conditional Help Message == */}

                 </div>
            </AnimatedBackgroundContainer>
        </>
    );
}

export default JournalReport;
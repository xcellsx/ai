// src/JournalReport.js
import React, { useMemo, useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import Chart from 'react-apexcharts';
// import dynamic from 'next/dynamic';
// const Chart = dynamic(() => import('react-apexcharts'), { ssr: false });

import AnimatedBackgroundContainer from './AnimatedBackgroundContainer';

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

// --- Constants & Helper Logic (Ideally move to a shared utils file) ---
const emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];
const emotionColors = {
  sadness: '#BCD8EC', joy: '#F9E1A8', love: '#FFCBE1',
  anger: '#FFDAB4', fear: '#DCCCEC', surprise: '#D6E5BD',
};

// Labels used in the legend/messages
const depressionCategories = {
    high: "Severe Depression",
    mid: "Moderate Depression",
    low: "Not Depressed",
    na: "No Entry",
};
// Colors associated with each category key
const depressionColors = {
    high: '#FFDAB4', // Corresponds to 'Severe' in the report messages
    mid: '#F9E1A8',  // Corresponds to 'Moderate' in the report messages
    low: '#D6E5BD',  // Corresponds to 'Not Depressed' in the report messages
    na: '#EAEAEA',
};

// --- Function to calculate derived daily depression level (from JournalHistory logic) ---
const getDayDepressionSummary = (dayEntriesArray) => {
    if (!dayEntriesArray || dayEntriesArray.length === 0) {
        return 'na'; // No entries for the day
    }
    let sadnessCount = 0, angerCount = 0, fearCount = 0;
    let joyCount = 0, loveCount = 0, surpriseCount = 0;

    dayEntriesArray.forEach(entry => {
        if (entry.emotion) {
             switch (entry.emotion) {
                case 'sadness': sadnessCount++; break;
                case 'anger': angerCount++; break;
                case 'fear': fearCount++; break;
                case 'joy': joyCount++; break;
                case 'love': loveCount++; break;
                case 'surprise': surpriseCount++; break;
                default: break;
            }
        }
    });
    const negativeCount = sadnessCount + angerCount + fearCount;
    const positiveCount = joyCount + loveCount + surpriseCount;
    if (negativeCount === 0 && positiveCount > 0) return 'low';
    if (negativeCount >= 5) return 'high';
    if (negativeCount > 0 && negativeCount < 5) return 'mid';
    return 'na'; // Fallback
};
// --- End Constants & Helper Logic ---

// --- Constants specifically for the Heatmap (using original depression data) ---
const heatmapDepressionLabels = ["Severe Depression", "Moderate Depression", "Not Depressed"];
// --- End Heatmap Constants ---


function JournalReport() {
    const [allStoredEntries, setAllStoredEntries] = useState({});

    // --- Fetch ALL Data from Local Storage ONCE on Mount ---
    useEffect(() => {
        try {
            const entries = JSON.parse(localStorage.getItem('journalEntries') || '{}');
            setAllStoredEntries(entries);
        } catch (error) {
            console.error("Failed to load entries from Local Storage:", error);
            setAllStoredEntries({});
        }
    }, []);
    // --- End Fetch Data ---

    // --- Calculate TOTAL Emotion Counts (for Bar Chart - Unchanged) ---
    const emotionCounts = useMemo(() => {
        const counts = {};
        emotions.forEach(emotion => { counts[emotion] = 0; });
        Object.values(allStoredEntries).forEach(dayEntriesArray => {
            if (Array.isArray(dayEntriesArray)) {
                dayEntriesArray.forEach(entry => {
                    if (entry && entry.emotion && counts.hasOwnProperty(entry.emotion)) {
                        counts[entry.emotion]++;
                    }
                });
            }
        });
        return counts;
    }, [allStoredEntries]);
    // --- End Calculation ---

    // --- *** MODIFIED: Determine the MOST FREQUENT *DAILY DERIVED* Depression Level *** ---
    const dominantDepressionLevel = useMemo(() => {
        // Counts for how many DAYS fall into each derived category
        const dailyDerivedCounts = { high: 0, mid: 0, low: 0 };

        // Iterate through each day's entries
        Object.values(allStoredEntries).forEach(dayEntriesArray => {
            if (Array.isArray(dayEntriesArray) && dayEntriesArray.length > 0) {
                // Calculate the derived level for this specific day
                const derivedLevelKey = getDayDepressionSummary(dayEntriesArray); // Returns 'high', 'mid', 'low', or 'na'

                // Increment the count for the corresponding category key if it's not 'na'
                if (dailyDerivedCounts.hasOwnProperty(derivedLevelKey)) {
                    dailyDerivedCounts[derivedLevelKey]++;
                }
            }
        });

        // Find the maximum count among the derived daily levels
        const maxCount = Math.max(dailyDerivedCounts.high, dailyDerivedCounts.mid, dailyDerivedCounts.low);

        // If no days had relevant entries to derive a level, return null
        if (maxCount === 0) {
             // console.log("Dominant Derived Daily Depression Level: None Found");
            return null;
        }

        // Determine the dominant level *label* based on frequency, prioritizing severity in ties
        // Returns the *LABEL* (e.g., "Severe Depression") expected by the message components
        if (dailyDerivedCounts.high === maxCount) {
             // console.log("Dominant Derived Daily Depression Level: Severe");
            return depressionCategories.high; // "Severe Depression"
        } else if (dailyDerivedCounts.mid === maxCount) {
             // console.log("Dominant Derived Daily Depression Level: Moderate");
            return depressionCategories.mid; // "Moderate Depression"
        } else { // Only remaining possibility is low === maxCount
             // console.log("Dominant Derived Daily Depression Level: Not Depressed");
            return depressionCategories.low; // "Not Depressed"
        }

    }, [allStoredEntries]); // Recalculate when fetched data changes
    // --- End Modified Depression Level Calculation ---


    // --- Process data for Heatmap (Unchanged Logic - Uses ORIGINAL entry.depression) ---
    const heatmapSeries = useMemo(() => {
        const counts = {};
        heatmapDepressionLabels.forEach(depLabel => {
            counts[depLabel] = {};
            emotions.forEach(emoLabel => {
                counts[depLabel][emoLabel] = 0;
            });
        });
        Object.values(allStoredEntries).forEach(dayEntriesArray => {
            if (Array.isArray(dayEntriesArray)) {
                dayEntriesArray.forEach(entry => {
                    const emotion = entry.emotion;
                    const depression = entry.depression; // Using the original field here
                    if (emotion && emotions.includes(emotion) && depression && heatmapDepressionLabels.includes(depression)) {
                        counts[depression][emotion]++;
                    }
                });
            }
        });
        const series = heatmapDepressionLabels.map(depLabel => {
            const dataPoints = emotions.map(emoLabel => ({
                x: emoLabel.charAt(0).toUpperCase() + emoLabel.slice(1),
                y: counts[depLabel][emoLabel]
            }));
            return { name: depLabel, data: dataPoints };
        });
        return series;
    }, [allStoredEntries]);
    // --- End Heatmap Data Calculation ---

    // --- Configure ApexCharts Heatmap Options (Unchanged) ---
    const heatmapOptions = useMemo(() => ({
        chart: { type: 'heatmap', toolbar: { show: true }, fontFamily: 'inherit', background: 'transparent' },
        plotOptions: {
            heatmap: {
                shadeIntensity: 0.7, enableShades: true, radius: 4, useFillColorAsStroke: false,
                colorScale: {
                    ranges: [
                        { from: 0, to: 0, name: '0 entries', color: '#FFFFFF' }, // Explicitly handle 0
                        { from: 1, to: 5, name: '1-5 entries', color: '#C8E6C9' },
                        { from: 6, to: 10, name: '6-10 entries', color: '#81C784' },
                        { from: 11, to: 20, name: '11-20 entries', color: '#4CAF50' },
                        { from: 21, to: 50, name: '21-50 entries', color: '#388E3C' },
                        { from: 51, to: 1000, name: '>50 entries', color: '#1B5E20' }
                    ]
                }
            }
        },
        dataLabels: { enabled: true, style: { fontSize: '12px', colors: ['#333'] }, formatter: function(val) { return val > 0 ? val : ''; } }, // Hide label if 0
        xaxis: { type: 'category', title: { text: 'Emotion Category', style: { color: '#555', fontSize: '13px' } }, tickPlacement: 'on', labels: { style: { colors: '#555', fontSize: '12px' } } },
        yaxis: { title: { text: 'Actual Recorded Depression Level', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' } } },
        stroke: { width: 1, colors: ['#fff'] },
        tooltip: {
            enabled: true,
            y: {
                formatter: function (value, { series, seriesIndex, dataPointIndex, w }) {
                    const emotion = w.globals.seriesX[seriesIndex]?.[dataPointIndex] || w.globals.labels[dataPointIndex];
                    const depressionLevel = series[seriesIndex]?.name;
                    if (value === 0) return '0 entries'; // Show 0 if value is 0
                    if (!emotion || !depressionLevel) return `${value} entries`;
                    return `${value} entries (${emotion} / ${depressionLevel})`;
                }
            },
            marker: { show: false }
        }
    }), []);
    // --- End Heatmap Options ---

    // --- Process data for Bar Chart (Unchanged) ---
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

    // --- Configure ApexCharts Bar Chart Options (Unchanged) ---
    const barChartOptions = useMemo(() => ({
        chart: { type: 'bar', height: 350, toolbar: { show: true }, fontFamily: 'inherit', background: 'transparent' },
        plotOptions: { bar: { borderRadius: 4, horizontal: false, distributed: true, dataLabels: { position: 'top' } } },
        colors: barChartData.chartColors,
        dataLabels: { enabled: true, offsetY: -20, style: { fontSize: '12px', colors: ["#333"] }, formatter: function (val) { return val > 0 ? val : ""; } },
        xaxis: { categories: barChartData.categories, title: { text: 'Emotion', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' } }, tooltip: { enabled: false } },
        yaxis: { title: { text: 'Total Entries Count', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' }, formatter: function (val) { return Math.floor(val); } }, tickAmount: 5 },
        legend: { show: false },
        tooltip: { enabled: true, y: { formatter: function (val) { return val + " entries" } } },
        grid: { borderColor: '#f0f0f0', yaxis: { lines: { show: true } }, xaxis: { lines: { show: false } } }
    }), [barChartData.categories, barChartData.chartColors]);

    // --- Styles (Unchanged) ---
    const fixedHeaderStyle = { position: 'fixed', top: 0, left: 0, width: '100%', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)', padding: '15px 0', textAlign: 'center', zIndex: 10, fontSize: '20px', fontWeight: 'bold', color: '#333',} ;
    const headerWordStyle = { fontWeight: 'normal', fontSize: '1.2em', fontFamily: 'serif' };
    const contentAreaStyle = { paddingTop: '80px', paddingBottom: '40px', width: '100%', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', textAlign: 'center', flex: 1, minHeight: 'calc(100vh - 80px)' };
    const topNavContainerStyle = { display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '40px', width: '100%' };
    const mainHeadingStyle = { fontSize: '2.8em', fontWeight: 'bold', color: '#000000', textAlign: 'center', marginBottom: '30px'};
    const journalWordStyle = { fontWeight: 'normal', fontFamily: 'serif',};
    const chartContainerBaseStyle = { width: '95%', maxWidth: '800px', height: '380px', marginBottom: '15px', marginTop: '10px', border: '1px solid #eee', borderRadius: '8px', background: '#fdfdfd', boxShadow: '0 1px 4px rgba(0,0,0,0.08)', padding: '15px 10px 10px 10px' };
    const heatmapContainerStyle = { ...chartContainerBaseStyle, height: '350px' };
    const barChartContainerStyle = { ...chartContainerBaseStyle, height: '380px', marginTop: '30px' };
    const subtitleStyle = { fontSize: '1em', color: '#444', textAlign: 'center', marginBottom: '30px', marginTop: '5px', fontStyle: 'italic'};
    const chartTitleStyle = { marginTop: '30px', marginBottom: '0px', color: '#333', fontWeight: '600', fontSize: '1.4em' };
    const messageBaseStyle = { width: '90%', maxWidth: '800px', marginTop: '20px', padding: '20px', borderRadius: '8px', textAlign: 'left', fontSize: '14px', lineHeight: '1.6', };
    const severeMessageContainerStyle = { ...messageBaseStyle, border: '1px solid #F5C6CB', backgroundColor: '#F8D7DA', color: '#721C24', };
    const moderateMessageContainerStyle = { ...messageBaseStyle, border: '1px solid #BEE5EB', backgroundColor: '#D1ECF1', color: '#0C5460', };
    const happyMessageContainerStyle = { ...messageBaseStyle, border: '1px solid #C3E6CB', backgroundColor: '#D4EDDA', color: '#155724', };
    const messageTitleStyle = { fontWeight: 'bold', marginBottom: '10px', fontSize: '15px', };
    const resourceListStyle = { listStyle: 'disc', paddingLeft: '25px', margin: '10px 0 0 0', };
    const resourceLinkStyle = { color: 'inherit', textDecoration: 'underline', };
    // --- End Styles ---

    // --- Message Content (Date updated) ---
    const severeMessage = (
        <div style={severeMessageContainerStyle}>
             <p style={messageTitleStyle}>Important Disclaimer & Support Resources:</p>
             <p>This journal app analyzes text patterns and infers emotions; it cannot provide a medical diagnosis. The information presented here is not a substitute for professional medical advice, diagnosis, or treatment.</p>
             <p>Based on your journal entries, reflections associated with significant distress appear most frequently across days. If you're consistently feeling down, overwhelmed, or finding it hard to cope, please know that support is available and reaching out is a sign of strength.</p>
             <p>Mental Health Resources in Singapore (Checked as of April 21, 2025):</p>
             <ul style={resourceListStyle}>
                 <li>Samaritans of Singapore (SOS) 24/7 Hotline: <strong>1-767</strong></li>
                 <li>Institute of Mental Health (IMH) Mental Health Helpline (24/7): <strong>6389 2222</strong></li>
                 <li>Singapore Association for Mental Health (SAMH): <strong>1800 283 7019</strong> (Mon-Fri, 9am-6pm)</li>
                 <li>CHAT (Youth mental health, ages 16-30): <a href="https://www.chat.mentalhealth.sg" target="_blank" rel="noopener noreferrer" style={resourceLinkStyle}>chat.mentalhealth.sg</a></li>
             </ul>
             <p>Please reach out if you need help. You are not alone.</p>
        </div>
    );

    const moderateMessage = (
         <div style={moderateMessageContainerStyle}>
             <p style={messageTitleStyle}>Reflecting on Your Entries:</p>
             <p>Your journal entries most often reflect days with challenging or moderately difficult emotions. Recognizing and acknowledging these patterns is an important part of understanding your emotional landscape.</p>
             <p>Remember that emotional well-being is a journey with natural fluctuations. Consider exploring small, consistent activities that bring moments of calm or gentle joy.</p>
             <p>If these feelings frequently feel overwhelming across days, talking to someone you trust or considering professional support can offer valuable perspective and coping strategies. (Resource numbers are available if needed).</p>
             <p>Be patient and kind to yourself through this process.</p>
         </div>
    );

     const happyMessage = (
         <div style={happyMessageContainerStyle}>
             <p style={messageTitleStyle}>A Positive Outlook!</p>
             <p>It's great to see that your journal entries most frequently reflect days where positive emotions were dominant or significant negative emotions were less prevalent! Journaling is a fantastic tool for maintaining self-awareness.</p>
             <p>Keep nurturing your emotional well-being through practices that support you, whether it's journaling, connecting with others, enjoying hobbies, or simply taking time for yourself.</p>
             <p>Continue the excellent habit of checking in with yourself!</p>
         </div>
    );
    // --- End Message Content ---


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

                    {/* ApexCharts Heatmap (Shows correlation with ACTUAL recorded depression) */}
                    <h2 style={chartTitleStyle}>Emotion Counts by Actual Depression Level</h2>
                    <div style={heatmapContainerStyle}>
                        {heatmapSeries && heatmapSeries.length > 0 && heatmapSeries.some(s => s.data.some(dp => dp.y > 0)) ? (
                            <Chart options={heatmapOptions} series={heatmapSeries} type="heatmap" width="100%" height="100%" />
                        ) : (
                            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#888', fontStyle: 'italic', padding: '20px', textAlign: 'center'}}>
                                {Object.keys(allStoredEntries).length > 0 ? 'No entries with valid emotion/depression data found for heatmap.' : 'Enter some journal entries to see a report!'}
                            </div>
                        )}
                    </div>
                    <p style={subtitleStyle}>
                        Heatmap showing total counts of each emotion grouped by the actual depression level recorded for the entry (excluding N/A).
                    </p>

                    {/* ApexCharts Bar Chart (Shows total emotion counts) */}
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

                    {/* --- Conditional Messages based on MOST FREQUENT *DAILY DERIVED* Depression Level --- */}
                    <h2 style={chartTitleStyle}>Overall Tendency (Based on Daily Emotion Balance)</h2>

                    {/* Display the message corresponding to the dominant *derived* daily level */}
                    {dominantDepressionLevel === depressionCategories.high && severeMessage}
                    {dominantDepressionLevel === depressionCategories.mid && moderateMessage}
                    {dominantDepressionLevel === depressionCategories.low && happyMessage}
                    {!dominantDepressionLevel && Object.keys(allStoredEntries).length > 0 && (
                         <div style={moderateMessageContainerStyle}> {/* Default message if no levels derived */}
                            <p style={messageTitleStyle}>Report Summary:</p>
                            <p>Could not determine a dominant daily trend based on recorded emotions. Ensure your entries include emotions like sadness, anger, fear, joy, love, or surprise.</p>
                         </div>
                    )}
                    {!dominantDepressionLevel && Object.keys(allStoredEntries).length === 0 && (
                         <div style={moderateMessageContainerStyle}> {/* Message if no entries exist */}
                             <p>Start journaling to see your overall tendency based on daily emotion balance!</p>
                         </div>
                    )}
                     <p style={subtitleStyle}>
                     </p>
                    {/* End Conditional Messages */}

                </div>
            </AnimatedBackgroundContainer>
        </>
    );
}

export default JournalReport;
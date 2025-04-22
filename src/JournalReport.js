// src/JournalReport.js
import React, { useMemo, useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import Chart from 'react-apexcharts';

import AnimatedBackgroundContainer from './AnimatedBackgroundContainer';

// --- Styled Components ---
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
const EMOTIONS_LIST = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];

const EMOTION_DISPLAY_COLORS = {
  sadness: '#BCD8EC', joy: '#F9E1A8', love: '#FFCBE1',
  anger: '#FFDAB4', fear: '#DCCCEC', surprise: '#D6E5BD',
};

const DERIVED_DEPRESSION_LABELS = {
    high: "Severe Depression",
    mid: "Moderate Depression",
    low: "Not Depressed",
};

const HEATMAP_DEPRESSION_AXIS_LABELS = ["Severe Depression", "Moderate Depression", "Not Depressed"];

// --- Helper Function: Calculate Derived Daily Depression Level ---
const calculateDailyDerivedLevelKey = (dayEntriesArray) => {
    if (!dayEntriesArray || dayEntriesArray.length === 0) {
        return 'na';
    }
    let sadnessCount = 0, angerCount = 0, fearCount = 0;
    let joyCount = 0, loveCount = 0, surpriseCount = 0;

    dayEntriesArray.forEach(entry => {
        if (entry && entry.emotion && EMOTIONS_LIST.includes(entry.emotion)) {
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
    return 'na';
};

// --- Style Definitions --- Moved Up ---
const fixedHeaderStyle = { position: 'fixed', top: 0, left: 0, width: '100%', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)', padding: '15px 0', textAlign: 'center', zIndex: 10, fontSize: '20px', fontWeight: 'bold', color: '#333',} ;
const headerWordStyle = { fontWeight: 'normal', fontSize: '1.2em', fontFamily: 'serif' };
const contentAreaStyle = { paddingTop: '80px', paddingBottom: '40px', width: '100%', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', textAlign: 'center', flex: 1, minHeight: 'calc(100vh - 80px)' };
const topNavContainerStyle = { display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '40px', width: '100%' };
const mainHeadingStyle = { fontSize: '2.8em', fontWeight: 'bold', color: '#000000', textAlign: 'center', marginBottom: '30px'};
const journalWordStyle = { fontWeight: 'normal', fontFamily: 'serif',};
const chartContainerBaseStyle = { width: '95%', maxWidth: '800px', minHeight: '380px', marginBottom: '15px', marginTop: '10px', border: '1px solid #eee', borderRadius: '8px', background: '#fdfdfd', boxShadow: '0 1px 4px rgba(0,0,0,0.08)', padding: '15px 10px 10px 10px', display: 'flex', flexDirection: 'column', justifyContent: 'center' };
const heatmapContainerStyle = { ...chartContainerBaseStyle, minHeight: '350px' };
const barChartContainerStyle = { ...chartContainerBaseStyle, minHeight: '380px', marginTop: '30px' };
const chartLoadingErrorStyle = { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#888', fontStyle: 'italic', padding: '20px', textAlign: 'center' };
const subtitleStyle = { fontSize: '1em', color: '#444', textAlign: 'center', marginBottom: '30px', marginTop: '5px', fontStyle: 'italic'};
const chartTitleStyle = { marginTop: '30px', marginBottom: '0px', color: '#333', fontWeight: '600', fontSize: '1.4em' };
const messageBaseStyle = { width: '90%', maxWidth: '800px', marginTop: '20px', padding: '20px', borderRadius: '8px', textAlign: 'left', fontSize: '14px', lineHeight: '1.6', };
const severeMessageContainerStyle = { ...messageBaseStyle, border: '1px solid #F5C6CB', backgroundColor: '#F8D7DA', color: '#721C24', };
const moderateMessageContainerStyle = { ...messageBaseStyle, border: '1px solid #BEE5EB', backgroundColor: '#D1ECF1', color: '#0C5460', };
const happyMessageContainerStyle = { ...messageBaseStyle, border: '1px solid #C3E6CB', backgroundColor: '#D4EDDA', color: '#155724', };
const messageTitleStyle = { fontWeight: 'bold', marginBottom: '10px', fontSize: '15px', };
const resourceListStyle = { listStyle: 'disc', paddingLeft: '25px', margin: '10px 0 0 0', };
const resourceLinkStyle = { color: 'inherit', textDecoration: 'underline', };

// --- Component ---
function JournalReport() {
    const [allJournalData, setAllJournalData] = useState({});
    const [isLoading, setIsLoading] = useState(true);
    const [errorLoading, setErrorLoading] = useState(null);

    // --- Effect to Load Data ---
    useEffect(() => {
        setIsLoading(true);
        setErrorLoading(null);
        try {
            const storedDataString = localStorage.getItem('journalEntries');
            const parsedData = storedDataString ? JSON.parse(storedDataString) : {};
            if (typeof parsedData === 'object' && parsedData !== null) {
                 setAllJournalData(parsedData);
            } else {
                console.error("Invalid data format found in Local Storage. Resetting.");
                setAllJournalData({});
                setErrorLoading("Invalid data format in storage.");
            }
        } catch (error) {
            console.error("Failed to load or parse entries from Local Storage:", error);
            setAllJournalData({});
            setErrorLoading("Failed to load journal data.");
        } finally {
             setIsLoading(false);
        }
    }, []);

    // --- Memoized Calculations ---

    const totalEmotionCounts = useMemo(() => {
        const counts = EMOTIONS_LIST.reduce((acc, emotion) => { acc[emotion] = 0; return acc; }, {});
        Object.values(allJournalData).forEach(dayEntriesArray => {
            if (Array.isArray(dayEntriesArray)) {
                dayEntriesArray.forEach(entry => {
                    if (entry && entry.emotion && counts.hasOwnProperty(entry.emotion)) {
                        counts[entry.emotion]++;
                    }
                });
            }
        });
        return counts;
    }, [allJournalData]);

    const heatmapChartSeries = useMemo(() => {
        const counts = {};
        HEATMAP_DEPRESSION_AXIS_LABELS.forEach(depLabel => {
            counts[depLabel] = EMOTIONS_LIST.reduce((acc, emotion) => { acc[emotion] = 0; return acc; }, {});
        });
        Object.values(allJournalData).forEach(dayEntriesArray => {
            if (Array.isArray(dayEntriesArray)) {
                dayEntriesArray.forEach(entry => {
                    const emotion = entry?.emotion;
                    const depression = entry?.depression;
                    if (emotion && EMOTIONS_LIST.includes(emotion) && depression && counts.hasOwnProperty(depression)) {
                        counts[depression][emotion]++;
                    }
                });
            }
        });
        const series = HEATMAP_DEPRESSION_AXIS_LABELS.map(depLabel => {
            const dataPoints = EMOTIONS_LIST.map(emoLabel => ({
                x: emoLabel.charAt(0).toUpperCase() + emoLabel.slice(1),
                y: counts[depLabel][emoLabel] ?? 0
            }));
            return { name: depLabel, data: dataPoints };
        });
        return series;
    }, [allJournalData]);

    const todaysDerivedLevelLabel = useMemo(() => {
        const todayDateKey = new Date().toISOString().split('T')[0];
        const todaysEntries = allJournalData[todayDateKey] || [];
        if (!Array.isArray(todaysEntries) || todaysEntries.length === 0) {
            console.log("Report - Today's Derived Level: No entries found for key:", todayDateKey);
            return null;
        }
        const derivedKey = calculateDailyDerivedLevelKey(todaysEntries);
        if (derivedKey === 'na') {
            console.log("Report - Today's Derived Level: Not applicable (na) for key:", todayDateKey);
            return null;
        }
        const displayLabel = DERIVED_DEPRESSION_LABELS[derivedKey];
        console.log(`Report - Today's Derived Level: ${displayLabel} (key: ${derivedKey})`);
        return displayLabel;
    }, [allJournalData]);

    const barChartDisplayData = useMemo(() => {
        if (!totalEmotionCounts) return { series: [], categories: [], chartColors: [] };
        const categories = EMOTIONS_LIST.map(e => e.charAt(0).toUpperCase() + e.slice(1));
        const data = EMOTIONS_LIST.map(emotion => totalEmotionCounts[emotion] || 0);
        const series = [{ name: 'Total Count', data: data }];
        const chartColors = EMOTIONS_LIST.map(emotion => EMOTION_DISPLAY_COLORS[emotion] || '#888888');
        return { series, categories, chartColors };
    }, [totalEmotionCounts]);

    const heatmapOptions = useMemo(() => ({
        chart: { type: 'heatmap', toolbar: { show: true }, fontFamily: 'inherit', background: 'transparent' },
        plotOptions: {
            heatmap: {
                shadeIntensity: 0.7, enableShades: true, radius: 4, useFillColorAsStroke: false,
                colorScale: {
                    ranges: [ { from: 0, to: 0, name: '0', color: '#FFFFFF' }, { from: 1, to: 5, name: '1-5', color: '#C8E6C9' }, { from: 6, to: 10, name: '6-10', color: '#81C784' }, { from: 11, to: 20, name: '11-20', color: '#4CAF50' }, { from: 21, to: 50, name: '21-50', color: '#388E3C' }, { from: 51, to: 1000, name: '>50', color: '#1B5E20' } ]
                }
            }
        },
        dataLabels: { enabled: true, style: { fontSize: '12px', colors: ['#333'] }, formatter: function(val) { return val > 0 ? val : ''; } },
        xaxis: { type: 'category', title: { text: 'Emotion Category', style: { color: '#555', fontSize: '13px' } }, tickPlacement: 'on', labels: { style: { colors: '#555', fontSize: '12px' } } },
        yaxis: { title: { text: 'Actual Recorded Depression Level', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' }, }, },
        stroke: { width: 1, colors: ['#fff'] },
        tooltip: { enabled: true, y: { formatter: function (value, { series, seriesIndex, dataPointIndex, w }) { const emotion = w.globals.seriesX[seriesIndex]?.[dataPointIndex] || w.globals.labels[dataPointIndex]; const depressionLevel = series[seriesIndex]?.name; if (value === 0) return '0 entries'; if (!emotion || !depressionLevel) return `${value} entries`; return `${value} entries (${emotion} / ${depressionLevel})`; } }, marker: { show: false } }
     }), []);

    const barChartOptions = useMemo(() => ({
        chart: { type: 'bar', height: 350, toolbar: { show: true }, fontFamily: 'inherit', background: 'transparent' },
        plotOptions: { bar: { borderRadius: 4, horizontal: false, distributed: true, dataLabels: { position: 'top' } } },
        colors: barChartDisplayData.chartColors,
        dataLabels: { enabled: true, offsetY: -20, style: { fontSize: '12px', colors: ["#333"] }, formatter: function (val) { return val > 0 ? val : ""; } },
        xaxis: { categories: barChartDisplayData.categories, title: { text: 'Emotion', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' } }, tooltip: { enabled: false } },
        yaxis: { title: { text: 'Total Entries Count', style: { color: '#555', fontSize: '13px' } }, labels: { style: { colors: '#555', fontSize: '12px' }, formatter: function (val) { return Math.floor(val); } }, allowDecimals: false, tickAmount: 5 },
        legend: { show: false },
        tooltip: { enabled: true, y: { formatter: function (val) { return val + " entries" } } },
        grid: { borderColor: '#f0f0f0', yaxis: { lines: { show: true } }, xaxis: { lines: { show: false } } }
    }), [barChartDisplayData.categories, barChartDisplayData.chartColors]);

    // --- Message Component Definitions --- Now defined AFTER styles ---
    const severeMessage = (
        <div style={severeMessageContainerStyle}>
            <p style={messageTitleStyle}>Important Disclaimer & Support Resources:</p>
            <p>This journal app analyzes text patterns and infers emotions; it cannot provide a medical diagnosis...</p>
            <p>Based on the emotion balance of your entries **today**, the summary suggests a state indicative of significant distress ("Severe"). If you're feeling down...</p>
            <p>Mental Health Resources in Singapore:</p>
            <ul style={resourceListStyle}>
    <li>Samaritans of Singapore (SOS) 24/7 Hotline: <strong>1-767</strong></li>
    <li>Institute of Mental Health (IMH) Mental Health Helpline (24/7): <strong>6389 2222</strong></li><li>Singapore Association for Mental Health (SAMH): <strong>1800 283 7019</strong> (Mon-Fri, 9am-6pm)</li>
    <li>CHAT (Youth mental health, ages 16-30): <a href="https://www.chat.mentalhealth.sg" target="_blank" rel="noopener noreferrer" style={resourceLinkStyle}>chat.mentalhealth.sg</a></li>
 </ul>
<p>Please reach out if you need help. You are not alone.</p>
        </div>
    );
    const moderateMessage = (
       <div style={moderateMessageContainerStyle}>
           <p style={messageTitleStyle}>Reflecting on Your Entries Today:</p>
           <p>Based on the emotion balance of your entries **today**, the summary suggests a state indicative of moderate distress ("Moderate"). Recognizing and acknowledging these feelings is an important part of understanding your emotional landscape for the day.</p>
           <p>Remember that emotional well-being fluctuates daily. Consider exploring small activities that might bring moments of calm or gentle joy.</p>
           <p>If these feelings feel overwhelming today, talking to someone you trust or considering professional support can offer valuable perspective and coping strategies. (Resource numbers are available if needed).</p>
           <p>Be patient and kind to yourself.</p>


       </div>
    );
     const happyMessage = (
       <div style={happyMessageContainerStyle}>
           <p style={messageTitleStyle}>A Positive Outlook Today!</p>
           <p>Based on the emotion balance of your entries **today**, the summary indicates a state where positive emotions were dominant... ("Not Depressed")!</p>
           <p>Keep nurturing your emotional well-being through practices that support you.</p>
           <p>Continue the excellent habit of checking in with yourself!</p>
       </div>
    );
    const noEntriesTodayMessage = (
        <div style={moderateMessageContainerStyle}>
           <p style={messageTitleStyle}>Today's Summary:</p>
           <p>No journal entries recorded for today, or today's entries could not be categorized based on emotion balance. Write an entry to see today's summary!</p>
       </div>
   );
   const loadingMessage = (
        <div style={moderateMessageContainerStyle}>
           <p>Loading report data...</p>
       </div>
   );
    const errorMessage = (
        <div style={severeMessageContainerStyle}>
            <p style={messageTitleStyle}>Error Loading Data</p>
            <p>{errorLoading || "An unknown error occurred while loading journal entries."}</p>
        </div>
    );

    // --- Render Logic ---
    return (
        <>
            <div style={fixedHeaderStyle}>
                emotional <span style={headerWordStyle}>journal</span>
            </div>

            <AnimatedBackgroundContainer>
                <div style={contentAreaStyle}>
                    {/* Top Navigation */}
                    <div style={topNavContainerStyle}>
                        <Link to="/" style={{ textDecoration: 'none' }}><StyledNavButton>Home</StyledNavButton></Link>
                        <Link to="/journal-entry" style={{ textDecoration: 'none' }}><StyledNavButton>Journal Entry</StyledNavButton></Link>
                        <Link to="/journal-history" style={{ textDecoration: 'none' }}><StyledNavButton>View History</StyledNavButton></Link>
                    </div>

                    <h1 style={mainHeadingStyle}>
                        Welcome to your <span style={journalWordStyle}>journal report</span>
                    </h1>

                    {/* Display loading or error state for charts */}
                    {isLoading && <p>Loading charts...</p>}
                    {errorLoading && <p style={{ color: 'red' }}>Error loading data: {errorLoading}</p>}

                    {!isLoading && !errorLoading && (
                        <>
                            {/* Heatmap Section */}
                            <h2 style={chartTitleStyle}>Emotion Counts by Actual Depression Level</h2>
                            <div style={heatmapContainerStyle}>
                                {heatmapChartSeries && heatmapChartSeries.length > 0 && heatmapChartSeries.some(s => s.data.some(dp => dp.y > 0)) ? (
                                    <Chart options={heatmapOptions} series={heatmapChartSeries} type="heatmap" width="100%" height="330px" />
                                ) : (
                                    <div style={chartLoadingErrorStyle}>
                                        {Object.keys(allJournalData).length > 0 ? 'No entries with valid emotion/depression data found for heatmap.' : 'Enter some journal entries to see a report!'}
                                    </div>
                                )}
                            </div>
                            <p style={subtitleStyle}>
                                Heatmap showing total counts of each emotion grouped by the actual depression level recorded for the entry.
                            </p>

                            {/* Bar Chart Section */}
                            <h2 style={chartTitleStyle}>Total Emotion Breakdown</h2>
                            <div style={barChartContainerStyle}>
                                {barChartDisplayData.series.length > 0 && barChartDisplayData.series[0].data.some(d => d > 0) ? (
                                    <Chart options={barChartOptions} series={barChartDisplayData.series} type="bar" width="100%" height="360px" />
                                ) : (
                                    <div style={chartLoadingErrorStyle}>
                                        No emotion counts to display.
                                    </div>
                                )}
                            </div>
                            <p style={subtitleStyle}>
                                Bar chart showing the total number of entries logged for each primary emotion.
                            </p>

                             {/* Today's Summary Message Section */}
                            <h2 style={chartTitleStyle}>Today's Summary (Based on Emotion Balance)</h2>
                            {/* Display specific message based on today's derived level */}
                            {todaysDerivedLevelLabel === DERIVED_DEPRESSION_LABELS.high && severeMessage}
                            {todaysDerivedLevelLabel === DERIVED_DEPRESSION_LABELS.mid && moderateMessage}
                            {todaysDerivedLevelLabel === DERIVED_DEPRESSION_LABELS.low && happyMessage}
                            {/* Fallback if no level determined for today */}
                            {!todaysDerivedLevelLabel && noEntriesTodayMessage}

                            <p style={subtitleStyle}>
                                This message reflects the derived emotional state based on all entries recorded today.
                            </p>
                        </>
                    )}
                     {/* Display overall loading/error message if needed */}
                     {isLoading && loadingMessage}
                     {errorLoading && !isLoading && errorMessage}

                </div>
            </AnimatedBackgroundContainer>
        </>
    );
}

export default JournalReport;
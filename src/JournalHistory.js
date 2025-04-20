// src/JournalHistory.js
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import AnimatedBackgroundContainer from './AnimatedBackgroundContainer';
// Import if moved to shared file: import { StyledNavButton } from './StyledComponents';

// Define StyledNavButton locally (ensure consistency with other files or import)
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

// --- Color & Category Definitions ---
const emotionColors = {
  sadness: '#BCD8EC', joy: '#F9E1A8', love: '#FFCBE1',
  anger: '#FFDAB4', fear: '#DCCCEC', surprise: '#D6E5BD',
  default: '#EAEAEA', // Default color for days with no entry
};
const emotions = Object.keys(emotionColors).filter(k => k !== 'default'); // Get emotion names

// Labels used in the legend (These remain the same conceptually)
const depressionCategories = {
    high: "Severe Depression",
    mid: "Moderate Depression",
    low: "Not Depressed",
    na: "No Entry",     // Label for days with no applicable entries based on new rules
};
// Colors associated with each category key (These remain the same)
const depressionColors = {
    high: '#FFDAB4', // Example: Orange/Reddish for high
    mid: '#F9E1A8',  // Example: Yellow for mid
    low: '#D6E5BD',  // Example: Green for low
    na: '#EAEAEA',  // Default Grey
};
// --- End Color & Category Definitions ---

// --- Helper Functions ---
const formatDateKey = (date) => {
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    return `${year}-${month}-${day}`;
};

// REMOVED/COMMENTED OUT: This function is no longer needed as depression category is derived from emotions.
// const getDepressionCategory = (value) => {
//     if (value === "Severe Depression") return 'high';
//     if (value === "Moderate Depression") return 'mid';
//     if (value === "Not Depressed") return 'low';
//     // Handle 'N/A', null, undefined, or other unexpected values as 'na'
//     return 'na';
// };

// Calculates emotion counts and finds highest count emotion (Unchanged - used for 'emotions' view and potentially modal)
const getDaySummary = (dayEntriesArray) => {
    if (!dayEntriesArray || dayEntriesArray.length === 0) {
        const zeroCounts = {};
        emotions.forEach(e => zeroCounts[e] = 0);
        return { counts: zeroCounts, highestEmotion: null, total: 0 };
    }
    const counts = {};
    emotions.forEach(e => counts[e] = 0); // Initialize

    dayEntriesArray.forEach(entry => {
        if (entry.emotion && counts.hasOwnProperty(entry.emotion)) {
            counts[entry.emotion]++;
        }
    });

    let highestEmotion = null;
    let maxCount = 0;
    // Find highest count emotion (simple tie-breaking: first wins)
    for (const [emotion, count] of Object.entries(counts)) {
        if (count > maxCount) {
            maxCount = count;
            highestEmotion = emotion;
        }
    }
    return { counts, highestEmotion, total: dayEntriesArray.length };
};

// *** MODIFIED: Calculates depression category ('high', 'mid', 'low', 'na') based on emotion counts ***
const getDayDepressionSummary = (dayEntriesArray) => {
    if (!dayEntriesArray || dayEntriesArray.length === 0) {
        return 'na'; // No entries for the day
    }

    // Count relevant emotions
    let sadnessCount = 0;
    let angerCount = 0;
    let fearCount = 0;
    let joyCount = 0;
    let loveCount = 0;
    let surpriseCount = 0;

    dayEntriesArray.forEach(entry => {
        // Only count entries that have a recognized emotion
        if (entry.emotion) {
             switch (entry.emotion) {
                case 'sadness': sadnessCount++; break;
                case 'anger': angerCount++; break;
                case 'fear': fearCount++; break;
                case 'joy': joyCount++; break;
                case 'love': loveCount++; break;
                case 'surprise': surpriseCount++; break;
                default: break; // Ignore other potential emotion strings
            }
        }
    });

    const negativeCount = sadnessCount + angerCount + fearCount;
    const positiveCount = joyCount + loveCount + surpriseCount;

    // --- Apply New Rules ---

    // Rule 1: If only positive emotions (joy, love, surprise) are present -> Not Depressed
    // (Handles cases where negativeCount is 0, but positiveCount > 0)
    if (negativeCount === 0 && positiveCount > 0) {
         return 'low'; // Not Depressed
    }

    // Rule 3: If sum of sadness, anger, fear is 5 or more -> Severe Depression
    // (This rule takes precedence if negative emotions exist, even if positive ones also exist)
    if (negativeCount >= 5) {
        return 'high'; // Severe Depression
    }

    // Rule 2: If sum of sadness, anger, fear is 1 to 4 -> Moderate Depression
    if (negativeCount > 0 && negativeCount < 5) {
        return 'mid'; // Moderate Depression
    }

    // Fallback: No relevant positive or negative emotions recorded (negativeCount is 0, positiveCount is 0),
    // or other edge cases.
    return 'na'; // No Entry / Not Applicable
};
// --- End Helper Functions ---

function JournalHistory() {
    // State hooks (Unchanged)
    const [viewMode, setViewMode] = useState('emotions'); // 'emotions' or 'depression'
    const [displayDate, setDisplayDate] = useState(new Date()); // Controls the calendar's month/year
    const [allStoredEntries, setAllStoredEntries] = useState({}); // Holds all entries from storage { 'YYYY-MM-DD': [entry1, entry2] }
    const [modalData, setModalData] = useState(null); // State for modal content { date, entries, counts }

    // --- Date Calculation (Unchanged) ---
    const currentYear = displayDate.getFullYear();
    const currentMonth = displayDate.getMonth();
    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const currentMonthName = monthNames[currentMonth];
    const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
    const firstDayOfMonth = new Date(currentYear, currentMonth, 1).getDay(); // 0 = Sunday

    // --- Fetch Data (Unchanged) ---
    useEffect(() => {
        try {
            // Ensure data is parsed correctly, handle potential errors
            const storedData = localStorage.getItem('journalEntries');
            const entries = storedData ? JSON.parse(storedData) : {};
             // Basic validation might be good here depending on data source
            setAllStoredEntries(entries);
        } catch (error) {
            console.error("Failed to load or parse entries from Local Storage:", error);
            setAllStoredEntries({}); // Set to empty if error
        }
    }, []);

    // --- Month Navigation Handlers (Unchanged) ---
    const handlePrevMonth = () => {
        setDisplayDate(prev => new Date(prev.getFullYear(), prev.getMonth() - 1, 1));
    };
     const handleNextMonth = () => {
         setDisplayDate(prev => new Date(prev.getFullYear(), prev.getMonth() + 1, 1));
     };

    // --- Styles (Assume unchanged, keeping brief for clarity) ---
    const fixedHeaderStyle = { position: 'fixed', top: 0, left: 0, width: '100%', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)', padding: '15px 0', textAlign: 'center', zIndex: 10, fontSize: '20px', fontWeight: 'bold', color: '#333', };
    const headerWordStyle = { fontWeight: 'normal', fontSize: '1.2em', fontFamily: 'serif' };
    const contentAreaStyle = { paddingTop: '80px', paddingBottom: '40px', width: '100%', maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', textAlign: 'center', flex: 1, minHeight: 'calc(100vh - 80px)', boxSizing: 'border-box' };
    const topNavContainerStyle = { display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '40px', width: '100%' };
    const mainHeadingStyle = { fontSize: '2.8em', fontWeight: 'bold', color: '#000000', textAlign: 'center', marginBottom: '25px'};
    const journalWordStyle = { fontWeight: 'normal', fontFamily: 'serif',};
    const toggleButtonContainerStyle = { display: 'flex', justifyContent: 'center', marginBottom: '40px', gap: '10px'};
    const toggleButtonStyle = { backgroundColor: 'transparent', color: '#333', border: '1px solid black', padding: '8px 25px', borderRadius: '20px', cursor: 'pointer', fontSize: '14px', fontWeight: '500', transition: 'background-color 0.2s, color 0.2s, border-color 0.2s'};
    const toggleButtonActiveStyle = { backgroundColor: 'black', color: '#FFFFFF', border: '1px solid black'};
    const monthDisplayContainerStyle = { display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '20px', gap: '20px'};
    const monthNavArrowStyle = { width: '35px', height: '35px', borderRadius: '50%', backgroundColor: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888', fontSize: '1.5em', fontWeight: 'bold', transition: 'background-color 0.2s ease', border: '0px solid #CCC', lineHeight: 1 };
    const monthNameStyle = { backgroundColor: 'transparent', color: '#333', padding: '10px 20px', borderRadius: '5px', fontSize: '1.3em', fontWeight: '500', textAlign: 'center', border: '0px solid #CCC'};
    const calendarContainerStyle = { maxWidth: '500px', width: '95%', margin: '0 auto 40px auto' };
    const dayHeaderStyle = { textAlign: 'center', fontWeight: '600', color: '#333', paddingBottom: '8px', fontSize: '0.9em', borderBottom: '1px solid #000', marginBottom: '5px' };
    const calendarGridStyle = { display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '8px', padding: '10px 0' };
    const dayCellStyleBase = { width: '42px', height: '42px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px', fontWeight: 'bold', boxSizing: 'border-box', border: '1px solid transparent', cursor: 'pointer', transition: 'transform 0.1s ease, border-color 0.1s ease, background-color 0.2s ease', color: '#333', boxShadow: '0 1px 3px rgba(0,0,0,0.5)', };
    const emptyDayCellStyle = { ...dayCellStyleBase, backgroundColor: 'transparent', opacity: 0.2, cursor: 'default', color: '#AAA', pointerEvents: 'none', boxShadow: 'none', };
    const legendContainerStyle = { width: '100%', maxWidth: '500px', backgroundColor: 'rgba(255, 255, 255, 0.8)', padding: '15px', borderRadius: '8px', boxShadow: '0 1px 4px rgba(0,0,0,0.1)', border: '1px solid rgba(0,0,0,0.08)', marginTop: '20px' };
    const legendTitleStyle = { fontSize: '15px', fontWeight: '600', color: '#333', marginBottom: '12px', textAlign: 'center', marginTop: '0' };
    const legendItemsContainerStyle = { display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '8px 15px' };
    const legendItemStyle = { display: 'flex', alignItems: 'center', fontSize: '13px', color: '#444' };
    const legendColorDotStyle = { width: '14px', height: '14px', borderRadius: '50%', marginRight: '6px', border: '1px solid rgba(0,0,0,0.1)' };
    const modalOverlayStyle = { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.65)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, backdropFilter: 'blur(3px)' };
    const modalContentStyle = { backgroundColor: '#fff', padding: '25px 30px', borderRadius: '8px', maxWidth: '90%', width: '600px', maxHeight: '85vh', overflowY: 'auto', position: 'relative', boxShadow: '0 5px 20px rgba(0,0,0,0.2)' };
    const modalCloseButtonStyle = { position: 'absolute', top: '8px', right: '12px', background: 'none', border: 'none', fontSize: '2em', cursor: 'pointer', color: '#aaa', padding: '0', lineHeight: 1 };
    const modalEntryStyle = { borderBottom: '1px solid #eee', padding: '12px 0', marginBottom: '12px', textAlign: 'left'}; // Removed pseudo-class for inline
    const modalEntryTextStyle = { fontStyle: 'italic', color: '#555', marginTop: '5px', whiteSpace: 'pre-wrap', fontSize: '0.95em' };
    const modalCountsStyle = { marginBottom: '20px', textAlign: 'left', borderBottom: '1px solid #ccc', paddingBottom: '15px' };
    const modalCountItemStyle = { marginRight: '15px', display: 'inline-flex', alignItems: 'center', fontSize: '14px', marginBottom: '5px' };
    const modalCountDotStyle = { width: '12px', height: '12px', borderRadius: '50%', marginRight: '5px', border: '1px solid rgba(0,0,0,0.1)'};
    // --- End Styles ---

    const totalCells = Math.ceil((firstDayOfMonth + daysInMonth) / 7) * 7;

    return (
        <>
            {/* Fixed Header */}
            <div style={fixedHeaderStyle}>
                emotional <span style={headerWordStyle}>journal</span>
            </div>

            <AnimatedBackgroundContainer>
                <div style={contentAreaStyle}>
                    {/* Top Navigation */}
                    <div style={topNavContainerStyle}>
                         <Link to="/" style={{ textDecoration: 'none' }}><StyledNavButton>Home</StyledNavButton></Link>
                         <Link to="/journal-entry" style={{ textDecoration: 'none' }}><StyledNavButton>Journal Entry</StyledNavButton></Link>
                         <Link to="/journal-report" style={{ textDecoration: 'none' }}><StyledNavButton>View Report</StyledNavButton></Link>
                    </div>

                    {/* Main Heading */}
                    <h1 style={mainHeadingStyle}>
                        Welcome to your <span style={journalWordStyle}>journal history</span>
                    </h1>

                    {/* Toggle Buttons */}
                    <div style={toggleButtonContainerStyle}>
                        <button style={viewMode === 'emotions' ? { ...toggleButtonStyle, ...toggleButtonActiveStyle } : toggleButtonStyle} onClick={() => setViewMode('emotions')} >Emotions</button>
                        <button style={viewMode === 'depression' ? { ...toggleButtonStyle, ...toggleButtonActiveStyle } : toggleButtonStyle} onClick={() => setViewMode('depression')} >Depression</button>
                    </div>

                    {/* Month Display & Navigation */}
                    <div style={monthDisplayContainerStyle}>
                         <button style={monthNavArrowStyle} onClick={handlePrevMonth} title="Previous Month">{'<'}</button>
                         <div style={monthNameStyle}>{currentMonthName} {currentYear}</div>
                         <button style={monthNavArrowStyle} onClick={handleNextMonth} title="Next Month">{'>'}</button>
                    </div>

                    {/* Calendar Container */}
                    <div style={calendarContainerStyle}>
                         {/* Day Headers */}
                         <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', padding: '0 5px' }}>
                              {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (<div key={day} style={dayHeaderStyle}>{day}</div>))}
                         </div>

                         {/* Calendar Grid Cells */}
                        <div style={calendarGridStyle}>
                            {Array.from({ length: totalCells }).map((_, index) => {
                                const dayNumber = index - firstDayOfMonth + 1;
                                const isCurrentMonthDay = dayNumber > 0 && dayNumber <= daysInMonth;

                                let cellStyle = {};
                                let cellContent = null;
                                let dayEntriesArray = []; // Initialize

                                if (isCurrentMonthDay) {
                                    const dateKey = formatDateKey(new Date(currentYear, currentMonth, dayNumber));
                                    dayEntriesArray = allStoredEntries[dateKey] || []; // Get array for the day

                                    let bgColor = emotionColors.default; // Default background
                                    let textColor = '#333'; // Default text color

                                    // --- LOGIC FOR CELL COLOR BASED ON VIEWMODE ---
                                    if (viewMode === 'emotions') {
                                        // Use existing summary function for emotions
                                        const { highestEmotion } = getDaySummary(dayEntriesArray);
                                        bgColor = highestEmotion ? emotionColors[highestEmotion] : emotionColors.default;
                                        // Optional: Adjust text color based on emotion background
                                        // textColor = ...
                                    } else { // viewMode === 'depression'
                                        // Use **MODIFIED** summary function for depression (derives level from emotions)
                                        const derivedDepressionCategory = getDayDepressionSummary(dayEntriesArray);
                                        // Get color based on the derived category key ('high', 'mid', 'low', 'na')
                                        bgColor = depressionColors[derivedDepressionCategory];
                                        textColor = '#333'; // Keep text dark for depression colors
                                    }
                                    // --- END LOGIC FOR CELL COLOR ---

                                    cellStyle = { ...dayCellStyleBase, backgroundColor: bgColor, color: textColor };
                                    cellContent = dayNumber;

                                } else {
                                    cellStyle = emptyDayCellStyle;
                                }

                                // Click Handler to Open Modal (Unchanged - Modal still shows raw entry data)
                                const handleDayClick = () => {
                                     if (isCurrentMonthDay && dayEntriesArray.length > 0) {
                                          const dateKey = formatDateKey(new Date(currentYear, currentMonth, dayNumber));
                                          const summary = getDaySummary(dayEntriesArray); // Emotion summary for modal
                                          // Modal will show raw entry data, including original 'entry.depression' if present.
                                          setModalData({ date: dateKey, entries: dayEntriesArray, counts: summary.counts });
                                     }
                                };

                                return (
                                    <div key={index} style={cellStyle} onClick={handleDayClick} title={isCurrentMonthDay ? `View entries for ${dayNumber}` : ''}>
                                        {cellContent}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Conditional Legends (Unchanged - Legends themselves are still valid) */}
                       {viewMode === 'emotions' && (
                           <div style={legendContainerStyle}>
                               <p style={legendTitleStyle}>Emotion Legend</p>
                               <div style={legendItemsContainerStyle}>
                                   {emotions.map(emotion => (
                                       <div key={emotion} style={legendItemStyle}>
                                           <span style={{...legendColorDotStyle, backgroundColor: emotionColors[emotion] }}></span>
                                           {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                                       </div>
                                   ))}
                                    <div key="default" style={legendItemStyle}>
                                        <span style={{ ...legendColorDotStyle, backgroundColor: emotionColors.default }}></span>
                                        No Entry
                                    </div>
                               </div>
                           </div>
                       )}

                       {viewMode === 'depression' && (
                           <div style={legendContainerStyle}>
                               <p style={legendTitleStyle}>Derived Depression Level</p> {/* Updated title slightly */}
                               <div style={legendItemsContainerStyle}>
                                   {/* Use depressionCategories for legend text */}
                                   {Object.entries(depressionCategories).map(([key, name]) => (
                                       <div key={key} style={legendItemStyle}>
                                           <span style={{...legendColorDotStyle, backgroundColor: depressionColors[key] }}></span>
                                           {name} {/* Display user-friendly name */}
                                       </div>
                                   ))}
                               </div>
                           </div>
                       )}
                    {/* End Conditional Legend */}

                </div>
            </AnimatedBackgroundContainer>

            {/* Modal Rendering (Unchanged - Still shows raw entry data including original depression field if exists) */}
            {modalData && (
                 <div style={modalOverlayStyle} onClick={() => setModalData(null)}>
                      <div style={modalContentStyle} onClick={(e) => e.stopPropagation()}>
                          <button style={modalCloseButtonStyle} onClick={() => setModalData(null)}>&times;</button>
                          <h3>Entries for {modalData.date}</h3>
                          {/* Display Emotion Counts */}
                          <div style={modalCountsStyle}>
                              <h4 style={{marginTop: 0, marginBottom: '10px'}}>Emotion Summary ({modalData.entries.length} {modalData.entries.length === 1 ? 'entry' : 'entries'}):</h4>
                              {Object.entries(modalData.counts).map(([emotion, count]) => (
                                  count > 0 && (
                                      <span key={emotion} style={modalCountItemStyle}>
                                          <span style={{...modalCountDotStyle, backgroundColor: emotionColors[emotion]}}></span>
                                          {emotion.charAt(0).toUpperCase() + emotion.slice(1)}: {count}
                                      </span>
                                  )
                              ))}
                                {Object.values(modalData.counts).every(c => c === 0) && <span>No emotions recorded.</span>}
                          </div>
                          {/* Display Individual Entries */}
                          <h4>Entries:</h4>
                          {modalData.entries
                              .sort((a, b) => { // Robust sorting
                                  const dateA = a.timestamp ? new Date(a.timestamp) : 0;
                                  const dateB = b.timestamp ? new Date(b.timestamp) : 0;
                                  return dateA - dateB;
                              })
                              .map((entry, idx) => (
                                 <div key={idx} style={{...modalEntryStyle, borderBottom: idx === modalData.entries.length - 1 ? 'none' : '1px solid #eee', marginBottom: idx === modalData.entries.length - 1 ? 0 : '12px'}}> {/* Handle last item border */}
                                      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px'}}>
                                          <span>
                                              <strong style={{color: emotionColors[entry.emotion] || '#333'}}>Emotion:</strong> {entry.emotion || 'N/A'} |
                                              {/* This still shows the original depression field from the data */}
                                              <strong> Depression:</strong> {entry.depression || 'N/A'}
                                          </span>
                                          <small style={{color: '#888'}}> {entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}</small>
                                      </div>
                                      <div style={{color: '#555', fontSize: '0.9em', marginBottom: '5px'}}>Bot: {entry.botResponse || 'N/A'}</div>
                                      {entry.text && <div style={modalEntryTextStyle}>"{entry.text}"</div>}
                                 </div>
                          ))}
                      </div>
                 </div>
            )}
             {/* End Modal Rendering */}
        </>
    );
}

export default JournalHistory;
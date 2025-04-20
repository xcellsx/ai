// src/JournalEntry.js
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import AnimatedBackgroundContainer from './AnimatedBackgroundContainer';

// Assuming StyledNavButton is defined here or imported from './StyledComponents'
const StyledNavButton = styled.button`
  background-color: transparent;
  color: #333;
  border: 1px solid black;
  padding: 10px 25px; // Padding for submit button
  margin: 0 5px;      // Margin for nav buttons primarily
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;    // Font size for submit button
  text-decoration: none;
  transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  font-family: inherit;

  &:hover {
    background-color: black;
    color: #FFF;
  }

  &:disabled {
    background-color: transparent;
    color: #AAAAAA;
    border-color: #CCCCCC;
    cursor: not-allowed;
    box-shadow: none;
    opacity: 0.6;
  }

  /* Example of adjusting specific instances if needed later via className or props */
  &.nav-button-style {
      padding: 8px 18px;
      font-size: 14px;
      border-radius: 4px;
  }
`;


function JournalEntry() {
    const [userInput, setUserInput] = useState('');
    const [response, setResponse] = useState('');
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // --- Core Logic ---
    const handleInputChange = (e) => {
        setUserInput(e.target.value);
    };

    const handleSubmit = async () => {
        // Prevent submission if already analyzing or input is empty
        if (isAnalyzing || userInput.trim() === '') {
             if (userInput.trim() === '') {
                 setResponse('Please write something before submitting.');
                 // Clear the message after 3 seconds
                 setTimeout(() => { setResponse(prev => prev === 'Please write something before submitting.' ? '' : prev); }, 3000);
             }
            return;
        }

        setIsAnalyzing(true);
        setResponse('Analyzing your thoughts...');

        try {
        //     // *** IMPORTANT: Update this URL if your Render backend URL changed ***
        //     const backendUrl = 'https://ai-f53i.onrender.com'; // Use the URL for the backend with BOTH models
        //     const endpoint = `${backendUrl}/analyze`;
        //     const apiResponse = await fetch(endpoint, {
        //         method: 'POST',
        //         headers: { 'Content-Type': 'application/json', },
        //         body: JSON.stringify({ text: userInput }),
        //     });

            // localhost version for testing
            const backendUrl = 'http://localhost:5000'; // Use the URL for the backend with BOTH models
            const endpoint = `${backendUrl}/analyze`;
            const apiResponse = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', },
                body: JSON.stringify({ text: userInput }),
            });

            if (!apiResponse.ok) {
                let errorDetails = `Server responded with status: ${apiResponse.status}`;
                try {
                    const errorData = await apiResponse.json();
                    // Use the specific error message from backend if available
                    errorDetails += `. ${errorData.error || JSON.stringify(errorData)}`;
                } catch (e) {
                    errorDetails += ". Could not parse error response body.";
                }
                console.error("Backend error:", errorDetails);
                // Display the error message to the user
                throw new Error(errorDetails);
            }

            // Expecting response like: {'emotion': '...', 'depression': '...'}
            const result = await apiResponse.json();
            console.log("Analysis result from backend:", result);

            // Extract results, providing defaults if keys are missing or null
            const detectedEmotion = result.emotion || 'N/A';
            const detectedDepression = result.depression || 'N/A'; // Get the depression result

            // Format the response string to show both results
            const formattedResponse = `Emotion: ${detectedEmotion} | Depression: ${detectedDepression}. Thank you for sharing.`;
            setResponse(formattedResponse); // Update the UI

            // --- SAVE TO LOCAL STORAGE (Handles Multiple Entries) ---
            try {
                const todayDate = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
                const allEntries = JSON.parse(localStorage.getItem('journalEntries') || '{}');
                const dayEntries = allEntries[todayDate] || []; // Get existing array or init empty

                const newEntryData = {
                    // Store the actual detected values
                    emotion: detectedEmotion,
                    depression: detectedDepression, // Store the actual depression result
                    text: userInput,
                    // Store the formatted string that was shown to the user
                    botResponse: formattedResponse,
                    timestamp: new Date().toISOString() // Add timestamp for ordering
                };

                dayEntries.push(newEntryData); // Append the new entry
                allEntries[todayDate] = dayEntries; // Update the main entries object

                localStorage.setItem('journalEntries', JSON.stringify(allEntries));
                console.log(`Entry saved for ${todayDate}. Total entries for day: ${dayEntries.length}`, newEntryData);

            } catch (storageError) {
                console.error("Failed to save entry to Local Storage:", storageError);
                // Inform user saving failed, but keep the analysis result visible
                setResponse(prev => `${prev} (Error saving entry)`);
            }
            // --- END SAVE TO LOCAL STORAGE ---

            // Clear input only after successful analysis and attempted save
            setUserInput('');

        } catch (error) {
             console.error("Error during submission or analysis:", error);
             // Display the specific error message from the backend or fetch process
             setResponse(`An error occurred: ${error.message}. Please try again.`);
             // Keep user input in the textarea in case of error for retry
        } finally {
            // CRITICAL: Ensure this always runs to re-enable input/button
            setIsAnalyzing(false);
        }
    };
    // --- End Core Logic ---

    // --- Styles ---
    // (Styles remain the same - using inline styles for brevity)
    const fixedHeaderStyle = { position: 'fixed', top: 0, left: 0, width: '100%', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)', padding: '15px 0', textAlign: 'center', zIndex: 10, fontSize: '20px', fontWeight: 'bold', color: '#333', };
    const headerWordStyle = { fontWeight: 'normal', fontSize: '1.2em', fontFamily: 'serif' };
    const contentAreaStyle = { paddingTop: '80px', paddingBottom: '40px', width: '100%', maxWidth: '700px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', textAlign: 'center', flex: 1, minHeight: 'calc(100vh - 80px)', boxSizing: 'border-box' }; // Adjusted minHeight and justifyContent
    const topNavContainerStyle = { display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '40px', width: '100%', };
    const mainHeadingStyle = { fontSize: '2.8em', fontWeight: 'bold', color: '#000000', textAlign: 'center', marginBottom: '10px', };
    const journalWordStyle = { fontWeight: 'normal', fontFamily: 'serif', marginLeft: '0.2em' };
    const subHeadingStyle = { fontSize: '1.1em', color: '#000', textAlign: 'center', marginBottom: '30px', };
    const textAreaStyle = { width: '100%', minHeight: '150px', padding: '15px', marginBottom: '20px', border: '1px solid #CCC', borderRadius: '5px', backgroundColor: 'rgba(234, 234, 234, 0.6)', fontSize: '1em', color: '#333', resize: 'vertical', boxSizing: 'border-box', fontFamily: 'inherit', boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.1)', '::placeholder': { color: '#777' }, };
    const outputBoxStyle = { width: '100%', minHeight: '60px', padding: '15px', marginBottom: '30px', border: '1px dashed #CCC', borderRadius: '5px', backgroundColor: 'rgba(234, 234, 234, 0.6)', fontSize: '1em', color: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', boxSizing: 'border-box', fontFamily: 'inherit', lineHeight: '1.5', };
    // --- End Styles ---

    const isProcessing = isAnalyzing;

    return (
        <>
            {/* Fixed Header */}
            <div style={fixedHeaderStyle}>
                emotional <span style={headerWordStyle}>journal</span>
            </div>

            {/* Animated Background and Main Content */}
            <AnimatedBackgroundContainer>
                <div style={contentAreaStyle}>

                    {/* Top Navigation Buttons */}
                    <div style={topNavContainerStyle}>
                        {/* Use className="nav-button-style" if you add specific styles for nav vs submit */}
                        <Link to="/" style={{ textDecoration: 'none' }}><StyledNavButton className="nav-button-style">Home</StyledNavButton></Link>
                        <Link to="/journal-history" style={{ textDecoration: 'none' }}><StyledNavButton className="nav-button-style">View History</StyledNavButton></Link>
                        <Link to="/journal-report" style={{ textDecoration: 'none' }}><StyledNavButton className="nav-button-style">View Report</StyledNavButton></Link>
                    </div>

                    {/* Main Heading */}
                    <h1 style={mainHeadingStyle}>
                        Welcome to your <span style={journalWordStyle}>journal entry</span>
                    </h1>

                    {/* Subheading */}
                    <p style={subHeadingStyle}>
                        How are you feeling today?
                    </p>

                    {/* User Input Area */}
                    <textarea
                        placeholder="Type your thoughts here..."
                        value={userInput}
                        onChange={handleInputChange}
                        style={textAreaStyle}
                        disabled={isProcessing} // Use isProcessing which is derived from isAnalyzing
                    />

                    {/* Bot Output Area */}
                    <div style={outputBoxStyle}>
                        {/* Display the response state, or a default message */}
                        {response || "Your analyzed emotion and depression level will appear here..."}
                    </div>

                    {/* Submit Button - Use StyledNavButton */}
                    <StyledNavButton
                        onClick={handleSubmit}
                        disabled={isProcessing} // Use isProcessing
                    >
                        {/* Show different text based on isAnalyzing state */}
                        {isAnalyzing ? 'Analyzing...' : 'Submit Entry'}
                    </StyledNavButton>

                </div>
            </AnimatedBackgroundContainer>
        </>
    );
}

export default JournalEntry;
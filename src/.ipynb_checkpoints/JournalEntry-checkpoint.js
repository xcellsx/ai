import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css'; // Import the shared CSS file

function JournalEntry() {
  const [userInput, setUserInput] = useState('');
  // const [isTyping, setIsTyping] = useState(false); // isTyping state is not currently used effectively
  const [response, setResponse] = useState('');
  const navigate = useNavigate(); // Hook to navigate between routes

  // --- Reusing Styles from HomePage ---
  const pageStyle = {
    backgroundImage: 'linear-gradient(to right, #CC99C9, #9EC1CF, #9EE09E, #FDFD97, #FEB144, #FF6663)',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px'
  };

  const contentBoxStyle = {
    backgroundColor: 'rgba(255, 255, 255, 0.2)', // Adjusted opacity back to 70%
    backdropFilter: 'blur(5px)', // Optional blur
    padding: '40px',
    borderRadius: '15px',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)',
    // textAlign: 'center', // Centering handled by flex container now
    maxWidth: '700px',
    width: '90%'
  };
  // --- End of Reused Styles ---

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const handleSubmit = () => {
    if (userInput.trim() !== '') {
      console.log("Journal Entry Submitted:", userInput);
      setResponse('Thank you for sharing your thoughts.');
      setUserInput('');
    } else {
      setResponse('Please write something before submitting.');
    }
     setTimeout(() => {
        setResponse('');
      }, 3000);
  };


  return (
    <div style={pageStyle}>
      <div style={contentBoxStyle}>

        {/* 2. Container for Heading and Back Arrow */}
        <div style={{
            display: 'flex',
            alignItems: 'center', // Vertically center arrow and text
            justifyContent: 'center', // Horizontally center the group
            marginBottom: '30px' // Space below heading group
         }}>
          {/* 3. Clickable Back Arrow */}
          <span
            onClick={() => navigate('/home')} // 4. Navigate on click
            style={{
              cursor: 'pointer', // 5. Indicate clickable
              fontSize: '1.8em', // Make arrow larger
              marginRight: '15px', // Space between arrow and text
              color: '#333', // Match heading color
              lineHeight: '1' // Adjust line height for better vertical alignment
            }}
            title="Back to Home" // Tooltip for accessibility
          >
            ←
          </span>
          {/* Heading Text */}
          <h1 style={{ fontWeight: 'bold', color: '#333', margin: 0 /* Remove default margin */ }}>
              NEW JOURNAL ENTRY
          </h1>
        </div>


        {/* Container for the chat-like interface elements */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center', // Center items like textarea/buttons within the box
            width: '100%'
          }}
        >
          {/* Bot Message */}
          <div
            style={{
              backgroundColor: '#e9e9eb',
              color: '#333',
              padding: '12px 18px',
              borderRadius: '20px',
              marginBottom: '20px',
              maxWidth: '80%',
              alignSelf: 'flex-start',
              textAlign: 'left',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}
          >
            <strong>Bot:</strong> What's on your mind today?
          </div>

          {/* User Input */}
          <textarea
            placeholder="Type your thoughts here..."
            value={userInput}
            onChange={handleInputChange}
            style={{
              width: '95%',
              minHeight: '120px',
              padding: '15px',
              fontSize: '16px',
              borderRadius: '10px',
              border: '1px solid #ccc',
              marginBottom: '15px',
              resize: 'vertical'
            }}
          />

           {/* Bot Response Area (shown after submit) */}
           {response && (
            <div
              style={{
                backgroundColor: '#e9e9eb',
                color: '#333',
                padding: '12px 18px',
                borderRadius: '20px',
                marginTop: '10px',
                marginBottom: '20px',
                maxWidth: '80%',
                alignSelf: 'flex-start',
                textAlign: 'left',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
              }}
            >
              <strong>Bot:</strong> {response}
            </div>
          )}

          {/* Buttons Container */}
           <div style={{width: '100%', textAlign: 'center', marginTop: '10px'}}>
             {/* Submit Button */}
             <button
               onClick={handleSubmit}
               className="home-button"
             >
               Submit Entry
             </button>

             {/* 1. Removed the Back to Home button from here */}
           </div>

        </div>
      </div>
      {/* Simple CSS for blinking dots animation - can be removed if isTyping state isn't used */}
      {/* <style>{`
        @keyframes blink { 50% { opacity: 0; } }
        .dots span { animation: blink 1s infinite; }
        .dots span:nth-child(2) { animation-delay: 0.2s; }
        .dots span:nth-child(3) { animation-delay: 0.4s; }
      `}</style> */}
    </div>
  );
}

export default JournalEntry;
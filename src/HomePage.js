// src/HomePage.js
import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import AnimatedBackgroundContainer from './AnimatedBackgroundContainer';

// Styled component for the navigation buttons
const StyledNavButton = styled.button`
  background-color: transparent;
  color: #000;
  border: 1px solid black;
  padding: 10px 25px;
  margin: 0 5px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  text-decoration: none;
  transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  font-family: inherit;

  &:hover {
    background-color: black;
    color: #FFF;
  }
`;

function HomePage() {
    // --- Styles ---

    // Style for the FIXED top header bar (no changes needed here)
    const fixedHeaderStyle = {
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        backdropFilter: 'blur(8px)',
        padding: '15px 0',
        textAlign: 'center',
        zIndex: 10,
        fontSize: '20px',
        fontWeight: 'bold',
        color: '#333',
    };

    // Style for the specific word "journal" in the fixed header (no changes needed here)
    const headerWordStyle = {
        fontWeight: 'normal',
        fontSize: '1.2em',
        fontFamily: 'serif',
    };

    // Style for the main content area
    const contentAreaStyle = {
         paddingTop: '80px', // Padding to clear the fixed header (adjust if needed)
         paddingBottom: '40px', // Padding at the bottom
         width: '100%',
         display: 'flex',
         flexDirection: 'column',
         alignItems: 'center',      // *** Centers content horizontally ***
         justifyContent: 'center',   // *** Centers content vertically ***
         textAlign: 'center',
         flex: 1,                  // *** Makes this div grow to fill available vertical space ***
                                   // This is crucial for justify-content: center to work correctly
                                   // within the AnimatedBackgroundContainer which has min-height: 100vh
    };

    // Other styles (no changes needed here)
    const mainHeadingStyle = {
        fontSize: '3em',
        fontWeight: 'bold',
        color: '#000000',
        marginBottom: '10px'
    };
    const journalWordStyle = {
        fontWeight: 'normal',
        fontFamily: 'serif',
        marginLeft: '0.2em',
    };
    const subHeadingStyle = {
        fontSize: '1.1em',
        color: '#000',
        marginBottom: '40px'
    };
    const buttonContainerStyle = {
        display: 'flex',
        justifyContent: 'center',
        flexWrap: 'wrap',
        gap: '10px',
    };

    // --- End Styles ---

    return (
        <>
            {/* Fixed Header */}
            <div style={fixedHeaderStyle}>
                emotional <span style={headerWordStyle}>journal</span>
            </div>

            {/* Animated Background and Main Content */}
            <AnimatedBackgroundContainer>
                {/* Content area wrapper applies padding and centering */}
                <div style={contentAreaStyle}>

                    {/* All content below will be centered vertically & horizontally */}
                    {/* within the space below the header */}

                    <h1 style={mainHeadingStyle}>
                        Welcome to your
                        <span style={journalWordStyle}>journal</span>
                    </h1>

                    <p style={subHeadingStyle}>
                        What would you like to do today?
                    </p>

                    <div style={buttonContainerStyle}>
                        <Link to="/journal-entry" style={{ textDecoration: 'none' }}>
                            <StyledNavButton>
                                Journal Entry
                            </StyledNavButton>
                        </Link>
                        <Link to="/journal-history" style={{ textDecoration: 'none' }}>
                           <StyledNavButton>
                                View History
                           </StyledNavButton>
                        </Link>
                        <Link to="/journal-report" style={{ textDecoration: 'none' }}>
                            <StyledNavButton>
                                View Report
                            </StyledNavButton>
                        </Link>
                    </div>
                </div>
            </AnimatedBackgroundContainer>
        </>
    );
}

export default HomePage;
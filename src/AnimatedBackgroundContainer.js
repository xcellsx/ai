// src/AnimatedBackgroundContainer.js
import styled, { keyframes } from 'styled-components';

// 1. Colors from the Gelato Days palette
const colors = [
    '#FFCBE1', // Pink
    '#D6E5BD', // Green
    '#F9E1A8', // Yellow
    '#BCD8EC', // Blue
    '#DCCCEC', // Lavender
    '#FFDAB4', // Peach
];

// 2. Create a seamless gradient string for looping
// We repeat the first color at the end to make the loop smoother visually
const gradientColors = [...colors, colors[0]].join(', ');

// 3. Define the keyframes for the background position animation
const moveGradient = keyframes`
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
`;

// 4. Create the styled div component
const AnimatedBackgroundContainer = styled.div`
  /* Base layout styles (similar to your previous pageStyle) */
  min-height: 100vh;         /* Ensure full viewport height */
  padding: 25px;             /* Consistent padding */
  box-sizing: border-box;    /* Include padding in dimensions */
  display: flex;             /* Use flexbox for centering content */
  flex-direction: column;    /* Stack content vertically */
  align-items: center;       /* Center content horizontally */
  font-family: sans-serif;   /* Default font */
  position: relative;        /* Needed for potential layering if required */
  overflow: hidden;          /* Hide gradient overflow */

  /* Animated Gradient Background */
  background: linear-gradient(-45deg, ${gradientColors}); /* Diagonal gradient */
  background-size: 400% 400%; /* Make gradient much larger than the view */
  animation: ${moveGradient} 25s ease infinite; /* Apply animation */

  /* Ensure content is rendered above the background */
  > * {
    position: relative;
    z-index: 1;
  }
`;

export default AnimatedBackgroundContainer;
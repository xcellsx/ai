// src/App.js
import React from 'react'; // Removed useState as it's no longer needed here
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'; // Removed Link as it's no longer used here
import HomePage from './HomePage';
import JournalEntry from './JournalEntry';
import JournalHistory from './JournalHistory';
import JournalReport from './JournalReport';

function App() {
  // Removed the state and styles related to the old landing page button
  // const [isButtonHovered, setIsButtonHovered] = useState(false);
  // const baseButtonStyle = { ... };
  // const hoverButtonStyle = { ... };

  return (
    <Router>
      <Routes>
        {/* The inline landing page route (path="/") has been removed. */}

        {/* Set HomePage to be the root route */}
        <Route path="/" element={<HomePage />} />

        {/* Keep the other routes as they were */}
        <Route path="/journal-entry" element={<JournalEntry />} />
        <Route path="/journal-history" element={<JournalHistory />} />
        <Route path="/journal-report" element={<JournalReport />} />

        {/* Note: If you previously had links pointing to "/home",
            they will now need to point to "/" to reach the HomePage.
            Alternatively, you could add a redirect from "/home" to "/",
            but making HomePage the root is cleaner if "/home" isn't strictly needed.
        */}
      </Routes>
    </Router>
  );
}

export default App;
import { useState } from 'react';
import { Alert, CircularProgress, TextField } from '@mui/material';
import { Shield } from 'lucide-react';
import axios from 'axios';

function Detect() {
  const [jobText, setJobText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:5000/predict', {
        text: jobText
      });

      setResult(response.data);
    } catch (err) {
      setError('An error occurred while analyzing the job posting. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-8">
        <Shield className="h-12 w-12 text-primary mx-auto mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Fake Job Posting Detector
        </h1>
        <p className="text-gray-600">
          Paste the job posting text below to check if it's legitimate or fraudulent
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <TextField
          multiline
          rows={10}
          fullWidth
          variant="outlined"
          placeholder="Paste the complete job posting text here..."
          value={jobText}
          onChange={(e) => setJobText(e.target.value)}
          disabled={loading}
        />

        <button
          type="submit"
          disabled={loading || !jobText.trim()}
          className="w-full bg-primary text-white py-3 rounded-lg font-semibold disabled:opacity-50"
        >
          {loading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Analyze Job Posting'
          )}
        </button>
      </form>

      {error && (
        <Alert severity="error" className="mt-6">
          {error}
        </Alert>
      )}

      {result && (
        <div className="mt-8 bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-2xl font-bold mb-4">Analysis Results</h2>
          <div className="space-y-4">
            <div className={`p-4 rounded-lg ${
              result.prediction ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
            }`}>
              <p className="text-lg font-semibold">
                {result.prediction ? 'Potential Fake Job Posting' : 'Likely Legitimate Job Posting'}
              </p>
              <p className="mt-2">
                Confidence: {(result.probability * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-gray-600">
              <p className="mb-2">What this means:</p>
              <ul className="list-disc list-inside space-y-1">
                {result.prediction ? (
                  <>
                    <li>This job posting shows characteristics commonly found in fraudulent listings</li>
                    <li>Exercise extreme caution before proceeding</li>
                    <li>Do not share sensitive personal information</li>
                    <li>Research the company thoroughly</li>
                  </>
                ) : (
                  <>
                    <li>This job posting appears to be legitimate</li>
                    <li>However, always conduct your own due diligence</li>
                    <li>Verify the company's existence and reputation</li>
                    <li>Be cautious with personal information</li>
                  </>
                )}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Detect;
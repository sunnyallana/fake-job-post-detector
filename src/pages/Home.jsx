import { Link } from 'react-router-dom';
import { Shield, AlertTriangle, CheckCircle, Search } from 'lucide-react';

function Home() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <Shield className="h-16 w-16 text-primary mx-auto mb-4" />
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Protect Yourself from Fake Job Postings
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
          Our advanced AI-powered system helps you identify fraudulent job postings
          with high accuracy, keeping your job search safe and secure.
        </p>
        <Link
          to="/detect"
          className="bg-primary text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-primary/90 transition-colors"
        >
          Start Detecting
        </Link>
      </div>

      {/* Features Section */}
      <div className="grid md:grid-cols-3 gap-8 mb-16">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <AlertTriangle className="h-12 w-12 text-red-500 mb-4" />
          <h3 className="text-xl font-semibold mb-2">Fraud Detection</h3>
          <p className="text-gray-600">
            Advanced LSTM neural network trained on thousands of job postings to
            identify suspicious patterns.
          </p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <CheckCircle className="h-12 w-12 text-green-500 mb-4" />
          <h3 className="text-xl font-semibold mb-2">High Accuracy</h3>
          <p className="text-gray-600">
            Our model achieves over 95% accuracy in identifying fraudulent job
            postings.
          </p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Search className="h-12 w-12 text-blue-500 mb-4" />
          <h3 className="text-xl font-semibold mb-2">Easy to Use</h3>
          <p className="text-gray-600">
            Simply paste the job posting text and get instant results about its
            authenticity.
          </p>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="text-center mb-16">
        <h2 className="text-3xl font-bold text-gray-900 mb-8">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="text-2xl font-bold text-primary mb-4">1</div>
            <h3 className="text-lg font-semibold mb-2">Paste Job Description</h3>
            <p className="text-gray-600">
              Copy and paste the complete job posting text into our system.
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="text-2xl font-bold text-primary mb-4">2</div>
            <h3 className="text-lg font-semibold mb-2">AI Analysis</h3>
            <p className="text-gray-600">
              Our AI model analyzes the text for suspicious patterns and indicators.
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="text-2xl font-bold text-primary mb-4">3</div>
            <h3 className="text-lg font-semibold mb-2">Get Results</h3>
            <p className="text-gray-600">
              Receive instant feedback about the legitimacy of the job posting.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
import { Link } from 'react-router-dom';
import { Shield } from 'lucide-react';

function Navbar() {
  return (
    <nav className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <Shield className="h-8 w-8 text-primary" />
              <span className="ml-2 text-xl font-bold text-gray-800">FakeJobGuard</span>
            </Link>
          </div>
          <div className="flex items-center space-x-4">
            <Link to="/" className="text-gray-600 hover:text-primary px-3 py-2 rounded-md">
              Home
            </Link>
            <Link to="/detect" className="bg-primary text-white px-4 py-2 rounded-md hover:bg-primary/90">
              Detect Fake Jobs
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
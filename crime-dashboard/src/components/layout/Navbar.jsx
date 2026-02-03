import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-blue-600">
              Crime Prediction Dashboard
            </h1>
          </div>
          
          <div className="flex items-center space-x-2">
            <Link 
              to="/" 
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                isActive('/') 
                  ? 'bg-blue-100 text-blue-700' 
                  : 'text-gray-700 hover:text-blue-600 hover:bg-blue-50'
              }`}
            >
              Overview
            </Link>
            <Link 
              to="/models" 
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                isActive('/models') 
                  ? 'bg-blue-100 text-blue-700' 
                  : 'text-gray-700 hover:text-blue-600 hover:bg-blue-50'
              }`}
            >
              Models
            </Link>
            <Link 
              to="/fairness" 
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                isActive('/fairness') 
                  ? 'bg-blue-100 text-blue-700' 
                  : 'text-gray-700 hover:text-blue-600 hover:bg-blue-50'
              }`}
            >
              Fairness
            </Link>
            <Link 
              to="/geographic" 
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                isActive('/geographic') 
                  ? 'bg-blue-100 text-blue-700' 
                  : 'text-gray-700 hover:text-blue-600 hover:bg-blue-50'
              }`}
            >
              Geographic
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const PredictionTypeSelector = ({ selectedTypes, onSelectionChange }) => {
  const predictionTypes = [
    { id: 'bioactivity_ic50', label: 'IC₅₀ / Bioactivity', category: 'Bioactivity', description: 'Half-maximal inhibitory concentration' },
    { id: 'toxicity', label: 'General Toxicity', category: 'Toxicity', description: 'Overall toxicity probability' },
    { id: 'logP', label: 'LogP', category: 'Physicochemical', description: 'Partition coefficient (lipophilicity)' },
    { id: 'solubility', label: 'Solubility (LogS)', category: 'Physicochemical', description: 'Aqueous solubility' }
  ];

  const handleTypeToggle = (typeId) => {
    const newSelection = selectedTypes.includes(typeId)
      ? selectedTypes.filter(id => id !== typeId)
      : [...selectedTypes, typeId];
    onSelectionChange(newSelection);
  };

  const categories = [...new Set(predictionTypes.map(type => type.category))];

  return (
    <div className="prediction-selector">
      <h3 className="text-lg font-semibold mb-4">Select Prediction Types</h3>
      {categories.map(category => (
        <div key={category} className="mb-4">
          <h4 className="text-md font-medium text-blue-700 mb-2">{category}</h4>
          <div className="grid grid-cols-1 gap-2">
            {predictionTypes.filter(type => type.category === category).map(type => (
              <label key={type.id} className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedTypes.includes(type.id)}
                  onChange={() => handleTypeToggle(type.id)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <div>
                  <div className="font-medium">{type.label}</div>
                  <div className="text-sm text-gray-600">{type.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

const ResultsDisplay = ({ results, isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3">Analyzing molecular properties...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <div className="text-red-700">
            <h3 className="font-medium">Prediction Error</h3>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return null;
  }

  return (
    <div className="results-section">
      <h3 className="text-xl font-bold mb-6">Prediction Results</h3>
      
      {/* Molecular Summary */}
      {results.summary && (
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <h4 className="font-semibold mb-2">Molecular Information</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>SMILES: <span className="font-mono text-blue-600">{results.summary.molecule}</span></div>
            <div>Target: <span className="font-semibold text-purple-600">{results.summary.target}</span></div>
            <div>Predictions: {results.summary.total_predictions}</div>
            <div>Enhanced Models: {results.summary.enhanced_models_used ? "✅ Yes" : "❌ No"}</div>
            {results.summary.molecular_properties && (
              <>
                <div>Molecular Weight: {results.summary.molecular_properties.molecular_weight?.toFixed(2)} g/mol</div>
                <div>QED: {results.summary.molecular_properties.qed?.toFixed(3)}</div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Predictions Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Property
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                ChemBERTa
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Chemprop
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Enhanced Model
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                RDKit
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {results.results.map((result, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {getPropertyDisplayName(result.prediction_type)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {result.molbert_prediction ? formatPredictionValue(result.molbert_prediction, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {result.chemprop_prediction ? formatPredictionValue(result.chemprop_prediction, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-blue-600">
                  {result.enhanced_chemprop_prediction ? formatEnhancedPrediction(result.enhanced_chemprop_prediction, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {result.rdkit_value ? formatPredictionValue(result.rdkit_value, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex flex-col">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      result.confidence > 0.8 ? 'bg-green-100 text-green-800' : 
                      result.confidence > 0.6 ? 'bg-yellow-100 text-yellow-800' : 
                      'bg-red-100 text-red-800'
                    }`}>
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                    {result.similarity && (
                      <span className="text-xs text-gray-500 mt-1">
                        Sim: {(result.similarity * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Enhanced Predictions Details */}
      {results.results.some(r => r.enhanced_chemprop_prediction) && (
        <div className="mt-6 bg-blue-50 rounded-lg p-4">
          <h4 className="font-semibold text-blue-900 mb-2">Enhanced IC₅₀ Prediction Details</h4>
          {results.results.filter(r => r.enhanced_chemprop_prediction).map((result, index) => (
            <div key={index} className="text-sm text-blue-800">
              {result.enhanced_chemprop_prediction.pic50 && (
                <div className="grid grid-cols-2 gap-4">
                  <div>pIC₅₀: <span className="font-mono">{result.enhanced_chemprop_prediction.pic50.toFixed(2)}</span></div>
                  <div>IC₅₀: <span className="font-mono">{result.enhanced_chemprop_prediction.ic50_nm?.toFixed(1)} nM</span></div>
                  <div>Model Type: {result.enhanced_chemprop_prediction.model_type}</div>
                  <div>Target-Specific: {result.enhanced_chemprop_prediction.target_specific ? "✅ Yes" : "❌ No"}</div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const getPropertyDisplayName = (type) => {
  const names = {
    'bioactivity_ic50': 'IC₅₀ (µM)',
    'toxicity': 'Toxicity Probability',
    'logP': 'LogP',
    'solubility': 'LogS'
  };
  return names[type] || type;
};

const formatPredictionValue = (value, type) => {
  if (typeof value !== 'number') return 'N/A';
  
  switch (type) {
    case 'bioactivity_ic50':
      return `${value.toFixed(3)} µM`;
    case 'toxicity':
      return `${(value * 100).toFixed(1)}%`;
    case 'logP':
    case 'solubility':
      return value.toFixed(2);
    default:
      return value.toFixed(3);
  }
};

const App = () => {
  const [smiles, setSmiles] = useState('');
  const [selectedTypes, setSelectedTypes] = useState(['bioactivity_ic50', 'toxicity']);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);

  // Example SMILES for testing
  const exampleMolecules = [
    { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
    { name: 'Ethanol', smiles: 'CCO' },
    { name: 'Benzene', smiles: 'C1=CC=CC=C1' }
  ];

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API}/health`);
      setHealth(response.data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const handlePredict = async () => {
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }

    if (selectedTypes.length === 0) {
      setError('Please select at least one prediction type');
      return;
    }

    setIsLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await axios.post(`${API}/predict`, {
        smiles: smiles.trim(),
        prediction_types: selectedTypes
      });
      
      setResults(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error.response?.data?.detail || 'Failed to get predictions. Please check your SMILES string.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (exampleSmiles) => {
    setSmiles(exampleSmiles);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">V</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Veridica AI</h1>
              <p className="text-gray-600">Predictive Chemistry Platform</p>
            </div>
            {health && (
              <div className="ml-auto">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  ● {health.status}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Input Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Molecular Input</h2>
              
              {/* SMILES Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  SMILES String
                </label>
                <textarea
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  placeholder="Enter SMILES string (e.g., CCO for ethanol)"
                  className="w-full h-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Example Molecules */}
              <div className="mb-6">
                <p className="text-sm font-medium text-gray-700 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {exampleMolecules.map((mol, index) => (
                    <button
                      key={index}
                      onClick={() => handleExampleClick(mol.smiles)}
                      className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors"
                    >
                      {mol.name}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={handlePredict}
                disabled={isLoading || !smiles.trim() || selectedTypes.length === 0}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isLoading ? 'Analyzing...' : 'Predict Properties'}
              </button>
            </div>
          </div>

          {/* Prediction Type Selector */}
          <div className="bg-white rounded-lg shadow p-6">
            <PredictionTypeSelector
              selectedTypes={selectedTypes}
              onSelectionChange={setSelectedTypes}
            />
          </div>
        </div>

        {/* Results Section */}
        <div className="mt-8">
          <div className="bg-white rounded-lg shadow p-6">
            <ResultsDisplay 
              results={results} 
              isLoading={isLoading} 
              error={error}
            />
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">About This Platform</h3>
          <p className="text-blue-800 text-sm leading-relaxed">
            This platform combines MolBERT (transformer-based) and Chemprop (graph neural network) models 
            to predict molecular properties for drug discovery. Results are compared with RDKit calculations 
            for validation. Priority predictions include bioactivity (IC₅₀), toxicity assessment, and 
            physicochemical properties (LogP, solubility).
          </p>
        </div>
      </main>
    </div>
  );
};

export default App;
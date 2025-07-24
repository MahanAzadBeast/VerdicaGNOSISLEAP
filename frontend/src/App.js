import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Navigation Component
const Navigation = ({ activeTab, setActiveTab, health }) => {
  const tabs = [
    { id: 'home', label: 'Home' },
    { id: 'predict', label: 'Predict Properties' },
    { id: 'analysis', label: 'Result Analysis' },
    { id: 'about', label: 'About' }
  ];

  return (
    <nav className="bg-gray-900 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">V</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Veridica AI</h1>
              <p className="text-gray-400 text-xs">Predictive Chemistry Platform</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex space-x-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? 'bg-purple-600 text-white shadow-lg'
                    : 'text-gray-300 hover:text-white hover:bg-gray-800'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Status */}
          {health && (
            <div className="flex items-center space-x-2">
              <div className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-900 text-green-300 border border-green-700">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
                {health.status}
              </div>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

import Spline from '@splinetool/react-spline';

// Home Tab Component
const HomeTab = ({ setActiveTab }) => {
  const handleSplineClick = () => {
    setActiveTab('predict');
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Spline Banner Section - Increased height and adjusted positioning to fix cropping */}
      <div className="relative w-full h-[850px] overflow-hidden cursor-pointer flex items-center justify-center" onClick={handleSplineClick}>
        <main className="w-full h-full flex items-center justify-center">
          <div 
            style={{
              width: '125%',
              height: '125%',
              transform: 'scale(0.92) translateY(-15%)',
              transformOrigin: 'center center'
            }}
            className="flex items-center justify-center"
          >
            <Spline
              scene="https://prod.spline.design/RnIHjsPRp09RPfVl/scene.splinecode"
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                objectPosition: 'center center'
              }}
            />
          </div>
        </main>
        {/* Clickable overlay */}
        <div 
          className="absolute inset-0 z-10 cursor-pointer"
          onClick={handleSplineClick}
          title="Click to start prediction"
        />
        
        {/* Overlayed Feature Boxes - positioned over the Spline animation */}
        <div className="absolute bottom-24 left-1/2 transform -translate-x-1/2 w-full max-w-6xl px-4 z-20">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-gray-800/90 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-purple-500 transition-all shadow-xl">
              <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center mb-4">
                <div className="w-6 h-6 bg-purple-400 rounded"></div>
              </div>
              <h3 className="text-xl font-semibold mb-3">Target-Specific IC50</h3>
              <p className="text-gray-400">
                Predict IC50 values for specific protein targets including EGFR, BRAF, CDK2, and more.
              </p>
            </div>
            <div className="bg-gray-800/90 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-cyan-500 transition-all shadow-xl">
              <div className="w-12 h-12 bg-cyan-600/20 rounded-lg flex items-center justify-center mb-4">
                <div className="w-6 h-6 bg-cyan-400 rounded"></div>
              </div>
              <h3 className="text-xl font-semibold mb-3">Multi-Model Analysis</h3>
              <p className="text-gray-400">
                Compare predictions from ChemBERTa, Chemprop, and enhanced RDKit-based models.
              </p>
            </div>
            <div className="bg-gray-800/90 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-green-500 transition-all shadow-xl">
              <div className="w-12 h-12 bg-green-600/20 rounded-lg flex items-center justify-center mb-4">
                <div className="w-6 h-6 bg-green-400 rounded"></div>
              </div>
              <h3 className="text-xl font-semibold mb-3">Advanced Analytics</h3>
              <p className="text-gray-400">
                Visualize results, compare compounds, and export data with our comprehensive analysis tools.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section - Now placed outside the Spline container */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-8 border border-gray-600">
          <div className="text-center">
            <div className="grid md:grid-cols-4 gap-8">
              <div>
                <div className="text-3xl font-bold text-purple-400">6</div>
                <div className="text-gray-400">Protein Targets</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-cyan-400">4</div>
                <div className="text-gray-400">Prediction Types</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-green-400">3</div>
                <div className="text-gray-400">AI Models</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-yellow-400">∞</div>
                <div className="text-gray-400">Molecules</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Prediction Type Selector Component
const PredictionTypeSelector = ({ selectedTypes, onSelectionChange }) => {
  const predictionTypes = [
    { id: 'bioactivity_ic50', label: 'IC50 / Bioactivity', category: 'Bioactivity', description: 'Half-maximal inhibitory concentration' },
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
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-4 text-white">Select Prediction Types</h3>
      {categories.map(category => (
        <div key={category} className="mb-4">
          <h4 className="text-md font-medium text-purple-400 mb-2">{category}</h4>
          <div className="grid grid-cols-1 gap-2">
            {predictionTypes.filter(type => type.category === category).map(type => (
              <label key={type.id} className="flex items-center space-x-3 p-3 border border-gray-600 rounded-lg hover:bg-gray-700 cursor-pointer transition-all">
                <input
                  type="checkbox"
                  checked={selectedTypes.includes(type.id)}
                  onChange={() => handleTypeToggle(type.id)}
                  className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                />
                <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
                <div>
                  <div className="font-medium text-white">{type.label}</div>
                  <div className="text-sm text-gray-400">{type.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

// Results Display Component
const ResultsDisplay = ({ results, isLoading, error, onAnalyze }) => {
  if (isLoading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-8">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
          <span className="ml-3 text-white">Analyzing molecular properties...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900 border border-red-700 rounded-xl p-4">
        <div className="flex">
          <div className="text-red-300">
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

  const formatEnhancedPrediction = (enhanced, type) => {
    if (!enhanced || enhanced.error) return 'Error';
    
    switch (type) {
      case 'bioactivity_ic50':
        if (enhanced.pic50 && enhanced.ic50_nm) {
          return (
            <div className="text-xs">
              <div>pIC₅₀: {enhanced.pic50.toFixed(2)}</div>
              <div>{enhanced.ic50_nm.toFixed(1)} nM</div>
            </div>
          );
        }
        return 'N/A';
      default:
        return 'N/A';
    }
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

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-xl font-bold text-white">Prediction Results</h3>
        {results.results && results.results.length > 0 && (
          <button
            onClick={() => onAnalyze(results)}
            className="bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center space-x-2"
          >
            <span>Analyze Results</span>
          </button>
        )}
      </div>
      
      {/* Molecular Summary */}
      {results.summary && (
        <div className="bg-gray-700 border border-gray-600 rounded-lg p-4 mb-6">
          <h4 className="font-semibold mb-2 text-white">Molecular Information</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="text-gray-300">SMILES: <span className="font-mono text-purple-400">{results.summary.molecule}</span></div>
            <div className="text-gray-300">Target: <span className="font-semibold text-cyan-400">{results.summary.target}</span></div>
            <div className="text-gray-300">Predictions: <span className="text-white">{results.summary.total_predictions}</span></div>
            <div className="text-gray-300">Enhanced Models: {results.summary.enhanced_models_used ? <span className="text-green-400">✅ Yes</span> : <span className="text-red-400">❌ No</span>}</div>
            {results.summary.molecular_properties && (
              <>
                <div className="text-gray-300">Molecular Weight: <span className="text-white">{results.summary.molecular_properties.molecular_weight?.toFixed(2)} g/mol</span></div>
                <div className="text-gray-300">QED: <span className="text-white">{results.summary.molecular_properties.qed?.toFixed(3)}</span></div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Predictions Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-600">
          <thead className="bg-gray-700">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Property</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">ChemBERTa</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Chemprop</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Enhanced Model</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">RDKit</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Confidence</th>
            </tr>
          </thead>
          <tbody className="bg-gray-800 divide-y divide-gray-600">
            {results.results.map((result, index) => (
              <tr key={index} className="hover:bg-gray-700">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                  {getPropertyDisplayName(result.prediction_type)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                  {result.molbert_prediction ? formatPredictionValue(result.molbert_prediction, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                  {result.chemprop_prediction ? formatPredictionValue(result.chemprop_prediction, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-purple-400">
                  {result.enhanced_chemprop_prediction ? formatEnhancedPrediction(result.enhanced_chemprop_prediction, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                  {result.rdkit_value ? formatPredictionValue(result.rdkit_value, result.prediction_type) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex flex-col">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      result.confidence > 0.8 ? 'bg-green-900 text-green-300 border border-green-700' : 
                      result.confidence > 0.6 ? 'bg-yellow-900 text-yellow-300 border border-yellow-700' : 
                      'bg-red-900 text-red-300 border border-red-700'
                    }`}>
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                    {result.similarity && (
                      <span className="text-xs text-gray-400 mt-1">
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
        <div className="mt-6 bg-purple-900 border border-purple-700 rounded-lg p-4">
          <h4 className="font-semibold text-purple-300 mb-2">Enhanced IC₅₀ Prediction Details</h4>
          {results.results.filter(r => r.enhanced_chemprop_prediction).map((result, index) => (
            <div key={index} className="text-sm text-purple-200">
              {result.enhanced_chemprop_prediction.pic50 && (
                <div className="grid grid-cols-2 gap-4">
                  <div>pIC₅₀: <span className="font-mono text-white">{result.enhanced_chemprop_prediction.pic50.toFixed(2)}</span></div>
                  <div>IC₅₀: <span className="font-mono text-white">{result.enhanced_chemprop_prediction.ic50_nm?.toFixed(1)} nM</span></div>
                  <div>Model Type: <span className="text-white">{result.enhanced_chemprop_prediction.model_type}</span></div>
                  <div>Target-Specific: {result.enhanced_chemprop_prediction.target_specific ? <span className="text-green-400">✅ Yes</span> : <span className="text-red-400">❌ No</span>}</div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Predict Tab Component  
const PredictTab = ({ onAnalyze }) => {
  const [smiles, setSmiles] = useState('');
  const [selectedTypes, setSelectedTypes] = useState(['bioactivity_ic50', 'toxicity']);
  const [selectedTarget, setSelectedTarget] = useState('EGFR');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Available targets
  const availableTargets = [
    { id: 'EGFR', name: 'EGFR', description: 'Epidermal Growth Factor Receptor' },
    { id: 'BRAF', name: 'BRAF', description: 'B-Raf Proto-Oncogene' },
    { id: 'CDK2', name: 'CDK2', description: 'Cyclin Dependent Kinase 2' },
    { id: 'PARP1', name: 'PARP1', description: 'Poly(ADP-ribose) Polymerase 1' },
    { id: 'BCL2', name: 'BCL2', description: 'BCL2 Apoptosis Regulator' },
    { id: 'VEGFR2', name: 'VEGFR2', description: 'Vascular Endothelial Growth Factor Receptor 2' }
  ];

  // Example SMILES for testing
  const exampleMolecules = [
    { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
    { name: 'Ethanol', smiles: 'CCO' },
    { name: 'Imatinib', smiles: 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5' }
  ];

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
        prediction_types: selectedTypes,
        target: selectedTarget
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
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Input Section */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4 text-white">Molecular Input</h2>
              
              {/* SMILES Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  SMILES String
                </label>
                <textarea
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  placeholder="Enter SMILES string (e.g., CCO for ethanol)"
                  className="w-full h-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-400"
                />
              </div>

              {/* Target Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Target Protein (for IC₅₀ predictions)
                </label>
                <select
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white"
                >
                  {availableTargets.map((target) => (
                    <option key={target.id} value={target.id}>
                      {target.name} - {target.description}
                    </option>
                  ))}
                </select>
              </div>

              {/* Example Molecules */}
              <div className="mb-6">
                <p className="text-sm font-medium text-gray-300 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {exampleMolecules.map((mol, index) => (
                    <button
                      key={index}
                      onClick={() => handleExampleClick(mol.smiles)}
                      className="px-3 py-1 text-sm bg-gray-700 text-gray-200 border border-gray-600 rounded-lg hover:bg-gray-600 hover:border-purple-500 transition-all"
                    >
                      {mol.name}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={handlePredict}
                disabled={isLoading || !smiles.trim() || selectedTypes.length === 0}
                className="w-full bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700 text-white py-3 px-6 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </div>
                ) : (
                  'Predict Properties'
                )}
              </button>
            </div>
          </div>

          {/* Prediction Type Selector */}
          <div>
            <PredictionTypeSelector
              selectedTypes={selectedTypes}
              onSelectionChange={setSelectedTypes}
            />
          </div>
        </div>

        {/* Results Section */}
        <div className="mt-8">
          <ResultsDisplay 
            results={results} 
            isLoading={isLoading} 
            error={error}
            onAnalyze={onAnalyze}
          />
        </div>
      </div>
    </div>
  );
};

// Analysis Tab Component (Simplified)
const AnalysisTab = ({ analysisData }) => {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            Result Analysis
          </h2>
          <p className="text-gray-400">
            Visualize, compare, and interpret molecular property predictions
          </p>
        </div>

        {analysisData ? (
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">Analysis Data Available</h3>
            <div className="text-sm text-gray-300 space-y-2">
              <div>Molecule: <span className="font-mono text-purple-400">{analysisData.summary?.molecule}</span></div>
              <div>Target: <span className="text-cyan-400">{analysisData.summary?.target}</span></div>
              <div>Predictions: <span className="text-white">{analysisData.summary?.total_predictions}</span></div>
            </div>

            {/* Simple Results Table */}
            {analysisData.results && (
              <div className="mt-6 overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-600">
                  <thead className="bg-gray-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Property</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Enhanced Prediction</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="bg-gray-800 divide-y divide-gray-600">
                    {analysisData.results.map((result, index) => (
                      <tr key={index} className="hover:bg-gray-700">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                          {result.prediction_type}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-purple-400">
                          {result.enhanced_chemprop_prediction?.pic50 ? 
                            `pIC₅₀: ${result.enhanced_chemprop_prediction.pic50.toFixed(2)}` : 
                            'N/A'
                          }
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                          {(result.confidence * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-12 text-center">
            <div className="w-16 h-16 bg-gray-700 rounded-lg mx-auto mb-4"></div>
            <h3 className="text-xl font-semibold text-gray-400 mb-2">No Analysis Data</h3>
            <p className="text-gray-500 mb-6">
              Run predictions first to analyze molecular data
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// About Tab Component
const AboutTab = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <div className="w-10 h-10 bg-white rounded-lg"></div>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent mb-4">
            About Veridica AI
          </h1>
          <p className="text-xl text-gray-300">
            Advanced molecular property prediction platform for drug discovery
          </p>
        </div>

        <div className="space-y-8">
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4 text-white">Platform Capabilities</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-purple-400 mb-2">AI Models</h3>
                <ul className="text-gray-300 space-y-1">
                  <li>• ChemBERTa (Transformer-based)</li>
                  <li>• Enhanced Chemprop (Graph Neural Network)</li>
                  <li>• Target-specific algorithms</li>
                  <li>• RDKit molecular descriptors</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-cyan-400 mb-2">Predictions</h3>
                <ul className="text-gray-300 space-y-1">
                  <li>• IC50 bioactivity (6 targets)</li>
                  <li>• Toxicity assessment</li>
                  <li>• Physicochemical properties</li>
                  <li>• Confidence scoring & similarity</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4 text-white">Supported Targets</h2>
            <div className="grid md:grid-cols-3 gap-4">
              {[
                { name: 'EGFR', desc: 'Epidermal Growth Factor Receptor' },
                { name: 'BRAF', desc: 'B-Raf Proto-Oncogene' },
                { name: 'CDK2', desc: 'Cyclin Dependent Kinase 2' },
                { name: 'PARP1', desc: 'Poly(ADP-ribose) Polymerase 1' },
                { name: 'BCL2', desc: 'BCL2 Apoptosis Regulator' },
                { name: 'VEGFR2', desc: 'Vascular Endothelial Growth Factor Receptor 2' }
              ].map((target, index) => (
                <div key={index} className="bg-gray-700 border border-gray-600 rounded-lg p-4">
                  <div className="font-semibold text-white">{target.name}</div>
                  <div className="text-sm text-gray-400">{target.desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-900 to-cyan-900 border border-gray-600 rounded-xl p-6 text-center">
            <h2 className="text-2xl font-semibold mb-4 text-white">Ready to Discover?</h2>
            <p className="text-gray-300 mb-6">
              Start predicting molecular properties and accelerate your drug discovery research.
            </p>
            <div className="text-sm text-gray-400">
              Powered by advanced AI and molecular science
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [health, setHealth] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);

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

  const handleAnalyze = (results) => {
    setAnalysisData(results);
    setActiveTab('analysis');
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} health={health} />
      
      {activeTab === 'home' && <HomeTab setActiveTab={setActiveTab} />}
      {activeTab === 'predict' && <PredictTab onAnalyze={handleAnalyze} />}
      {activeTab === 'analysis' && <AnalysisTab analysisData={analysisData} />}
      {activeTab === 'about' && <AboutTab />}
    </div>
  );
};

export default App;
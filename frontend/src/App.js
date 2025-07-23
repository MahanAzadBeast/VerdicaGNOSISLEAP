import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";
import { 
  Beaker, 
  BarChart3, 
  Home, 
  Info, 
  ChevronDown,
  Download,
  Upload,
  Zap,
  Target,
  Activity,
  Atom,
  TrendingUp
} from "lucide-react";
import Plot from 'react-plotly.js';
import { CSVLink } from "react-csv";
import html2canvas from 'html2canvas';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Navigation Component
const Navigation = ({ activeTab, setActiveTab, health }) => {
  const tabs = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'predict', label: 'Predict Properties', icon: Beaker },
    { id: 'analysis', label: 'Result Analysis', icon: BarChart3 },
    { id: 'about', label: 'About', icon: Info }
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
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === tab.id
                      ? 'bg-purple-600 text-white shadow-lg'
                      : 'text-gray-300 hover:text-white hover:bg-gray-800'
                  }`}
                >
                  <Icon size={16} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
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

// Home Tab Component
const HomeTab = ({ setActiveTab }) => {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-6xl mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <Atom size={40} className="text-white" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent mb-6">
            Advanced Molecular Prediction
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Harness the power of AI to predict molecular properties, bioactivity, and toxicity 
            with unprecedented accuracy using state-of-the-art ChemBERTa and Chemprop models.
          </p>
          <button
            onClick={() => setActiveTab('predict')}
            className="bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700 text-white px-8 py-4 rounded-xl text-lg font-semibold transition-all shadow-2xl hover:shadow-purple-500/25"
          >
            Start Predicting
          </button>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:border-purple-500 transition-all">
            <Target className="text-purple-400 mb-4" size={32} />
            <h3 className="text-xl font-semibold mb-3">Target-Specific IC₅₀</h3>
            <p className="text-gray-400">
              Predict IC₅₀ values for specific protein targets including EGFR, BRAF, CDK2, and more.
            </p>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:border-cyan-500 transition-all">
            <Activity className="text-cyan-400 mb-4" size={32} />
            <h3 className="text-xl font-semibold mb-3">Multi-Model Analysis</h3>
            <p className="text-gray-400">
              Compare predictions from ChemBERTa, Chemprop, and enhanced RDKit-based models.
            </p>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:border-green-500 transition-all">
            <TrendingUp className="text-green-400 mb-4" size={32} />
            <h3 className="text-xl font-semibold mb-3">Advanced Analytics</h3>
            <p className="text-gray-400">
              Visualize results, compare compounds, and export data with our comprehensive analysis tools.
            </p>
          </div>
        </div>

        {/* Stats */}
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
    { id: 'bioactivity_ic50', label: 'IC₅₀ / Bioactivity', category: 'Bioactivity', description: 'Half-maximal inhibitory concentration', icon: Target },
    { id: 'toxicity', label: 'General Toxicity', category: 'Toxicity', description: 'Overall toxicity probability', icon: Zap },
    { id: 'logP', label: 'LogP', category: 'Physicochemical', description: 'Partition coefficient (lipophilicity)', icon: Activity },
    { id: 'solubility', label: 'Solubility (LogS)', category: 'Physicochemical', description: 'Aqueous solubility', icon: Activity }
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
            {predictionTypes.filter(type => type.category === category).map(type => {
              const Icon = type.icon;
              return (
                <label key={type.id} className="flex items-center space-x-3 p-3 border border-gray-600 rounded-lg hover:bg-gray-700 cursor-pointer transition-all">
                  <input
                    type="checkbox"
                    checked={selectedTypes.includes(type.id)}
                    onChange={() => handleTypeToggle(type.id)}
                    className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                  />
                  <Icon size={16} className="text-gray-400" />
                  <div>
                    <div className="font-medium text-white">{type.label}</div>
                    <div className="text-sm text-gray-400">{type.description}</div>
                  </div>
                </label>
              );
            })}
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
            <BarChart3 size={16} />
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
                <div className="relative">
                  <select
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white appearance-none"
                  >
                    {availableTargets.map((target) => (
                      <option key={target.id} value={target.id}>
                        {target.name} - {target.description}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                </div>
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

// Analysis Tab Component
const AnalysisTab = ({ analysisData, setAnalysisData }) => {
  const [uploadedData, setUploadedData] = useState(null);
  const [selectedChart, setSelectedChart] = useState('bar');
  const [sortBy, setSortBy] = useState('ic50');
  const [filterTarget, setFilterTarget] = useState('all');

  // CSV upload handler
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const csv = e.target.result;
          const lines = csv.split('\n');
          const headers = lines[0].split(',').map(h => h.trim());
          const data = lines.slice(1).filter(line => line.trim()).map(line => {
            const values = line.split(',').map(v => v.trim());
            const obj = {};
            headers.forEach((header, index) => {
              obj[header] = values[index];
            });
            return obj;
          });
          setUploadedData(data);
        } catch (error) {
          console.error('Error parsing CSV:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  // Get data for visualization
  const getVisualizationData = () => {
    if (uploadedData) return uploadedData;
    if (analysisData && analysisData.results) {
      return analysisData.results.map(result => ({
        'Drug Name': `Compound-${result.id.slice(0, 8)}`,
        'SMILES': result.smiles,
        'IC50 (nM)': result.enhanced_chemprop_prediction?.ic50_nm || 0,
        'pIC50': result.enhanced_chemprop_prediction?.pic50 || 0,
        'Target': result.target || 'Unknown',
        'Confidence': result.confidence,
        'Similarity': result.similarity || 0,
        'LogP': analysisData.summary?.molecular_properties?.logP || 0,
        'MW': analysisData.summary?.molecular_properties?.molecular_weight || 0
      }));
    }
    return [];
  };

  const data = getVisualizationData();
  const filteredData = filterTarget === 'all' ? data : data.filter(d => d.Target === filterTarget);
  
  // Sort data
  const sortedData = [...filteredData].sort((a, b) => {
    switch (sortBy) {
      case 'ic50':
        return (parseFloat(a['IC50 (nM)']) || 0) - (parseFloat(b['IC50 (nM)']) || 0);
      case 'confidence':
        return (parseFloat(b.Confidence) || 0) - (parseFloat(a.Confidence) || 0);
      case 'mw':
        return (parseFloat(a.MW) || 0) - (parseFloat(b.MW) || 0);
      default:
        return 0;
    }
  });

  // Get unique targets for filter
  const targets = [...new Set(data.map(d => d.Target))];

  // Chart configurations
  const getBarChartData = () => ({
    data: [{
      x: sortedData.map(d => d['Drug Name'] || 'Unknown'),
      y: sortedData.map(d => parseFloat(d['IC50 (nM)']) || 0),
      type: 'bar',
      marker: {
        color: sortedData.map((_, i) => `hsl(${(i * 360) / sortedData.length}, 70%, 60%)`),
        line: { color: 'rgba(255,255,255,0.2)', width: 1 }
      },
      name: 'IC₅₀ (nM)'
    }],
    layout: {
      title: { text: 'IC₅₀ Values by Compound', font: { color: 'white' } },
      xaxis: { title: 'Compounds', color: 'white', gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: { title: 'IC₅₀ (nM)', color: 'white', gridcolor: 'rgba(255,255,255,0.1)' },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: 'white' }
    },
    config: { displayModeBar: true, displaylogo: false }
  });

  const getScatterPlotData = () => ({
    data: [{
      x: sortedData.map(d => parseFloat(d.MW) || 0),
      y: sortedData.map(d => parseFloat(d['IC50 (nM)']) || 0),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: sortedData.map(d => (parseFloat(d.Confidence) || 0.5) * 30 + 5),
        color: sortedData.map(d => parseFloat(d.LogP) || 0),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { title: 'LogP' },
        line: { color: 'white', width: 1 }
      },
      text: sortedData.map(d => `${d['Drug Name']}<br>MW: ${d.MW}<br>LogP: ${d.LogP}<br>Confidence: ${(parseFloat(d.Confidence) * 100).toFixed(0)}%`),
      name: 'Compounds'
    }],
    layout: {
      title: { text: 'IC₅₀ vs Molecular Weight', font: { color: 'white' } },
      xaxis: { title: 'Molecular Weight (g/mol)', color: 'white', gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: { title: 'IC₅₀ (nM)', color: 'white', gridcolor: 'rgba(255,255,255,0.1)' },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: 'white' }
    },
    config: { displayModeBar: true, displaylogo: false }
  });

  const getHeatmapData = () => {
    const targets = [...new Set(sortedData.map(d => d.Target))];
    const compounds = sortedData.map(d => d['Drug Name']);
    const z = targets.map(target => 
      compounds.map(compound => {
        const item = sortedData.find(d => d['Drug Name'] === compound && d.Target === target);
        return item ? parseFloat(item['IC50 (nM)']) || 0 : 0;
      })
    );

    return {
      data: [{
        z: z,
        x: compounds,
        y: targets,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true
      }],
      layout: {
        title: { text: 'IC₅₀ Heatmap by Target and Compound', font: { color: 'white' } },
        xaxis: { title: 'Compounds', color: 'white' },
        yaxis: { title: 'Targets', color: 'white' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: 'white' }
      },
      config: { displayModeBar: true, displaylogo: false }
    }
  };

  // Export functions
  const exportChart = () => {
    const chartElement = document.querySelector('.plotly');
    if (chartElement) {
      html2canvas(chartElement).then(canvas => {
        const link = document.createElement('a');
        link.download = `chart-${selectedChart}-${new Date().toISOString().slice(0, 10)}.png`;
        link.href = canvas.toDataURL();
        link.click();
      });
    }
  };

  const csvData = sortedData.map(d => ({
    'Drug Name': d['Drug Name'],
    'SMILES': d.SMILES,
    'IC50 (nM)': d['IC50 (nM)'],
    'pIC50': d.pIC50,
    'Target': d.Target,
    'Confidence': d.Confidence,
    'Similarity': d.Similarity,
    'LogP': d.LogP,
    'Molecular Weight': d.MW
  }));

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

        {/* Data Input Options */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Upload className="mr-2" size={20} />
              Upload CSV Data
            </h3>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-600 file:text-white hover:file:bg-purple-700"
            />
            <p className="text-xs text-gray-400 mt-2">
              Upload CSV with columns: Drug Name, SMILES, IC50 (nM), Target, etc.
            </p>
          </div>

          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">Current Data</h3>
            <div className="text-sm text-gray-300">
              <div>Compounds: <span className="text-white font-semibold">{data.length}</span></div>
              <div>Data Source: <span className="text-white font-semibold">
                {uploadedData ? 'Uploaded CSV' : analysisData ? 'Prediction Results' : 'None'}
              </span></div>
            </div>
          </div>
        </div>

        {data.length > 0 && (
          <>
            {/* Controls */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 mb-8">
              <div className="grid md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Chart Type</label>
                  <select
                    value={selectedChart}
                    onChange={(e) => setSelectedChart(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="bar">Bar Chart</option>
                    <option value="scatter">Scatter Plot</option>
                    <option value="heatmap">Heatmap</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Sort By</label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="ic50">IC₅₀ Value</option>
                    <option value="confidence">Confidence</option>
                    <option value="mw">Molecular Weight</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Filter Target</label>
                  <select
                    value={filterTarget}
                    onChange={(e) => setFilterTarget(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="all">All Targets</option>
                    {targets.map(target => (
                      <option key={target} value={target}>{target}</option>
                    ))}
                  </select>
                </div>

                <div className="flex items-end space-x-2">
                  <button
                    onClick={exportChart}
                    className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2"
                  >
                    <Download size={16} />
                    <span>Export Chart</span>
                  </button>
                  <CSVLink
                    data={csvData}
                    filename={`molecular-analysis-${new Date().toISOString().slice(0, 10)}.csv`}
                    className="bg-cyan-600 hover:bg-cyan-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2"
                  >
                    <Download size={16} />
                    <span>Export CSV</span>
                  </CSVLink>
                </div>
              </div>
            </div>

            {/* Visualization */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 mb-8">
              {selectedChart === 'bar' && <Plot {...getBarChartData()} style={{width: '100%', height: '500px'}} />}
              {selectedChart === 'scatter' && <Plot {...getScatterPlotData()} style={{width: '100%', height: '500px'}} />}
              {selectedChart === 'heatmap' && <Plot {...getHeatmapData()} style={{width: '100%', height: '500px'}} />}
            </div>

            {/* Data Table */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">Data Table</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-600">
                  <thead className="bg-gray-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Drug Name</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">SMILES</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">IC₅₀ (nM)</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Target</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Confidence</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">LogP</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">MW</th>
                    </tr>
                  </thead>
                  <tbody className="bg-gray-800 divide-y divide-gray-600">
                    {sortedData.slice(0, 20).map((item, index) => (
                      <tr key={index} className="hover:bg-gray-700">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">{item['Drug Name']}</td>
                        <td className="px-6 py-4 text-sm text-gray-300 font-mono max-w-xs truncate">{item.SMILES}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white">{parseFloat(item['IC50 (nM)']).toFixed(1)}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-cyan-400">{item.Target}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white">{(parseFloat(item.Confidence) * 100).toFixed(0)}%</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white">{parseFloat(item.LogP).toFixed(2)}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white">{parseFloat(item.MW).toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {sortedData.length > 20 && (
                  <div className="mt-4 text-center text-sm text-gray-400">
                    Showing first 20 of {sortedData.length} compounds
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {data.length === 0 && (
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-12 text-center">
            <BarChart3 size={64} className="mx-auto text-gray-600 mb-4" />
            <h3 className="text-xl font-semibold text-gray-400 mb-2">No Data Available</h3>
            <p className="text-gray-500 mb-6">
              Upload a CSV file or run predictions to analyze molecular data
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
            <Atom size={40} className="text-white" />
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
                  <li>• IC₅₀ bioactivity (6 targets)</li>
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

          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4 text-white">Technology Stack</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-2">Frontend</h3>
                <ul className="text-gray-300 space-y-1">
                  <li>• React.js with modern UI components</li>
                  <li>• Plotly.js for interactive visualizations</li>
                  <li>• Tailwind CSS for styling</li>
                  <li>• Responsive design</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-yellow-400 mb-2">Backend</h3>
                <ul className="text-gray-300 space-y-1">
                  <li>• FastAPI with async support</li>
                  <li>• MongoDB for data persistence</li>
                  <li>• PyTorch & Transformers</li>
                  <li>• RDKit for molecular processing</li>
                </ul>
              </div>
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
      {activeTab === 'analysis' && <AnalysisTab analysisData={analysisData} setAnalysisData={setAnalysisData} />}
      {activeTab === 'about' && <AboutTab />}
    </div>
  );
};

export default App;
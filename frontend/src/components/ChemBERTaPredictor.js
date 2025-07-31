import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ChemBERTaPredictor = () => {
  const [smiles, setSmiles] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [chembertaStatus, setChembertaStatus] = useState(null);

  // Example molecules for testing
  const exampleMolecules = [
    { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
    { name: 'Imatinib (Gleevec)', smiles: 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5' },
    { name: 'Gefitinib (Iressa)', smiles: 'COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4' },
    { name: 'Sorafenib', smiles: 'CNC(=O)C1=CC=CC=C1NC(=O)NC2=CC(=C(C=C2)Cl)C(F)(F)F' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
    { name: 'Ethanol', smiles: 'CCO' }
  ];

  // Check ChemBERTa status on component mount
  useEffect(() => {
    checkChemBERTaStatus();
  }, []);

  const checkChemBERTaStatus = async () => {
    try {
      const response = await axios.get(`${API}/chemberta/status`);
      setChembertaStatus(response.data);
    } catch (error) {
      console.error('Error checking ChemBERTa status:', error);
      setChembertaStatus({ status: 'error', available: false });
    }
  };

  const handlePredict = async () => {
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }

    setIsLoading(true);
    setError('');
    setPredictions(null);

    try {
      const response = await axios.post(`${API}/chemberta/predict`, {
        smiles: smiles.trim()
      }, {
        timeout: 120000, // 2 minutes timeout for GPU inference
      });

      if (response.data.status === 'success') {
        setPredictions(response.data);
        setError('');
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('ChemBERTa prediction error:', error);
      
      let errorMessage = 'Failed to get ChemBERTa predictions. Please try again.';
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Prediction timed out. GPU may be busy. Please try again.';
      } else if (error.response?.status === 503) {
        errorMessage = 'ChemBERTa service unavailable. Please try again later.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (exampleSmiles) => {
    setSmiles(exampleSmiles);
    setError('');
    setPredictions(null);
  };

  const formatIC50 = (ic50_um) => {
    if (ic50_um < 0.001) {
      return `${(ic50_um * 1000000).toFixed(0)} nM`;
    } else if (ic50_um < 1) {
      return `${(ic50_um * 1000).toFixed(0)} nM`;
    } else {
      return `${ic50_um.toFixed(2)} μM`;
    }
  };

  const getActivityIcon = (activity_class) => {
    switch (activity_class) {
      case 'Very High': return '🔥';
      case 'High': return '⚡';
      case 'Moderate': return '⚠️';
      case 'Low': return '📉';
      case 'Very Low': return '❄️';
      default: return '❓';
    }
  };

  const getPerformanceIcon = (r2_score) => {
    if (r2_score > 0.6) return '🌟';
    if (r2_score > 0.4) return '✅';
    if (r2_score > 0.2) return '⚠️';
    return '❌';
  };

  const sortedTargets = predictions?.predictions
    ? Object.entries(predictions.predictions)
        .sort(([, a], [, b]) => a.ic50_um - b.ic50_um)
    : [];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            ChemBERTa Multi-Task Predictor
          </h1>
          <p className="text-xl text-gray-300">
            Predict IC50 values for 10 oncoproteins using our trained ChemBERTa transformer model
          </p>
          
          {/* Status Badge */}
          {chembertaStatus && (
            <div className="mt-4 flex items-center space-x-2">
              <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                chembertaStatus.available 
                  ? 'bg-green-900 text-green-300 border border-green-700' 
                  : 'bg-red-900 text-red-300 border border-red-700'
              }`}>
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  chembertaStatus.available ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                }`}></div>
                {chembertaStatus.available ? 'ChemBERTa Ready' : 'ChemBERTa Unavailable'}
              </div>
              {chembertaStatus.available && (
                <span className="text-sm text-gray-400">
                  Mean R² Score: 0.516 | A100 GPU
                </span>
              )}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Section */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h2 className="text-2xl font-semibold mb-4 text-white">Molecular Input</h2>
              
              {/* SMILES Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  SMILES String
                </label>
                <textarea
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  placeholder="Enter SMILES string (e.g., CC(=O)OC1=CC=CC=C1C(=O)O for aspirin)"
                  className="w-full h-32 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-400 font-mono"
                />
              </div>

              {/* Example Molecules */}
              <div className="mb-6">
                <p className="text-sm font-medium text-gray-300 mb-3">Try these drug examples:</p>
                <div className="grid grid-cols-2 gap-2">
                  {exampleMolecules.map((mol, index) => (
                    <button
                      key={index}
                      onClick={() => handleExampleClick(mol.smiles)}
                      className="px-3 py-2 text-sm bg-gray-700 text-gray-200 border border-gray-600 rounded-lg hover:bg-gray-600 hover:border-purple-500 transition-all text-left"
                    >
                      <div className="font-medium">{mol.name}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Predict Button */}
              <button
                onClick={handlePredict}
                disabled={isLoading || !smiles.trim() || !chembertaStatus?.available}
                className="w-full bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700 text-white py-3 px-6 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Predicting on A100 GPU...
                  </div>
                ) : (
                  'Predict IC50 for All Targets'
                )}
              </button>
            </div>
          </div>

          {/* Model Info */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Model Information</h3>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-gray-400">Architecture:</span>
                <span className="text-white ml-2">ChemBERTa Multi-Task</span>
              </div>
              <div>
                <span className="text-gray-400">Training Targets:</span>
                <span className="text-white ml-2">10 Oncoproteins</span>
              </div>
              <div>
                <span className="text-gray-400">Training Samples:</span>
                <span className="text-white ml-2">5,022 Compounds</span>
              </div>
              <div>
                <span className="text-gray-400">Mean R²:</span>
                <span className="text-white ml-2">0.516</span>
              </div>
              <div>
                <span className="text-gray-400">Best Target:</span>
                <span className="text-white ml-2">EGFR (R² = 0.751)</span>
              </div>
              <div>
                <span className="text-gray-400">Infrastructure:</span>
                <span className="text-white ml-2">Modal A100 GPU</span>
              </div>
            </div>

            <div className="mt-6 p-4 bg-purple-900 bg-opacity-30 rounded-lg">
              <h4 className="text-sm font-medium text-purple-300 mb-2">🎯 Target Performance</h4>
              <div className="space-y-1 text-xs">
                <div>🌟 Excellent: EGFR, MDM2</div>
                <div>✅ Good: BRAF, PI3KCA, HER2, VEGFR2, MET, ALK</div>
                <div>⚠️ Fair: CDK4, CDK6</div>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-8 bg-red-900 border border-red-700 rounded-xl p-4">
            <div className="flex items-center">
              <div className="text-red-300">
                <h3 className="font-medium">Prediction Error</h3>
                <p className="text-sm mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {predictions && (
          <div className="mt-8 space-y-6">
            {/* Summary */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h3 className="text-2xl font-bold mb-4 text-white">Prediction Summary</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                <div className="bg-purple-900 bg-opacity-30 rounded-lg p-4">
                  <div className="text-2xl font-bold text-purple-400">
                    {predictions.summary?.best_target}
                  </div>
                  <div className="text-sm text-gray-300">Most Active Target</div>
                  <div className="text-lg text-white">
                    {formatIC50(predictions.summary?.best_ic50_um)}
                  </div>
                </div>
                <div className="bg-cyan-900 bg-opacity-30 rounded-lg p-4">
                  <div className="text-2xl font-bold text-cyan-400">
                    {predictions.summary?.highly_active_targets}
                  </div>
                  <div className="text-sm text-gray-300">High Activity Targets</div>
                  <div className="text-xs text-gray-400">(IC50 ≤ 1 μM)</div>
                </div>
                <div className="bg-green-900 bg-opacity-30 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-400">
                    {formatIC50(predictions.summary?.median_ic50_um)}
                  </div>
                  <div className="text-sm text-gray-300">Median IC50</div>
                  <div className="text-xs text-gray-400">Across all targets</div>
                </div>
              </div>
            </div>

            {/* Detailed Results */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h3 className="text-2xl font-bold mb-6 text-white">Detailed IC50 Predictions</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {sortedTargets.map(([target, data], index) => (
                  <div key={target} className="bg-gray-700 border border-gray-600 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg font-bold text-white">{target}</span>
                        <span className="text-xs px-2 py-1 bg-gray-600 text-gray-300 rounded">
                          #{index + 1}
                        </span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <span>{getActivityIcon(data.activity_class)}</span>
                        <span>{getPerformanceIcon(data.r2_score)}</span>
                      </div>
                    </div>
                    
                    <div className="text-sm text-gray-300 mb-2">
                      {predictions.target_info?.[target]?.description}
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-400">IC50:</span>
                        <span className="text-white font-mono text-lg">
                          {formatIC50(data.ic50_um)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Activity:</span>
                        <span 
                          className="font-medium"
                          style={{ color: data.activity_color }}
                        >
                          {data.activity_class}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Model R²:</span>
                        <span className="text-white">
                          {data.r2_score.toFixed(3)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Confidence:</span>
                        <span className="text-white">
                          {(data.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChemBERTaPredictor;
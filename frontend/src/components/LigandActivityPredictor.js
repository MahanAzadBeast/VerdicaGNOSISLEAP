import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const LigandActivityPredictor = () => {
  const [smiles, setSmiles] = useState('');
  const [selectedModel, setSelectedModel] = useState('unified');
  const [selectedProperties, setSelectedProperties] = useState(['bioactivity_ic50', 'toxicity']);
  const [selectedTarget, setSelectedTarget] = useState('EGFR');
  const [predictions, setPredictions] = useState(null);
  const [chembertaPredictions, setChembertaPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelStatus, setModelStatus] = useState({});

  // Available prediction models
  const predictionModels = [
    {
      id: 'unified',
      name: 'Unified Multi-Model',
      description: 'Combines all available models for comprehensive predictions',
      models: ['ChemBERTa', 'Chemprop', 'Enhanced RDKit'],
      icon: '🔬'
    },
    {
      id: 'chemberta-multitask',
      name: 'ChemBERTa Multi-Task',
      description: 'Transformer-based model for 10 oncoproteins simultaneously',
      models: ['ChemBERTa Multi-Task (10 targets)'],
      icon: '🧬'
    },
    {
      id: 'chemprop-multitask',
      name: 'Chemprop Multi-Task',
      description: 'Graph neural network for multi-target prediction',
      models: ['Chemprop GNN Multi-Task'],
      icon: '📊'
    },
    {
      id: 'enhanced-rdkit',
      name: 'Enhanced RDKit',
      description: 'Enhanced molecular descriptors with target-specific models',
      models: ['RDKit + ML Models'],
      icon: '⚗️'
    }
  ];

  // Available properties for prediction
  const propertyTypes = [
    { 
      id: 'bioactivity_ic50', 
      label: 'IC₅₀ / Bioactivity', 
      category: 'Bioactivity', 
      description: 'Half-maximal inhibitory concentration',
      unit: 'μM',
      models: ['unified', 'chemberta-multitask', 'chemprop-multitask', 'enhanced-rdkit']
    },
    { 
      id: 'toxicity', 
      label: 'General Toxicity', 
      category: 'Toxicity', 
      description: 'Overall toxicity probability',
      unit: '%',
      models: ['unified', 'chemprop-multitask', 'enhanced-rdkit']
    },
    { 
      id: 'logP', 
      label: 'LogP (Lipophilicity)', 
      category: 'Physicochemical', 
      description: 'Partition coefficient between octanol and water',
      unit: 'LogP',
      models: ['unified', 'chemprop-multitask', 'enhanced-rdkit']
    },
    { 
      id: 'solubility', 
      label: 'Aqueous Solubility', 
      category: 'Physicochemical', 
      description: 'Water solubility (LogS)',
      unit: 'LogS',
      models: ['unified', 'chemprop-multitask', 'enhanced-rdkit']
    }
  ];

  // Available targets
  const availableTargets = [
    { id: 'EGFR', name: 'EGFR', description: 'Epidermal Growth Factor Receptor' },
    { id: 'HER2', name: 'HER2', description: 'Human Epidermal Growth Factor Receptor 2' },
    { id: 'VEGFR2', name: 'VEGFR2', description: 'Vascular Endothelial Growth Factor Receptor 2' },
    { id: 'BRAF', name: 'BRAF', description: 'B-Raf Proto-Oncogene' },
    { id: 'MET', name: 'MET', description: 'MET Proto-Oncogene' },
    { id: 'CDK4', name: 'CDK4', description: 'Cyclin Dependent Kinase 4' },
    { id: 'CDK6', name: 'CDK6', description: 'Cyclin Dependent Kinase 6' },
    { id: 'ALK', name: 'ALK', description: 'Anaplastic Lymphoma Kinase' },
    { id: 'MDM2', name: 'MDM2', description: 'MDM2 Proto-Oncogene' },
    { id: 'PI3KCA', name: 'PI3KCA', description: 'Phosphatidylinositol-4,5-Bisphosphate 3-Kinase' }
  ];

  // Example molecules
  const exampleMolecules = [
    { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
    { name: 'Imatinib (Gleevec)', smiles: 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5' },
    { name: 'Gefitinib (Iressa)', smiles: 'COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4' },
    { name: 'Sorafenib', smiles: 'CNC(=O)C1=CC=CC=C1NC(=O)NC2=CC(=C(C=C2)Cl)C(F)(F)F' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
    { name: 'Ethanol', smiles: 'CCO' }
  ];

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      // Check multiple endpoints
      const [healthResponse, chembertaResponse] = await Promise.allSettled([
        axios.get(`${API}/health`),
        axios.get(`${API}/chemberta/status`)
      ]);

      const status = {};
      
      if (healthResponse.status === 'fulfilled') {
        status.unified = healthResponse.value.data.status === 'healthy';
        status.enhanced = healthResponse.value.data.enhanced_predictions || false;
      }
      
      if (chembertaResponse.status === 'fulfilled') {
        status.chemberta = chembertaResponse.value.data.available || false;
      }

      setModelStatus(status);
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const handlePredict = async () => {
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }

    if (selectedProperties.length === 0) {
      setError('Please select at least one property to predict');
      return;
    }

    setIsLoading(true);
    setError('');
    setPredictions(null);
    setChembertaPredictions(null);

    try {
      const promises = [];

      // Run unified predictions if selected
      if (selectedModel === 'unified' || selectedModel === 'enhanced-rdkit') {
        promises.push(
          axios.post(`${API}/predict`, {
            smiles: smiles.trim(),
            prediction_types: selectedProperties,
            target: selectedTarget
          }, { timeout: 60000 })
        );
      }

      // Run ChemBERTa multi-task if available and selected
      if ((selectedModel === 'unified' || selectedModel === 'chemberta-multitask') && 
          selectedProperties.includes('bioactivity_ic50') && modelStatus.chemberta) {
        promises.push(
          axios.post(`${API}/chemberta/predict`, {
            smiles: smiles.trim()
          }, { timeout: 120000 })
        );
      }

      const results = await Promise.allSettled(promises);

      // Process results
      if (results[0]?.status === 'fulfilled') {
        setPredictions(results[0].value.data);
      }

      if (results[1]?.status === 'fulfilled') {
        setChembertaPredictions(results[1].value.data);
      }

      if (results.every(r => r.status === 'rejected')) {
        throw new Error('All prediction models failed');
      }

    } catch (error) {
      console.error('Prediction error:', error);
      
      let errorMessage = 'Prediction failed. Please try again.';
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Prediction timed out. Models may be busy. Please try again.';
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
    setChembertaPredictions(null);
  };

  const handlePropertyToggle = (propertyId) => {
    const newSelection = selectedProperties.includes(propertyId)
      ? selectedProperties.filter(id => id !== propertyId)
      : [...selectedProperties, propertyId];
    setSelectedProperties(newSelection);
  };

  const getAvailableProperties = () => {
    return propertyTypes.filter(prop => 
      prop.models.includes(selectedModel) || selectedModel === 'unified'
    );
  };

  const formatValue = (value, property) => {
    if (typeof value !== 'number') return 'N/A';
    
    switch (property) {
      case 'bioactivity_ic50':
        return `${value.toFixed(2)} μM`;
      case 'toxicity':
        return `${(value * 100).toFixed(1)}%`;
      case 'logP':
      case 'solubility':
        return value.toFixed(2);
      default:
        return value.toFixed(3);
    }
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

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
          Ligand Activity Predictor Module
        </h2>
        <p className="text-gray-400">
          Comprehensive molecular property prediction using multiple AI models
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Input Section */}
        <div className="xl:col-span-2 space-y-6">
          {/* SMILES Input */}
          <div className="bg-gray-700 border border-gray-600 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Molecular Input</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                SMILES String
              </label>
              <textarea
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Enter SMILES string (e.g., CC(=O)OC1=CC=CC=C1C(=O)O for aspirin)"
                className="w-full h-24 px-3 py-2 bg-gray-600 border border-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 text-white placeholder-gray-400 font-mono text-sm"
              />
            </div>

            {/* Example Molecules */}
            <div className="mb-4">
              <p className="text-sm font-medium text-gray-300 mb-2">Quick Examples:</p>
              <div className="grid grid-cols-2 gap-2">
                {exampleMolecules.map((mol, index) => (
                  <button
                    key={index}
                    onClick={() => handleExampleClick(mol.smiles)}
                    className="px-3 py-2 text-sm bg-gray-600 text-gray-200 border border-gray-500 rounded-lg hover:bg-gray-500 hover:border-purple-400 transition-all text-left"
                  >
                    <div className="font-medium">{mol.name}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Model Selection */}
          <div className="bg-gray-700 border border-gray-600 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">AI Model Selection</h3>
            
            <div className="space-y-3">
              {predictionModels.map((model) => (
                <label key={model.id} className="flex items-start space-x-3 p-3 border border-gray-500 rounded-lg hover:bg-gray-600 cursor-pointer transition-all">
                  <input
                    type="radio"
                    name="model"
                    value={model.id}
                    checked={selectedModel === model.id}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="mt-1 w-4 h-4 text-purple-600 border-gray-400 focus:ring-purple-500"
                  />
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-lg">{model.icon}</span>
                      <div className="font-medium text-white">{model.name}</div>
                      {model.id === 'chemberta-multitask' && modelStatus.chemberta && (
                        <div className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-900 text-green-300 border border-green-700">
                          <div className="w-1 h-1 bg-green-400 rounded-full mr-1 animate-pulse"></div>
                          Ready
                        </div>
                      )}
                    </div>
                    <div className="text-sm text-gray-400 mb-2">{model.description}</div>
                    <div className="text-xs text-gray-500">
                      Models: {model.models.join(', ')}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Configuration Section */}
        <div className="xl:col-span-2 space-y-6">
          {/* Property Selection */}
          <div className="bg-gray-700 border border-gray-600 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Properties to Predict</h3>
            
            <div className="space-y-3">
              {getAvailableProperties().map((property) => (
                <label key={property.id} className="flex items-center space-x-3 p-3 border border-gray-500 rounded-lg hover:bg-gray-600 cursor-pointer transition-all">
                  <input
                    type="checkbox"
                    checked={selectedProperties.includes(property.id)}
                    onChange={() => handlePropertyToggle(property.id)}
                    className="w-4 h-4 text-purple-600 border-gray-400 rounded focus:ring-purple-500"
                  />
                  <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
                  <div className="flex-1">
                    <div className="font-medium text-white">{property.label}</div>
                    <div className="text-sm text-gray-400">{property.description}</div>
                    <div className="text-xs text-gray-500">Category: {property.category} | Unit: {property.unit}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Target Selection (for IC50) */}
          {selectedProperties.includes('bioactivity_ic50') && (
            <div className="bg-gray-700 border border-gray-600 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 text-white">Target Selection (IC₅₀)</h3>
              
              <select
                value={selectedTarget}
                onChange={(e) => setSelectedTarget(e.target.value)}
                className="w-full px-3 py-2 bg-gray-600 border border-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 text-white"
              >
                {availableTargets.map((target) => (
                  <option key={target.id} value={target.id}>
                    {target.name} - {target.description}
                  </option>
                ))}
              </select>

              {selectedModel === 'chemberta-multitask' && (
                <div className="mt-3 p-3 bg-purple-900 bg-opacity-30 rounded-lg">
                  <div className="text-sm text-purple-300">
                    <strong>ChemBERTa Multi-Task:</strong> Predicts IC₅₀ for all 10 targets simultaneously
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={isLoading || !smiles.trim() || selectedProperties.length === 0}
            className="w-full bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700 text-white py-4 px-6 rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all text-lg"
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                Running AI Predictions...
              </div>
            ) : (
              `Predict with ${predictionModels.find(m => m.id === selectedModel)?.name}`
            )}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-6 bg-red-900 border border-red-700 rounded-xl p-4">
          <div className="flex items-center">
            <div className="text-red-300">
              <h3 className="font-medium">Prediction Error</h3>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results Display */}
      {(predictions || chembertaPredictions) && (
        <div className="mt-6 space-y-6">
          {/* ChemBERTa Multi-Task Results */}
          {chembertaPredictions && selectedProperties.includes('bioactivity_ic50') && (
            <div className="bg-gray-700 border border-gray-600 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-white flex items-center">
                <span className="text-2xl mr-2">🧬</span>
                ChemBERTa Multi-Task Results
              </h3>
              
              {/* Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-purple-900 bg-opacity-30 rounded-lg p-4 text-center">
                  <div className="text-xl font-bold text-purple-400">
                    {chembertaPredictions.summary?.best_target}
                  </div>
                  <div className="text-sm text-gray-300">Most Active Target</div>
                  <div className="text-lg text-white">
                    {formatIC50(chembertaPredictions.summary?.best_ic50_um)}
                  </div>
                </div>
                <div className="bg-cyan-900 bg-opacity-30 rounded-lg p-4 text-center">
                  <div className="text-xl font-bold text-cyan-400">
                    {chembertaPredictions.summary?.highly_active_targets}
                  </div>
                  <div className="text-sm text-gray-300">High Activity Targets</div>
                  <div className="text-xs text-gray-400">(IC50 ≤ 1 μM)</div>
                </div>
                <div className="bg-green-900 bg-opacity-30 rounded-lg p-4 text-center">
                  <div className="text-xl font-bold text-green-400">
                    {formatIC50(chembertaPredictions.summary?.median_ic50_um)}
                  </div>
                  <div className="text-sm text-gray-300">Median IC50</div>
                  <div className="text-xs text-gray-400">Across all targets</div>
                </div>
              </div>

              {/* Target Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                {Object.entries(chembertaPredictions.predictions || {})
                  .sort(([, a], [, b]) => a.ic50_um - b.ic50_um)
                  .map(([target, data], index) => (
                  <div key={target} className="bg-gray-600 border border-gray-500 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-white text-sm">{target}</span>
                      <span className="text-xs px-1 py-1 bg-gray-500 text-gray-300 rounded">
                        #{index + 1}
                      </span>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-mono text-white">
                        {formatIC50(data.ic50_um)}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        R² = {data.r2_score.toFixed(3)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Unified Model Results */}
          {predictions && (
            <div className="bg-gray-700 border border-gray-600 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-white flex items-center">
                <span className="text-2xl mr-2">🔬</span>
                Enhanced Multi-Model Predictions
              </h3>

              {/* Summary */}
              {predictions.summary && (
                <div className="bg-gray-600 border border-gray-500 rounded-lg p-4 mb-6">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="text-gray-300">
                      Molecule: <span className="font-mono text-purple-400">{predictions.summary.molecule}</span>
                    </div>
                    <div className="text-gray-300">
                      Target: <span className="font-semibold text-cyan-400">{predictions.summary.target}</span>
                    </div>
                    <div className="text-gray-300">
                      Predictions: <span className="text-white">{predictions.summary.total_predictions}</span>
                    </div>
                    <div className="text-gray-300">
                      Enhanced Models: {predictions.summary.enhanced_models_used ? 
                        <span className="text-green-400">✅ Active</span> : 
                        <span className="text-red-400">❌ Not Used</span>
                      }
                    </div>
                  </div>
                </div>
              )}

              {/* Results Table */}
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-500">
                  <thead className="bg-gray-600">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Property</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Enhanced Model</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">ChemBERTa</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="bg-gray-700 divide-y divide-gray-500">
                    {predictions.results?.map((result, index) => (
                      <tr key={index} className="hover:bg-gray-600">
                        <td className="px-4 py-3 text-sm font-medium text-white">
                          {propertyTypes.find(p => p.id === result.prediction_type)?.label || result.prediction_type}
                        </td>
                        <td className="px-4 py-3 text-sm font-semibold text-purple-400">
                          {result.enhanced_chemprop_prediction ? (
                            <div>
                              {result.enhanced_chemprop_prediction.ic50_nm ? 
                                formatIC50(result.enhanced_chemprop_prediction.ic50_nm / 1000) :
                                formatValue(result.enhanced_chemprop_prediction, result.prediction_type)
                              }
                              <div className="text-xs text-gray-400">
                                {result.enhanced_chemprop_prediction.model_type || 'Enhanced'}
                              </div>
                            </div>
                          ) : (
                            formatValue(result.chemprop_prediction, result.prediction_type)
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300">
                          {formatValue(result.molbert_prediction, result.prediction_type)}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                            result.confidence > 0.8 ? 'bg-green-900 text-green-300' : 
                            result.confidence > 0.6 ? 'bg-yellow-900 text-yellow-300' : 
                            'bg-red-900 text-red-300'
                          }`}>
                            {(result.confidence * 100).toFixed(0)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LigandActivityPredictor;
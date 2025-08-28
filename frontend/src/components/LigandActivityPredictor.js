import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { saveAs } from 'file-saver';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const LigandActivityPredictor = () => {
  const [smiles, setSmiles] = useState('');
  const [selectedTargets, setSelectedTargets] = useState(['all']);
  const [selectedAssayTypes, setSelectedAssayTypes] = useState(['IC50', 'EC50']); // Removed Ki due to invalid training data
  const [availableTargetsFiltered, setAvailableTargetsFiltered] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [sortBy, setSortBy] = useState({ column: null, direction: 'asc' });
  const [isExporting, setIsExporting] = useState(false);

  // PDF Export functionality
  const handleExportPDF = async () => {
    if (!predictions) {
      setError('No predictions available to export');
      return;
    }

    setIsExporting(true);
    try {
      const response = await axios.post(`${API}/reports/export-pdf`, predictions, {
        responseType: 'blob',
        timeout: 30000, // 30 second timeout for PDF generation
        headers: {
          'Content-Type': 'application/json'
        }
      });

      // Get filename from response headers or create default
      const contentDisposition = response.headers['content-disposition'];
      let filename = 'veridica_prediction_report.pdf';
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      // Save the PDF file
      saveAs(new Blob([response.data], { type: 'application/pdf' }), filename);
      
    } catch (error) {
      console.error('PDF export failed:', error);
      setError(`PDF export failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  // Column sorting functionality
  const handleSort = (column) => {
    setSortBy(prev => ({
      column,
      direction: prev.column === column && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  // Sort predictions based on current sort settings
  const sortedPredictions = predictions ? (() => {
    const entries = Object.entries(predictions.predictions);
    
    if (!sortBy.column) return entries;
    
    return entries.sort((a, b) => {
      const [targetA, dataA] = a;
      const [targetB, dataB] = b;
      
      let valueA, valueB;
      
      if (sortBy.column === 'potency') {
        // Sort by IC50 potency (lower IC50 = higher potency = higher pIC50)
        valueA = dataA.IC50?.pActivity || 0;
        valueB = dataB.IC50?.pActivity || 0;
      } else if (sortBy.column === 'confidence') {
        // Sort by average confidence across assay types
        const avgConfA = ['IC50', 'EC50'].reduce((sum, type) => 
          sum + (dataA[type]?.confidence || 0), 0) / 2;
        const avgConfB = ['IC50', 'EC50'].reduce((sum, type) => 
          sum + (dataB[type]?.confidence || 0), 0) / 2;
        valueA = avgConfA;
        valueB = avgConfB;
      } else {
        return 0;
      }
      
      const result = valueA - valueB;
      return sortBy.direction === 'desc' ? -result : result;
    });
  })() : [];
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [availableTargets, setAvailableTargets] = useState({});
  const [modelInfo, setModelInfo] = useState(null);
  const [trainingData, setTrainingData] = useState({});

  // Load available targets, model info, and training data on component mount
  useEffect(() => {
    loadGnosisIInfo();
    loadAvailableTargets();
    loadTrainingData();
  }, []);

  // Update filtered targets when assay types or training data changes
  useEffect(() => {
    if (availableTargets && trainingData) {
      const filtered = filterTargetsByAssayTypes(availableTargets, selectedAssayTypes);
      setAvailableTargetsFiltered(filtered);
    }
  }, [availableTargets, selectedAssayTypes, trainingData]);

  const loadGnosisIInfo = async () => {
    try {
      const response = await axios.get(`${API}/gnosis-i/info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Error loading Gnosis I info:', error);
    }
  };

  const loadTrainingData = async () => {
    try {
      const response = await axios.get(`${API}/gnosis-i/training-data`);
      if (response.data.available) {
        setTrainingData(response.data.training_data);
      }
    } catch (error) {
      console.error('Error loading training data:', error);
    }
  };

  const loadAvailableTargets = async () => {
    try {
      const response = await axios.get(`${API}/gnosis-i/targets`);
      setAvailableTargets(response.data);
    } catch (error) {
      console.error('Error loading targets:', error);
      setError('Failed to load available targets');
    }
  };

  const handleTargetChange = (target) => {
    if (target === 'all') {
      setSelectedTargets(['all']);
    } else {
      const newTargets = selectedTargets.includes('all') ? [target] : [...selectedTargets];
      if (newTargets.includes(target)) {
        const filtered = newTargets.filter(t => t !== target);
        setSelectedTargets(filtered.length === 0 ? ['all'] : filtered);
      } else {
        setSelectedTargets([...newTargets.filter(t => t !== 'all'), target]);
      }
    }
  };

  // Handle assay type selection changes
  const handleAssayTypeChange = (assayType) => {
    let newSelectedTypes;
    if (selectedAssayTypes.includes(assayType)) {
      // Remove if already selected, but keep at least one
      newSelectedTypes = selectedAssayTypes.filter(type => type !== assayType);
      if (newSelectedTypes.length === 0) {
        newSelectedTypes = [assayType]; // Keep at least one selected
        return;
      }
    } else {
      // Add if not selected
      newSelectedTypes = [...selectedAssayTypes, assayType];
    }
    setSelectedAssayTypes(newSelectedTypes);
  };

  // Filter available targets based on selected assay types using training data
  const filterTargetsByAssayTypes = (targets, assayTypes) => {
    if (!targets || !targets.categorized_targets || !trainingData) return targets;

    const filterTargetList = (targetList) => {
      return targetList.filter(target => {
        // Check if target has training data available for ANY of the selected assay types
        return assayTypes.some(assayType => {
          const targetTraining = trainingData[target];
          if (!targetTraining) return false;
          
          // Map frontend assay types to backend training data keys
          let trainingKeys = [];
          if (assayType === 'IC50') {
            trainingKeys = ['IC50']; // Backend has IC50 training data
          } else if (assayType === 'EC50') {
            trainingKeys = ['EC50']; // Backend has EC50 training data  
          } else if (assayType === 'Ki') {
            trainingKeys = ['Ki']; // Backend has Ki training data
          }
          
          // Check if this assay type has sufficient training data (>= 25 samples)
          return trainingKeys.some(key => {
            const sampleCount = targetTraining[key];
            return sampleCount && sampleCount >= 25; // Minimum threshold for reliable predictions
          });
        });
      });
    };

    return {
      ...targets,
      categorized_targets: {
        oncoproteins: filterTargetList(targets.categorized_targets.oncoproteins || []),
        tumor_suppressors: filterTargetList(targets.categorized_targets.tumor_suppressors || []),
        other_targets: filterTargetList(targets.categorized_targets.other_targets || []),
        all_targets: [...filterTargetList(targets.categorized_targets.oncoproteins || []), 
                     ...filterTargetList(targets.categorized_targets.tumor_suppressors || []), 
                     ...filterTargetList(targets.categorized_targets.other_targets || [])]
      },
      available_targets: [...filterTargetList(targets.categorized_targets.oncoproteins || []), 
                         ...filterTargetList(targets.categorized_targets.tumor_suppressors || []), 
                         ...filterTargetList(targets.categorized_targets.other_targets || [])]
    };
  };

  // Update filtered targets when assay types change
  useEffect(() => {
    if (availableTargets) {
      const filtered = filterTargetsByAssayTypes(availableTargets, selectedAssayTypes);
      setAvailableTargetsFiltered(filtered);
      
      // Reset selected targets if they're no longer available
      if (selectedTargets.length > 0 && !selectedTargets.includes('all')) {
        const stillAvailable = selectedTargets.filter(target => 
          filtered.available_targets.includes(target)
        );
        if (stillAvailable.length === 0) {
          setSelectedTargets(['all']);
        } else if (stillAvailable.length !== selectedTargets.length) {
          setSelectedTargets(stillAvailable);
        }
      }
    }
  }, [availableTargets, selectedAssayTypes]);

  // Heat-map color calculation with continuous HSL gradient
  const calculatePotencyColor = (pValue, confidence = 0.8, isUnreliable = false, isNotTrained = false, isGated = false) => {
    // **UNIVERSAL GATING SYSTEM** - Special styling for gated predictions
    if (isGated) {
      return {
        backgroundColor: 'hsl(0, 0%, 30%)', // Gray for gated predictions
        opacity: 0.9,
        color: '#fbbf24', // Amber text for gated
        fontWeight: 'bold',
        border: '2px solid #f59e0b',
        textShadow: '0 1px 2px rgba(0,0,0,0.5)'
      };
    }
    
    // Special styling for not trained predictions (Ki)
    if (isNotTrained) {
      return {
        backgroundColor: 'hsl(0, 0%, 25%)', // Dark gray for not trained
        opacity: 0.8,
        color: '#fbbf24', // Amber text for warning
        fontWeight: 'bold',
        border: '2px solid #f59e0b',
        textShadow: '0 1px 2px rgba(0,0,0,0.5)'
      };
    }
    
    // Special styling for unreliable predictions
    if (isUnreliable) {
      return {
        backgroundColor: 'hsl(0, 0%, 40%)', // Gray for unreliable
        opacity: 0.6,
        color: 'white',
        fontWeight: 'bold',
        border: '2px dashed #64748b'
      };
    }
    
    // Clip pIC50 values - should never go below 0
    const clippedPValue = Math.max(0, pValue);
    
    // Continuous HSL gradient mapping: 220° (blue) → 0° (red)
    const hue = (p) => {
      if (p >= 9) return 220; // deep blue (picomolar)
      if (p >= 7) return 140 + (220-140)*(p-7)/2; // blue-green (low-nano)
      if (p >= 5) return 50 + (140-50)*(p-5)/2;   // yellow-green (μM)  
      if (p >= 3) return 25 + (50-25)*(p-3)/2;    // orange (high μM)
      return 0; // red (inactive)
    };
    
    // Lightness/opacity modulated by reliability index (0.4-1.0)
    const opacity = 0.4 + 0.6 * confidence;
    const lightness = 45 + 10 * confidence; // 45-55% lightness range
    
    return {
      backgroundColor: `hsl(${hue(clippedPValue)}, 70%, ${lightness}%)`,
      opacity: opacity,
      color: 'white',
      fontWeight: 'bold'
    };
  };

  // Check if compound is inactive (pIC50 < 3 or IC50 > 100 μM)
  const isInactive = (pValue, activityUM) => {
    return pValue < 3 || activityUM > 100;
  };

  // Format potency display with extreme value capping and assay-specific handling
  const formatPotencyDisplay = (pValue, activityUM, assayType, qualityFlag, status) => {
    // **UNIVERSAL GATING SYSTEM** - Handle HYPOTHESIS_ONLY status
    if (status === 'HYPOTHESIS_ONLY') {
      return {
        primaryText: "Hypothesis only",
        secondaryText: "Out of domain", 
        isExtreme: true,
        isUnreliable: true,
        isGated: true,
        isNotTrained: false
      };
    }
    
    const clippedPValue = Math.max(0, pValue);
    
    // Special handling for Ki predictions that were not trained
    if (qualityFlag === 'not_trained') {
      return {
        primaryText: "Not trained",
        secondaryText: "(No Ki data)", 
        isExtreme: true,
        isUnreliable: true,
        isNotTrained: true,
        isGated: false
      };
    }
    
    // Special handling for other unreliable predictions
    if (qualityFlag === 'low_confidence' && activityUM > 100000) {
      return {
        primaryText: "No binding",
        secondaryText: `(${assayType} unreliable)`, 
        isExtreme: true,
        isUnreliable: true,
        isNotTrained: false,
        isGated: false
      };
    }
    
    if (activityUM > 100) {
      return {
        primaryText: "> 100 μM",
        secondaryText: "—", // Omit pIC50 for extreme values
        isExtreme: true,
        isUnreliable: false,
        isNotTrained: false,
        isGated: false
      };
    }
    
    return {
      primaryText: `${activityUM.toFixed(1)} μM`,
      secondaryText: `p${assayType} = ${clippedPValue.toFixed(2)}`,
      isExtreme: false,
      isUnreliable: false,
      isNotTrained: false,
      isGated: false
    };
  };

  // Enhanced selectivity analysis with N/A handling
  const calculateSelectivityDisplay = (targetData, allPredictions) => {
    const selectivityRatio = targetData.selectivity_ratio;
    
    // Check if we have sufficient off-target data
    const numTargets = Object.keys(allPredictions).length;
    if (numTargets < 2 || selectivityRatio === null || selectivityRatio === undefined) {
      return {
        ratio: null,
        category: 'Panel not available',
        color: 'gray',
        tooltip: 'Selectivity ratio could not be calculated – off-target panel missing.'
      };
    }
    
    // Standard selectivity calculation
    if (selectivityRatio >= 10) {
      return { 
        ratio: selectivityRatio, 
        category: 'Selective', 
        color: 'green',
        tooltip: `Highly selective (${selectivityRatio.toFixed(1)}× vs off-targets)`
      };
    }
    if (selectivityRatio >= 3) {
      return { 
        ratio: selectivityRatio, 
        category: 'Moderate', 
        color: 'yellow',
        tooltip: `Moderately selective (${selectivityRatio.toFixed(1)}× vs off-targets)`
      };
    }
    return { 
      ratio: selectivityRatio, 
      category: 'Non-selective', 
      color: 'red',
      tooltip: `Poor selectivity (${selectivityRatio.toFixed(1)}× vs off-targets)`
    };
  };

  // Get potency description
  const getPotencyDescription = (pValue) => {
    if (pValue >= 9) return 'Very High (picomolar)';
    if (pValue >= 7) return 'High (low nanomolar)';
    if (pValue >= 5) return 'Moderate (micromolar)';
    if (pValue >= 3) return 'Low (high micromolar)';
    return 'Inactive (>100 μM)';
  };

  const handlePredict = async () => {
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }

    const isAllTargets = selectedTargets.includes('all');
    const targetCount = isAllTargets ? availableTargets.total_targets : selectedTargets.length;

    setIsLoading(true);
    setError('');
    setPredictions(null);
    
    // **UX IMPROVEMENT**: Show immediate visual feedback
    console.log('🔄 Starting prediction with Universal Gating System...');

    try {
      // Prepare targets array - if 'all' is selected, use all available targets
      const targetsToPredict = selectedTargets.includes('all') 
        ? availableTargets.available_targets || []
        : selectedTargets;
      
      const response = await axios.post(`${API}/gnosis-i/predict-with-hp-ad`, {
        smiles: smiles.trim(),
        targets: targetsToPredict,
        assay_types: selectedAssayTypes // Send selected assay types to backend
      }, {
        timeout: 120000 // 2 minutes timeout for large predictions
      });

      setPredictions(response.data);
      
      // **UX IMPROVEMENT**: Auto-scroll to results after prediction completes
      setTimeout(() => {
        const resultsElement = document.getElementById('prediction-results');
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (error) {
      console.error('Error making prediction:', error);
      
      // Handle different types of errors properly
      let errorMessage = 'Error making prediction';
      
      if (error.response?.data) {
        if (typeof error.response.data === 'string') {
          errorMessage = error.response.data;
        } else if (error.response.data.detail) {
          if (typeof error.response.data.detail === 'string') {
            errorMessage = error.response.data.detail;
          } else if (Array.isArray(error.response.data.detail)) {
            // Handle validation errors array
            errorMessage = error.response.data.detail.map(err => 
              typeof err === 'string' ? err : err.msg || 'Validation error'
            ).join(', ');
          } else {
            errorMessage = 'Validation error occurred';
          }
        } else if (error.response.data.message) {
          errorMessage = error.response.data.message;
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectAllTargets = () => {
    if (availableTargets.available_targets) {
      setSelectedTargets([...availableTargets.available_targets]);
    }
  };

  const handleDeselectAllTargets = () => {
    setSelectedTargets([]);
  };

  const getSelectableTargets = () => {
    return availableTargetsFiltered || availableTargets;
  };

  const renderTargetSection = (categoryTargets, categoryName, categoryKey) => {
    const selectableTargets = getSelectableTargets();
    if (!selectableTargets.categorized_targets) return null;
    
    const targets = selectableTargets.categorized_targets[categoryKey] || [];
    if (targets.length === 0) return null;
    
    return (
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-300 mb-2">{categoryName} ({targets.length})</h4>
        <div className="grid grid-cols-2 gap-1">
          {targets.map(target => (
            <label key={target} className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedTargets.includes(target) || selectedTargets.includes('all')}
                onChange={() => handleTargetChange(target)}
                className="w-3 h-3 text-blue-600 bg-gray-700 border-gray-500 rounded focus:ring-blue-500"
              />
              <span className="text-xs text-gray-300">{target}</span>
            </label>
          ))}
        </div>
      </div>
    );
  };

  const renderTargetSelection = () => {
    const selectableTargets = getSelectableTargets();
    if (!selectableTargets.categorized_targets) return null;

    const { oncoproteins, tumor_suppressors, other_targets } = selectableTargets.categorized_targets;
    const allTargetsSelected = selectedTargets.includes('all');
    const someTargetsSelected = selectedTargets.length > 0 && !allTargetsSelected;

    return (
      <div className="space-y-4">
        {/* Control Buttons */}
        <div className="flex flex-wrap gap-2 mb-4">
          <button
            onClick={() => handleTargetChange('all')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              allTargetsSelected
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Select All ({availableTargets.total_targets})
          </button>
          
          <button
            onClick={handleSelectAllTargets}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-purple-600 text-white hover:bg-purple-700 transition-all"
          >
            Select Individual Targets
          </button>
          
          <button
            onClick={handleDeselectAllTargets}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-600 text-gray-300 hover:bg-gray-500 transition-all"
          >
            Clear Selection
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Oncoproteins */}
          <div className="space-y-2">
            <h4 className="font-semibold text-purple-400 border-b border-gray-600 pb-1">🎯 Oncoproteins ({oncoproteins?.length || 0})</h4>
            <div className="max-h-48 overflow-y-auto space-y-1">
              {oncoproteins?.map(target => (
                <label key={target} className="flex items-center space-x-2 p-2 hover:bg-gray-700 rounded cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={selectedTargets.includes(target) || allTargetsSelected}
                    onChange={() => handleTargetChange(target)}
                    className="w-3 h-3 text-red-600 bg-gray-700 border-gray-500 rounded focus:ring-red-500"
                  />
                  <span className="text-sm text-gray-300">{target}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Tumor Suppressors */}
          <div className="space-y-2">
            <h4 className="font-semibold text-green-400 border-b border-gray-600 pb-1">🛡️ Tumor Suppressors ({tumor_suppressors?.length || 0})</h4>
            <div className="max-h-48 overflow-y-auto space-y-1">
              {tumor_suppressors?.map(target => (
                <label key={target} className="flex items-center space-x-2 p-2 hover:bg-gray-700 rounded cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={selectedTargets.includes(target) || allTargetsSelected}
                    onChange={() => handleTargetChange(target)}
                    className="w-3 h-3 text-green-600 bg-gray-700 border-gray-500 rounded focus:ring-green-500"
                  />
                  <span className="text-sm text-gray-300">{target}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Other Targets */}
          <div className="space-y-2">
            <h4 className="font-semibold text-cyan-400 border-b border-gray-600 pb-1">⚗️ Other Targets ({other_targets?.length || 0})</h4>
            <div className="max-h-48 overflow-y-auto space-y-1">
              {other_targets?.map(target => (
                <label key={target} className="flex items-center space-x-2 p-2 hover:bg-gray-700 rounded cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={selectedTargets.includes(target) || allTargetsSelected}
                    onChange={() => handleTargetChange(target)}
                    className="w-3 h-3 text-cyan-600 bg-gray-700 border-gray-500 rounded focus:ring-cyan-500"
                  />
                  <span className="text-sm text-gray-300">{target}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Selection Summary */}
        {someTargetsSelected && (
          <div className="mt-3 p-3 bg-gray-700 rounded-lg">
            <span className="text-sm text-gray-300">
              Selected: <span className="font-medium text-blue-400">{selectedTargets.length}</span> of {availableTargets.total_targets} targets
            </span>
          </div>
        )}
      </div>
    );
  };

  const renderResults = () => {
    if (!predictions) return null;

    const { properties, predictions: targetPredictions, model_info } = predictions;

    return (
      <div id="prediction-results" className="mt-8 space-y-6">
        {/* Warning Banner */}
        <div className="bg-gradient-to-r from-amber-900 to-orange-900 border border-amber-700 rounded-lg p-3 text-center">
          <span className="text-amber-200">🔍 In-silico estimates – wet-lab validation required.</span>
        </div>

        {/* Model Training Notice - Enhanced */}
        {/* Molecular Properties */}
        <div className="bg-gradient-to-r from-blue-900 to-indigo-900 p-6 rounded-lg border border-blue-700">
          <h3 className="text-lg font-bold text-white mb-4">📊 Molecular Properties</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-gray-800 border border-gray-600 rounded-lg">
              <div className="text-2xl font-bold text-blue-400">{properties.LogP}</div>
              <div className="text-sm text-gray-300">LogP (Partition Coefficient)</div>
            </div>
            <div className="text-center p-4 bg-gray-800 border border-gray-600 rounded-lg">
              <div className="text-2xl font-bold text-green-400">{properties.LogS}</div>
              <div className="text-sm text-gray-300">LogS (Water Solubility)</div>
            </div>
          </div>
        </div>

        {/* Target Predictions Table */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg shadow-lg overflow-hidden">
          <div className="p-6 border-b border-gray-700">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-lg font-bold text-white">🎯 Target Activity Predictions</h3>
                <div className="text-sm text-gray-400 mt-1">
                  Gnosis I • R² = {model_info.r2_score?.toFixed(3)} • {Object.keys(targetPredictions).length} targets
                </div>
              </div>
              
              {/* Export PDF Button */}
              <button
                onClick={handleExportPDF}
                disabled={isExporting}
                className="inline-flex items-center px-4 py-2 border border-gray-600 text-sm font-medium rounded-lg text-gray-300 bg-gray-700 hover:bg-gray-600 hover:text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title="Download branded PDF report of current predictions"
              >
                {isExporting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Generating PDF...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3M3 17V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                    </svg>
                    Export PDF
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Predictions Table with Sorting */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Target Protein
                  </th>
                  <th 
                    className="px-4 py-3 text-center text-xs font-medium text-gray-300 uppercase tracking-wider cursor-pointer hover:text-white transition-colors"
                    onClick={() => handleSort('potency')}
                  >
                    <div className="flex items-center justify-center space-x-1">
                      <span>Potency (Heat-map)</span>
                      {sortBy.column === 'potency' && (
                        <span className="text-purple-400">
                          {sortBy.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-4 py-3 text-center text-xs font-medium text-gray-300 uppercase tracking-wider cursor-help"
                    title="Selectivity measures how specific a compound is for the target protein versus off-target proteins. Higher ratios indicate better selectivity: ≥10× = Selective (low off-target effects), 3-10× = Moderate selectivity, <3× = Non-selective (potential side effects)"
                  >
                    Selectivity
                  </th>
                  <th 
                    className="px-4 py-3 text-center text-xs font-medium text-gray-300 uppercase tracking-wider cursor-pointer hover:text-white transition-colors cursor-help"
                    onClick={() => handleSort('confidence')}
                    title="Confidence represents the model's certainty in the prediction based on Monte Carlo dropout analysis. Higher confidence (closer to 100%) indicates more reliable predictions. Derived from the reliability index (RI) which quantifies prediction uncertainty."
                  >
                    <div className="flex items-center justify-center space-x-1">
                      <span>Confidence</span>
                      {sortBy.column === 'confidence' && (
                        <span className="text-purple-400">
                          {sortBy.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-4 py-3 text-center text-xs font-medium text-gray-300 uppercase tracking-wider cursor-help"
                    title="Assay Type indicates the experimental method: Binding IC50 = direct protein binding inhibition, Functional IC50 = protein function inhibition, Ki = equilibrium binding affinity (dissociation constant), EC50 = concentration for 50% maximal biological response. All values in μM, with lower values indicating stronger binding/activity."
                  >
                    Assay Type
                  </th>
                </tr>
              </thead>
              <tbody className="bg-gray-800 divide-y divide-gray-700">
                {Object.entries(targetPredictions).map(([target, targetData], index) => {
                  // Show only selected assay types that have data
                  const assayTypeMap = {'IC50': ['Binding_IC50', 'Functional_IC50'], 'EC50': ['EC50']};
                  const availableAssayTypes = [];
                  
                  selectedAssayTypes.forEach(selectedType => {
                    const backendTypes = assayTypeMap[selectedType] || [selectedType];
                    backendTypes.forEach(backendType => {
                      if (targetData[backendType]) {
                        availableAssayTypes.push(backendType);
                      }
                    });
                  });
                  
                  const selectivity = calculateSelectivityDisplay(targetData, targetPredictions);
                  
                  return availableAssayTypes.map((assayType, assayIndex) => {
                    const prediction = targetData[assayType];
                    if (!prediction) return null;
                    
                    // **UNIVERSAL GATING SYSTEM** - Handle gated predictions
                    const status = prediction.status || 'OK';
                    const isGated = status === 'HYPOTHESIS_ONLY';
                    
                    // For gated predictions, numeric fields are suppressed by backend
                    const pValue = isGated ? null : prediction.pActivity;
                    const activityUM = isGated ? null : prediction.activity_uM;
                    const confidence = prediction.confidence || 0.8;
                    const sigma = prediction.sigma || 0.2;
                    const reliability = Math.exp(-sigma * sigma);
                    const qualityFlag = prediction.quality_flag || 'good';
                    const confidenceNote = prediction.confidence_note || '';
                    
                    const potencyDisplay = formatPotencyDisplay(pValue, activityUM, assayType, qualityFlag, status);
                    const potencyColor = calculatePotencyColor(pValue, confidence, potencyDisplay.isUnreliable, potencyDisplay.isNotTrained, potencyDisplay.isGated);
                    const inactive = isGated ? false : isInactive(pValue, activityUM);
                    const potencyDesc = isGated ? 'Gated prediction' : getPotencyDescription(Math.max(0, pValue || 0));

                    return (
                      <tr key={`${target}-${assayType}`} className="hover:bg-gray-700 transition-colors">
                        {/* Target Name - show only for first assay type */}
                        {assayIndex === 0 ? (
                          <td className="px-4 py-4" rowSpan={availableAssayTypes.length}>
                            <div className="font-medium text-white">{target}</div>
                            <div className="text-xs text-gray-400">Oncoprotein</div>
                          </td>
                        ) : null}

                        {/* Enhanced Potency Heat-map Cell */}
                        <td className="px-4 py-4 text-center">
                          <div className="relative">
                            <div
                              className="inline-block px-3 py-2 rounded-lg font-mono cursor-pointer relative"
                              style={potencyColor}
                              title={potencyDisplay.isGated ? 
                                `${assayType} prediction gated: ${prediction.message || 'Out of domain for this target class. Numeric potency suppressed.'} Reasons: ${prediction.why ? prediction.why.join(', ') : 'Biologically implausible'}` :
                                potencyDisplay.isNotTrained ? 
                                `${assayType} predictions not available: Insufficient training data for this target. Only targets with ≥10 experimental samples provide reliable predictions.` :
                                potencyDisplay.isUnreliable ? 
                                `${assayType} prediction unreliable: ${confidenceNote || 'Model uncertainty too high for reliable prediction'}` :
                                `${potencyDesc} (p${assayType} = ${Math.max(0, pValue || 0).toFixed(2)}), σ = ${sigma.toFixed(2)}, RI = ${reliability.toFixed(2)}${confidenceNote ? ` - ${confidenceNote}` : ''}`
                              }
                            >
                              {/* **UNIVERSAL GATING SYSTEM** - Gating indicator */}
                              {potencyDisplay.isGated && (
                                <span className="absolute -top-1 -left-1 text-amber-400 text-sm" title="Prediction gated by Universal Gating System">
                                  🛡️
                                </span>
                              )}
                              
                              {/* Quality warning for unreliable predictions */}
                              {qualityFlag === 'not_trained' && (
                                <span className="absolute -top-1 -left-1 text-amber-400 text-sm" title="Not trained on this assay type">
                                  🚫
                                </span>
                              )}
                              
                              {qualityFlag === 'uncertain' && !potencyDisplay.isGated && (
                                <span className="absolute -top-1 -left-1 text-yellow-400 text-xs" title={`Prediction quality: ${qualityFlag}`}>
                                  ⚠️
                                </span>
                              )}
                              
                              {/* Color-blind aid for inactive compounds */}
                              {inactive && !potencyDisplay.isNotTrained && !potencyDisplay.isGated && (
                                <span className="absolute -top-1 -right-1 text-white text-xs">❌</span>
                              )}
                              
                              {/* Visual hierarchy: emphasize p-units over raw μM */}
                              <div className="text-base font-medium">{potencyDisplay.primaryText}</div>
                              <div className="text-xs text-slate-300 opacity-90">{potencyDisplay.secondaryText}</div>
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                              {potencyDisplay.isGated ? 'Gated prediction' : 
                               potencyDisplay.isNotTrained ? 'Training data unavailable' : potencyDesc}
                              {potencyDisplay.isGated && (
                                <div className="text-amber-400 text-xs mt-1">
                                  🛡️ Universal Gating System
                                </div>
                              )}
                              {qualityFlag === 'not_trained' && (
                                <div className="text-amber-400 text-xs mt-1">
                                  🚫 Requires {assayType}-specific training
                                </div>
                              )}
                              {qualityFlag === 'uncertain' && !potencyDisplay.isNotTrained && !potencyDisplay.isGated && (
                                <div className="text-yellow-400 text-xs mt-1">
                                  ⚠️ {qualityFlag === 'low_confidence' ? 'Low confidence' : 'Uncertain'}
                                </div>
                              )}
                            </div>
                          </div>
                        </td>

                        {/* Enhanced Selectivity - show only for first assay type */}
                        {assayIndex === 0 ? (
                          <td className="px-4 py-4 text-center" rowSpan={availableAssayTypes.length}>
                            <div 
                              className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                                selectivity.color === 'green' ? 'bg-green-900 text-green-300 border border-green-700' :
                                selectivity.color === 'yellow' ? 'bg-yellow-900 text-yellow-300 border border-yellow-700' :
                                selectivity.color === 'red' ? 'bg-red-900 text-red-300 border border-red-700' :
                                'bg-gray-900 text-gray-400 border border-gray-600'
                              }`}
                              title={selectivity.tooltip}
                            >
                              {/* Color-blind aid for selective compounds */}
                              {selectivity.color === 'green' && <span className="mr-1">✔︎</span>}
                              {selectivity.color === 'yellow' && <span className="mr-1">🟡</span>}
                              {selectivity.color === 'red' && <span className="mr-1">❌</span>}
                              {selectivity.color === 'gray' && <span className="mr-1">—</span>}
                              
                              {selectivity.ratio ? 
                                `${selectivity.ratio.toFixed(1)}×` : 
                                selectivity.category
                              }
                            </div>
                            <div className="text-xs text-gray-400 mt-1">{selectivity.category}</div>
                          </td>
                        ) : null}

                        {/* Enhanced Confidence with inactive compound handling */}
                        <td className="px-4 py-4 text-center">
                          <div className="flex flex-col items-center">
                            <div className="text-white font-medium">{(confidence * 100).toFixed(0)}%</div>
                            <div className="w-16 bg-gray-600 rounded-full h-2 mt-1">
                              <div 
                                className={`h-2 rounded-full transition-all ${
                                  inactive ? 'bg-gray-400' : 'bg-gradient-to-r from-purple-500 to-cyan-500'
                                }`}
                                style={{ width: `${confidence * 100}%` }}
                                title={inactive ? 
                                  `High confidence that compound is inactive on ${target}.` :
                                  `Confidence: ${(confidence * 100).toFixed(0)}%, RI: ${reliability.toFixed(2)}`
                                }
                              ></div>
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                              RI = {reliability.toFixed(2)}
                            </div>
                          </div>
                        </td>

                        {/* Assay Type with enhanced styling and hover descriptions */}
                        <td className="px-4 py-4 text-center">
                          <div 
                            className={`inline-flex items-center px-2 py-1 border rounded-full text-xs font-medium cursor-help ${
                              assayType === 'Binding_IC50' ? 'bg-purple-900 text-purple-300 border-purple-700' :
                              assayType === 'Functional_IC50' ? 'bg-blue-900 text-blue-300 border-blue-700' :
                              assayType === 'Ki' ? 'bg-orange-900 text-orange-300 border-orange-700' :
                              assayType === 'EC50' ? 'bg-green-900 text-green-300 border-green-700' :
                              'bg-gray-900 text-gray-300 border-gray-700'
                            }`}
                            title={
                              assayType === 'Binding_IC50' ? 
                                'Binding IC50: Measures direct binding affinity to the target protein. Represents the concentration required to inhibit 50% of specific ligand-receptor binding in a binding assay.' :
                              assayType === 'Functional_IC50' ? 
                                'Functional IC50: Measures inhibition of protein function or enzymatic activity. Represents the concentration required to inhibit 50% of the protein biological function in a functional assay.' :
                              assayType === 'Ki' ? 
                                'Ki (Inhibition Constant): Measures the equilibrium binding affinity between inhibitor and target. Represents the dissociation constant for inhibitor binding. Lower Ki = stronger binding.' :
                              assayType === 'EC50' ? 
                                'EC50 (Half Maximal Effective Concentration): Measures the concentration producing 50% of maximal biological response. Represents functional potency in cell-based or biochemical assays.' :
                              'Other assay type'
                            }
                          >
                            {assayType === 'Binding_IC50' ? 'Binding IC₅₀' : 
                             assayType === 'Functional_IC50' ? 'Functional IC₅₀' : 
                             assayType === 'Ki' ? 'Ki' :
                             assayType === 'EC50' ? 'EC₅₀' :
                             assayType}
                          </div>
                        </td>
                      </tr>
                    );
                  }).filter(Boolean);
                }).flat()}
              </tbody>
            </table>
          </div>

          {/* Summary Stats */}
          <div className="p-4 bg-gray-750 border-t border-gray-700">
            <div className="flex justify-between text-sm text-gray-400">
              <span>Total Targets: {Object.keys(targetPredictions).length}</span>
              <span>Total Predictions: {model_info.num_total_predictions || Object.keys(targetPredictions).length * 3}</span>
              <span>Model: {model_info.name} (R² = {model_info.r2_score?.toFixed(3)})</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">
          🧬 Ligand Activity Predictor
        </h1>
        <p className="text-gray-300 max-w-2xl mx-auto">
          Predict binding affinity for oncoproteins and tumor suppressors using Gnosis I
        </p>
        
        {/* Model Info */}
        {modelInfo && modelInfo.available && (
          <div className="mt-4 inline-flex items-center px-4 py-2 bg-green-900 text-green-300 border border-green-700 rounded-full text-sm">
            <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
            Gnosis I • R² = {modelInfo.r2_score?.toFixed(3)} • {modelInfo.num_targets} targets
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-900 border border-red-700 text-red-300 rounded-lg">
          {error}
        </div>
      )}

      {/* Input Section */}
      <div className="bg-gray-800 border border-gray-700 p-6 rounded-lg shadow-lg mb-6">
        <div className="space-y-6">
          {/* SMILES Input */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              SMILES String *
            </label>
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            />
            
            {/* SMILES Examples */}
            <div className="mt-3">
              <label className="block text-xs font-medium text-gray-400 mb-2">
                📚 Click examples to try:
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {[
                  { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O', type: 'Anti-inflammatory' },
                  { name: 'Imatinib', smiles: 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C', type: 'Cancer Drug' },
                  { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', type: 'Stimulant' },
                  { name: 'Paracetamol', smiles: 'CC(=O)Nc1ccc(O)cc1', type: 'Analgesic' },
                  { name: 'Penicillin', smiles: 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C', type: 'Antibiotic' },
                  { name: 'Ethanol', smiles: 'CCO', type: 'Simple Alcohol' }
                ].map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setSmiles(example.smiles)}
                    className="text-left p-2 bg-gray-600 hover:bg-gray-500 border border-gray-500 rounded-md transition-colors text-sm"
                  >
                    <div className="font-medium text-white">{example.name}</div>
                    <div className="text-xs text-gray-300">{example.type}</div>
                    <div className="text-xs text-gray-400 font-mono mt-1 truncate">
                      {example.smiles.slice(0, 20)}...
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Dynamic Assay Type Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Select Assay Types - {availableTargetsFiltered?.available_targets?.length || 0} targets available (union of selected assays)
            </label>
            <div className="grid grid-cols-2 gap-4">
              {['IC50', 'EC50'].map(assayType => {
                const isSelected = selectedAssayTypes.includes(assayType);
                
                // Calculate how many targets have this specific assay type available
                const targetsWithAssay = availableTargets?.available_targets?.filter(target => {
                  const targetTraining = trainingData[target];
                  if (!targetTraining) return false;
                  const assayData = targetTraining[assayType];
                  return assayData && assayData.available;
                }).length || 0;
                
                return (
                  <label key={assayType} className={`flex items-center p-3 rounded-lg border-2 cursor-pointer transition-all ${
                    isSelected 
                      ? 'bg-blue-900 border-blue-500 text-blue-200' 
                      : 'bg-gray-700 border-gray-500 text-gray-300 hover:bg-gray-600'
                  }`}>
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => handleAssayTypeChange(assayType)}
                      className="w-4 h-4 mr-3 text-blue-600 bg-gray-600 border-gray-500 rounded focus:ring-blue-500"
                    />
                    <div className="flex-1">
                      <div className="font-medium">
                        {assayType === 'IC50' ? 'IC₅₀' : assayType}
                      </div>
                      <div className="text-xs opacity-75">
                        {assayType === 'IC50' ? 'Binding & Functional' : 'Functional potency'}
                      </div>
                      <div className="text-xs mt-1 text-blue-300">
                        {targetsWithAssay} targets available
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
            <div className="mt-2 p-2 bg-gray-800 rounded-lg">
              <div className="text-xs text-gray-400">
                {availableTargetsFiltered?.available_targets?.length || 0} targets available with selected assay types
              </div>
            </div>
          </div>

          {/* Target Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Target Selection ({selectedTargets.includes('all') ? 'All' : selectedTargets.length} selected)
            </label>
            {renderTargetSelection()}
          </div>

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={isLoading || !smiles.trim() || !modelInfo?.available}
            className="w-full bg-gradient-to-r from-purple-600 to-cyan-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                {selectedTargets.includes('all') ? (
                  <span>
                    Predicting {availableTargets.total_targets || 62} targets × 3 assays... 
                    <br className="sm:hidden" />
                    <span className="text-sm opacity-90">(~1-2 minutes)</span>
                  </span>
                ) : selectedTargets.length > 10 ? (
                  <span>
                    Predicting {selectedTargets.length} targets × 3 assays...
                    <br className="sm:hidden" />
                    <span className="text-sm opacity-90">(~30-60 seconds)</span>
                  </span>
                ) : (
                  <span>Predicting...</span>
                )}
              </div>
            ) : (
              `🔬 Predict with Gnosis I`
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {renderResults()}
    </div>
  );
};

export default LigandActivityPredictor;
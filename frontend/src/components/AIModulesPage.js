import React, { useState, useEffect } from 'react';
import LigandActivityPredictor from './LigandActivityPredictor';
import CytotoxicityPredictionModel from './CytotoxicityPredictionModel';

const AIModulesPage = () => {
  const [activeModule, setActiveModule] = useState('ligand-predictor');
  
  // Available AI Modules
  const modules = [
    {
      id: 'ligand-predictor',
      name: 'Ligand Activity Predictor',
      description: 'Multi-model prediction system for IC50, toxicity, LogP, and solubility',
      icon: '🧬',
      models: ['ChemBERTa Multi-Task', 'Chemprop Multi-Task', 'Enhanced RDKit'],
      status: 'active'
    },
    {
      id: 'cytotoxicity-predictor',
      name: 'Cytotoxicity Prediction Model',
      description: 'ChemBERTa-based multi-modal drug sensitivity prediction for cancer cell lines',
      icon: '🦠',
      models: ['ChemBERTa + Genomics', 'Neural Network (A100 GPU Trained)', '74K Samples Dataset'],
      status: 'active'
    },
    {
      id: 'molecular-analysis',
      name: 'Molecular Structure Analysis',
      description: 'Advanced molecular property analysis and visualization',
      icon: '⚛️',
      models: ['RDKit Descriptors', 'Quantum Calculations'],
      status: 'coming-soon'
    },
    {
      id: 'drug-design',
      name: 'AI-Driven Drug Design',
      description: 'Generative models for novel compound discovery',
      icon: '💊',
      models: ['Generative Chemistry', 'Optimization Algorithms'],
      status: 'coming-soon'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            AI Modules
          </h1>
          <p className="text-xl text-gray-300">
            Advanced artificial intelligence modules for molecular discovery and analysis
          </p>
        </div>

        {/* Module Selection */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {modules.map((module) => (
            <div
              key={module.id}
              className={`bg-gray-800 border rounded-xl p-6 cursor-pointer transition-all ${
                activeModule === module.id
                  ? 'border-purple-500 ring-2 ring-purple-500/20'
                  : 'border-gray-700 hover:border-gray-600'
              } ${module.status === 'coming-soon' ? 'opacity-50' : ''}`}
              onClick={() => module.status === 'active' && setActiveModule(module.id)}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="text-3xl">{module.icon}</div>
                {module.status === 'active' && (
                  <div className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-900 text-green-300 border border-green-700">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></div>
                    Active
                  </div>
                )}
                {module.status === 'coming-soon' && (
                  <div className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-900 text-yellow-300 border border-yellow-700">
                    Coming Soon
                  </div>
                )}
              </div>
              
              <h3 className="text-lg font-semibold mb-2 text-white">
                {module.name}
              </h3>
              <p className="text-sm text-gray-400 mb-4">
                {module.description}
              </p>
              
              <div className="space-y-1">
                <div className="text-xs text-gray-500 font-medium">AI Models:</div>
                {module.models.map((model, index) => (
                  <div key={index} className="text-xs text-gray-300 flex items-center">
                    <div className="w-1 h-1 bg-purple-400 rounded-full mr-2"></div>
                    {model}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Module Content */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl">
          {activeModule === 'ligand-predictor' && <LigandActivityPredictor />}
          
          {activeModule === 'cytotoxicity-predictor' && <CytotoxicityPredictionModel />}
          
          {activeModule === 'molecular-analysis' && (
            <div className="p-12 text-center">
              <div className="text-6xl mb-4">⚛️</div>
              <h3 className="text-2xl font-bold text-gray-300 mb-4">
                Molecular Structure Analysis
              </h3>
              <p className="text-gray-400 mb-6">
                Advanced molecular property analysis and quantum calculations coming soon
              </p>
              <div className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-yellow-900 text-yellow-300 border border-yellow-700">
                🚧 Under Development
              </div>
            </div>
          )}
          
          {activeModule === 'drug-design' && (
            <div className="p-12 text-center">
              <div className="text-6xl mb-4">💊</div>
              <h3 className="text-2xl font-bold text-gray-300 mb-4">
                AI-Driven Drug Design
              </h3>
              <p className="text-gray-400 mb-6">
                Generative models for novel compound discovery and optimization
              </p>
              <div className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-yellow-900 text-yellow-300 border border-yellow-700">
                🚧 Under Development
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIModulesPage;
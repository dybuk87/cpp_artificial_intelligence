/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Network.cpp
 * Author: arturduch
 * 
 * Created on 8 listopada 2018, 13:40
 */

#include "Network.h"


#include <iostream>

Network::Network(int _inputCount, const std::vector<LayerMeta>& _layers) {
    this->inputCount = _inputCount;
    
    int countInputs = _inputCount;
    int countSumOutputs = 0;
    int countParams = 0;
    
    int prevCount = _inputCount;
    
    for(int i=0; i<_layers.size(); i++) {
        const LayerMeta& layer = _layers[i];
        countInputs += layer.getNeuronCount();
        countSumOutputs += layer.getNeuronCount();
 
        countParams += (prevCount + 1) * layer.getNeuronCount();
        
        prevCount = layer.getNeuronCount();
    }
    
    this->layersInputs.reset(new float[countInputs]);
    this->layersSums.reset(new float[countSumOutputs]);
    this->weights.reset(new float[countParams]);
    
    int prevInputCount = _inputCount;
    int inputPos = 0;
    int outputPos = 0;
    int weightPos = 0;
    for(int i=0; i<_layers.size(); i++) {
         const LayerMeta& layerMeta = _layers[i];
         
         std::cout<<layerMeta.getNeuronCount()<<" IN " << inputPos<<" WEIGHTS " <<weightPos <<" SUM " << outputPos <<" OUT "<<(inputPos + prevInputCount) << std::endl;
         
        Layer* layer  = new Layer(layerMeta.getActivationFunction(), layerMeta.getDerivativeFunc(),
                 prevInputCount, this->layersInputs.get() + inputPos,   
                 this->weights.get() + weightPos,              
                 layerMeta.getNeuronCount(), 
                 this->layersSums.get()   + outputPos,
                 this->layersInputs.get() + inputPos + prevInputCount);
         
         this->layers.push_back( std::shared_ptr<Layer>(layer));
         
         output = this->layersInputs.get() + inputPos + prevInputCount; // last layer output is network output
         
         inputPos += prevInputCount;
         outputPos += prevInputCount;
         
         weightPos += (prevInputCount + 1) * layerMeta.getNeuronCount();
         
         prevInputCount = layerMeta.getNeuronCount();
    }

    std::cout<<"Network: "<< countInputs << " " << countSumOutputs << " " << countParams << std::endl;

}

float Network::totalError(float *target) {
    int outputLen = this->layers[this->layers.size() - 1]->getOutputCount();
    
    float total = 0.0f;
    for(int i=0; i<outputLen; i++) {
        float diff = (target[i] - this->output[i]);
        total += diff * diff;
    }
    
    return 0.5f * total;
}

void Network::calculate() {
    for(int i=0; i<this->layers.size(); i++) {
        this->layers[i]->calculate();
    }
}
    
float* Network::getInput() const {
    return this->layersInputs.get();
}
    
float* Network::getOutput() const {
    return this->output;
}

float* Network::getWeights() const {
    return weights.get();
}

void Network::backprop(float *target, float roScale) {
    int outputLen = this->layers[this->layers.size() - 1]->getOutputCount();
    
    for(int i=0; i<outputLen; i++) {
        this->output[i] = -(target[i] - this->output[i]);
    }
    
    for(int i=layers.size() - 1; i>=0; i--) {
        this->layers[i]->backprop(roScale);
    }
}
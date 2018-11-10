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
#include "DenseLayer.h"
#include "DropoutLayer.h"
#include <iostream>

class NetStructAnalizer : public LayerMetaVisitor {
public:
    void analize(int _inputCount, const std::vector<std::shared_ptr<LayerMeta> >& _layers);
    virtual void visit(DenseLayerMeta& dense);
    virtual void visit(DropoutLayerMeta& dropout);
    
    int allInputsCount() const;
    int allOutputCount() const;
    int allParamCount() const;
    
private:       
    int countInputs;
    int countSumOutputs;
    int countParams;
    int prevCount;       
};

class NetBuilderVisitor: public LayerMetaVisitor {
public:    
    NetBuilderVisitor(
            std::vector<std::shared_ptr<Layer> > &_layers,
            std::unique_ptr<float[]> &_layersInputs,                        
            std::unique_ptr<float[]> &_layersSums,    
            std::unique_ptr<float[]> &_weights) 
            :  layers(_layers), layersInputs(_layersInputs), layersSums(_layersSums), weights(_weights) {}
    
    void build(int _inputCount, const std::vector<std::shared_ptr<LayerMeta> >& _layers);    
    virtual void visit(DenseLayerMeta& dense);
    virtual void visit(DropoutLayerMeta& dropout); 
    
    float* getOutput();
    
private:
    float* prevInput;
    
    int prevInputCount;
    int inputPos;
    int outputPos;
    int weightPos;
    
    float* output;
    
    std::vector<std::shared_ptr<Layer> >& layers;                
    std::unique_ptr<float[]>& layersInputs;
    
    std::unique_ptr<float[]>& layersSums;
    
    std::unique_ptr<float[]>& weights;   
};

Network::Network(int _inputCount, const std::vector<std::shared_ptr<LayerMeta> >& _layers) {
    this->inputCount = _inputCount;
    
    NetStructAnalizer netStructAnalizer;    
    netStructAnalizer.analize(_inputCount, _layers);
    
    this->layersInputs.reset(new float[netStructAnalizer.allInputsCount()]);
    this->layersSums.reset(new float[netStructAnalizer.allOutputCount()]);
    this->weights.reset(new float[netStructAnalizer.allParamCount()]);
    
    NetBuilderVisitor netBuilder(this->layers, this->layersInputs, this->layersSums, this->weights);
    
    netBuilder.build(_inputCount, _layers);
    
    this->output = netBuilder.getOutput();

    std::cout<<"Network: "<< netStructAnalizer.allInputsCount()
            << " " << netStructAnalizer.allOutputCount() 
            << " " << netStructAnalizer.allParamCount() << std::endl;

}

float Network::totalError(float *target) const {
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

void Network::disableDropouts() {
    for(int i=0; i<this->layers.size(); i++) {
        DropoutLayer* dropout = dynamic_cast<DropoutLayer*>(this->layers[i].get());
        if (dropout) {
            dropout->disable();
        }
    }
}
    
void Network::enableDropouts() {
    for(int i=0; i<this->layers.size(); i++) {
        DropoutLayer* dropout = dynamic_cast<DropoutLayer*>(this->layers[i].get());
        if (dropout) {
            dropout->enable();
        }
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


void NetStructAnalizer::analize(int _inputCount, const std::vector<std::shared_ptr<LayerMeta> >& _layers) {
        countInputs = _inputCount;        
        countSumOutputs = 0;
        countParams = 0;
        prevCount = _inputCount;
        
        for(int i=0; i<_layers.size(); i++) {
            _layers[i]->accept(*this);
        }
        
}
    
void NetStructAnalizer::visit(DenseLayerMeta& layer) {
    countInputs += layer.getNeuronCount();
    countSumOutputs += layer.getNeuronCount(); 
    countParams += (prevCount + 1) * layer.getNeuronCount();
    prevCount = layer.getNeuronCount();        
}
    
void NetStructAnalizer::visit(DropoutLayerMeta& dropout) {
}

int NetStructAnalizer::allInputsCount() const {
    return this->countInputs;
}

int NetStructAnalizer::allOutputCount() const {
    return this->countSumOutputs;
}

int NetStructAnalizer::allParamCount() const {
    return this->countParams;
}



void NetBuilderVisitor::build(int _inputCount, const std::vector<std::shared_ptr<LayerMeta> >& _layers) {
    prevInputCount = _inputCount;
    inputPos = 0;
    outputPos = 0;
    weightPos = 0;    
    for(int i=0; i<_layers.size(); i++) {
        _layers[i]->accept(*this);
    }
}

void NetBuilderVisitor::visit(DenseLayerMeta& dense) {     
    std::cout<<dense.getNeuronCount()<<" IN " << inputPos<<" WEIGHTS " <<weightPos <<" SUM " << outputPos <<" OUT "<<(inputPos + prevInputCount) << std::endl;
         
    Layer* layer  = new DenseLayer(dense.getActivationFunction(), dense.getDerivativeFunc(),
                prevInputCount, this->layersInputs.get() + inputPos,   
                this->weights.get() + weightPos,              
                dense.getNeuronCount(), 
                this->layersSums.get()   + outputPos,
                this->layersInputs.get() + inputPos + prevInputCount);
         
    this->layers.push_back( std::shared_ptr<Layer>(layer));
         
    output = this->layersInputs.get() + inputPos + prevInputCount; // last layer output is network output
      
    prevInput = this->layersInputs.get() + inputPos; //this layer input - used by dropout
    
    inputPos += prevInputCount;
    outputPos += prevInputCount;         
    weightPos += (prevInputCount + 1) * dense.getNeuronCount();
         
    prevInputCount = dense.getNeuronCount();

}

void NetBuilderVisitor::visit(DropoutLayerMeta& dropout) {        
    this->layers.push_back(std::shared_ptr<Layer>(new DropoutLayer(0.2f, prevInputCount, prevInput)));
}

float* NetBuilderVisitor::getOutput() {
    return this->output;
}
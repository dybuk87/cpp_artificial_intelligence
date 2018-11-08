/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   NeuralNetBuilder.h
 * Author: arturduch
 *
 * Created on 5 listopada 2018, 16:04
 */

#ifndef NEURALNETBUILDER_H
#define NEURALNETBUILDER_H

#include <vector>
#include <memory>

#include "activ_func.h"

class LayerMeta;
class Network;

class NeuralNetBuilder {
public:
    NeuralNetBuilder(int _inputCount);
    
    NeuralNetBuilder(const NeuralNetBuilder& orig);
    
    void addDenseLayer(int neuronCount, ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc);
    
    void summary() const;
    
    Network* build() const;
    
    virtual ~NeuralNetBuilder();
private:
    int inputCount;
    
    std::vector<LayerMeta> layers;
};




#endif /* NEURALNETBUILDER_H */


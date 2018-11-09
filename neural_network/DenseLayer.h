/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DenseLayer.h
 * Author: arturduch
 *
 * Created on 9 listopada 2018, 11:21
 */

#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Layer.h"

class DenseLayer : public Layer {
public:
    DenseLayer(ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc, 
            int _inputCount, float* const _input,
            float * const _weights,
            int _outputCount, float* const _sum, float* const _output);
    
    virtual void calculate();
    
    virtual void backprop(float ro);
    
    virtual int getOutputCount() const;
   
    virtual ~DenseLayer();
    
private:
    ActivationFunc activationFunction;
    DerivativeFunc derivativeFunc;
    
    int inputCount; 
    float* const input;
    
    float* const weights;
    
    std::unique_ptr<float[]> oldWeights;
    
    int outputCount;
    float* const sum;
    float* const output;
};

#endif /* DENSELAYER_H */


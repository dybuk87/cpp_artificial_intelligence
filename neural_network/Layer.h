/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Layer.h
 * Author: arturduch
 *
 * Created on 8 listopada 2018, 13:37
 */

#ifndef LAYER_H
#define LAYER_H

#include "activ_func.h"
#include <memory>

class Layer {
public:
    Layer(ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc, 
            int _inputCount, float* _input,
            float *_weights,
            int _outputCount, float* _sum, float* _output);
    
    void calculate();
    
    void backprop(float ro);
    
    int getOutputCount();
   
    ~Layer();
    
private:
    ActivationFunc activationFunction;
    DerivativeFunc derivativeFunc;
    
    int inputCount; 
    float* input;
    
    float* weights;
    
    std::unique_ptr<float> oldWeights;
    
    int outputCount;
    float* sum;
    float* output;
};

#endif /* LAYER_H */


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LayerMeta.cpp
 * Author: arturduch
 * 
 * Created on 8 listopada 2018, 13:35
 */

#include "LayerMeta.h"

LayerMeta::LayerMeta(int _neuronCount, ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc):
    neuronCount(_neuronCount), activationFunction(_activationFunction), derivativeFunc(_derivativeFunc)
{
    
}

int LayerMeta::getNeuronCount() const {
    return neuronCount;
}

LayerMeta::~LayerMeta() {
}

ActivationFunc LayerMeta::getActivationFunction() const {
    return activationFunction;
}
    
DerivativeFunc  LayerMeta::getDerivativeFunc() const {
    return derivativeFunc;
}

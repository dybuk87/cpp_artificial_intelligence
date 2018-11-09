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

 LayerMeta::LayerMeta(const char *_name) : 
   name(std::unique_ptr<char[]>(new char[strlen(_name) + 1])) {
     memset(name.get(), 0, strlen(_name) + 1);
     memcpy(name.get(), _name, strlen(_name));
     
 }
 
const char*  LayerMeta::getName() const {
    return name.get();
}

LayerMeta::~LayerMeta() {
    
}

DenseLayerMeta::DenseLayerMeta(int _neuronCount, ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc):
    LayerMeta("Dense"), neuronCount(_neuronCount), activationFunction(_activationFunction), derivativeFunc(_derivativeFunc)
{
    
}

int DenseLayerMeta::getNeuronCount() const {
    return neuronCount;
}

void DenseLayerMeta::accept(LayerMetaVisitor&) {
    
}

DenseLayerMeta::~DenseLayerMeta() {
}

ActivationFunc DenseLayerMeta::getActivationFunction() const {
    return activationFunction;
}
    
DerivativeFunc  DenseLayerMeta::getDerivativeFunc() const {
    return derivativeFunc;
}


DropoutLayerMeta::DropoutLayerMeta(LayerMeta& _previous, float _ratio) : 
    LayerMeta("Dropout"),
    previous(_previous), ratio(_ratio) {
    
}

int DropoutLayerMeta::getNeuronCount() const {
    return previous.getNeuronCount();
}

void DropoutLayerMeta::accept(LayerMetaVisitor&) {
    
}

DropoutLayerMeta::~DropoutLayerMeta() {

}
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   NeuralNetBuilder.cpp
 * Author: arturduch
 * 
 * Created on 5 listopada 2018, 16:04
 */

#include "NeuralNetBuilder.h"
#include <iostream>
#include <memory>
#include <utility>
#include <cmath>
#include <algorithm>

#include "LayerMeta.h"
#include "Network.h"


NeuralNetBuilder::NeuralNetBuilder(int _inputCount) : inputCount(_inputCount) {
}

NeuralNetBuilder::NeuralNetBuilder(const NeuralNetBuilder& orig) {
}

NeuralNetBuilder::~NeuralNetBuilder() {
}

 void NeuralNetBuilder::addDenseLayer(int neuronCount, ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc) {
     layers.push_back(LayerMeta(neuronCount, _activationFunction, _derivativeFunc));
 }

 void NeuralNetBuilder::summary() const {
     std::cout<<"Net summary"<<std::endl;
     std::cout<<"    Input Size: " << inputCount<<std::endl;
     
     int prevSize = inputCount;
     for(int i=0; i<layers.size(); i++) {
         const LayerMeta& layer = layers[i];
         std::cout<<"    "<<"DENSE "<<layer.getNeuronCount()<<" parms: " <<(prevSize + 1) * layer.getNeuronCount()<<std::endl;
         prevSize = layer.getNeuronCount();
     }
 }




Network* NeuralNetBuilder::build() const {
    return new Network(this->inputCount, this->layers);
}

Network::~Network() {
  //  std::cout<<"Network release"<<std::endl;
}
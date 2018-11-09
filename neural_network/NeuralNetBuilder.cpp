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
#include "DropoutLayer.h"


NeuralNetBuilder::NeuralNetBuilder(int _inputCount) : inputCount(_inputCount) {
}

NeuralNetBuilder::NeuralNetBuilder(const NeuralNetBuilder& orig) {
}

NeuralNetBuilder::~NeuralNetBuilder() {
}

 void NeuralNetBuilder::addDenseLayer(int neuronCount, ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc) {
     layers.push_back(std::shared_ptr<LayerMeta> (new DenseLayerMeta(neuronCount, _activationFunction, _derivativeFunc)) );
 }
 
 void NeuralNetBuilder::addDropout(float rate) {
     layers.push_back(std::shared_ptr<LayerMeta> (new DropoutLayerMeta(*layers[layers.size() - 1].get(), rate)));
 }

 void NeuralNetBuilder::summary() const {
     std::cout<<std::endl<<"-----------------------------------------" << std::endl;
     std::cout<<"Net summary"<<std::endl;
     std::cout<<"    Input Size: " << inputCount<<std::endl;
     
     int totalParams = 0;
     
     int prevSize = inputCount;
     for(int i=0; i<layers.size(); i++) {
         const std::shared_ptr<LayerMeta>& layer = layers[i];
         
         int paramCount = (prevSize + 1) * layer->getNeuronCount();
         
         DropoutLayerMeta* dropout = dynamic_cast<DropoutLayerMeta*>(layer.get());
         if (dropout) {
             paramCount = 0;
         }
         
         totalParams += paramCount;
         
         std::cout<<"    "<<layer->getName()<<" : "<<layer->getNeuronCount()<<" parms: " <<paramCount<<std::endl;
         prevSize = layer->getNeuronCount();
     }
     std::cout<<"Total params : " << totalParams << std::endl;
     std::cout<<"-----------------------------------------" << std::endl << std::endl;
 }




Network* NeuralNetBuilder::build() const {
    return new Network(this->inputCount, this->layers);
}

Network::~Network() {
  //  std::cout<<"Network release"<<std::endl;
}
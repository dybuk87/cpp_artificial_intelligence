/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Network.h
 * Author: arturduch
 *
 * Created on 8 listopada 2018, 13:40
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "LayerMeta.h"
#include "Layer.h"

class Network {
public:
    Network(int inputCount, const std::vector<LayerMeta>& layers);
    
    void calculate();
    
    void backprop(float *target, float roScale);
    
    float totalError(float *target);
    
    float* getInput() const ;
    
    float* getOutput() const ;
    
    float* getWeights() const;
    
    ~Network();
private:
    std::vector<std::shared_ptr<Layer> > layers;
    
    int inputCount;
    std::unique_ptr<float> layersInputs;
    
    std::unique_ptr<float> layersSums;
    
    std::unique_ptr<float> weights;
    
    float *output;
   
};



#endif /* NETWORK_H */


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DropoutLayer.h
 * Author: arturduch
 *
 * Created on 9 listopada 2018, 11:36
 */

#ifndef DROPOUTLAYER_H
#define DROPOUTLAYER_H

#include "Layer.h"

class DropoutLayer : public Layer {
public:
    DropoutLayer(float _ratio, int _inputCount, float * const _input);
    
    virtual void calculate();
    
    virtual void backprop(float ro);
    
    virtual int getOutputCount() const ;
   
    virtual ~DropoutLayer();
private:
    int inputCount; 
    float * const input;
    std::unique_ptr<float[]> dropout;
};

#endif /* DROPOUTLAYER_H */


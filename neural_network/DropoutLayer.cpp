/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DropoutLayer.cpp
 * Author: arturduch
 * 
 * Created on 9 listopada 2018, 11:36
 */

#include "DropoutLayer.h"
#include <algorithm> 

DropoutLayer::DropoutLayer(float _ratio, int _inputCount, float * const _input) : 
    Layer("Dropout"), inputCount(_inputCount), input(_input), dropout(new float[_inputCount]) {
    
    bool enabled = true;
    
    int dropCount = (int)(inputCount * _ratio);
    dropCount = std::min(std::max(0, dropCount), inputCount);
    
    for(int i=0; i<dropCount; i++) {
        dropout[i] = 0.0f;
    }
    
    for(int i=dropCount; i<inputCount; i++) {
        dropout[i] = 1.0f;
    }    
}

void DropoutLayer::enable() {
    this->enabled = true;
}
    
void DropoutLayer::disable() {
    this->enabled = false;
}

void DropoutLayer::calculate() {
    if (this->enabled) {
        std::random_shuffle(dropout.get(), dropout.get() + inputCount);

        for(int i=0; i<inputCount; i++) {
            input[i] *= dropout[i];
        }
    }
}

void DropoutLayer::backprop(float ro) {
    
}
    
int DropoutLayer::getOutputCount() const {
    return inputCount;
}
   
DropoutLayer::~DropoutLayer() {
    
}



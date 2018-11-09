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
#include <iostream>

// TODO: Dropout, Convolution, MaxPool

class Layer {
public:
    Layer(const char *name);
    
    virtual void calculate() = 0;
    
    virtual void backprop(float ro) = 0;
    
    virtual int getOutputCount() const = 0;
    
    const char* getName() const;
   
    virtual ~Layer();
    
private:
    const std::unique_ptr<char[]> name;
   
};

#endif /* LAYER_H */


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   PoolingLayer.h
 * Author: arturduch
 *
 * Created on 9 listopada 2018, 11:30
 */

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "Layer.h"

class PoolingLayer : public Layer {
public:
    PoolingLayer();
    
    virtual void calculate();
    
    virtual void backprop(float ro);
    
    virtual int getOutputCount() const;
   
    virtual ~PoolingLayer();
private:

};

#endif /* POOLINGLAYER_H */


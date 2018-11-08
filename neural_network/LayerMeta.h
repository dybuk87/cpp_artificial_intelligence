/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LayerMeta.h
 * Author: arturduch
 *
 * Created on 8 listopada 2018, 13:35
 */

#ifndef LAYERMETA_H
#define LAYERMETA_H

#include "activ_func.h"

class LayerMeta {
public:
    LayerMeta(int neuronCount, ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc);
    
    int getNeuronCount() const;
    
    ActivationFunc getActivationFunction() const;
    
    DerivativeFunc getDerivativeFunc() const;
    
    virtual ~LayerMeta();
private:
    int neuronCount;
    ActivationFunc activationFunction;
    DerivativeFunc derivativeFunc;
};


#endif /* LAYERMETA_H */


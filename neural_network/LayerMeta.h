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
#include <iostream>
#include <memory>
#include <algorithm>

class LayerMetaVisitor;

class LayerMeta {
public:
    LayerMeta(const char *name);
    
    const char* getName() const;
    
    virtual void accept(LayerMetaVisitor &) = 0;
    
    virtual int getNeuronCount() const = 0;
    
    virtual ~LayerMeta();
 
private:
    const std::unique_ptr<char[]> name;
   
};

class DenseLayerMeta : public LayerMeta {
public:
    DenseLayerMeta(int neuronCount, ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc);
    
    ActivationFunc getActivationFunction() const;
    
    DerivativeFunc getDerivativeFunc() const;
    
    virtual int getNeuronCount() const ;
    
    virtual void accept(LayerMetaVisitor &);
    
    virtual ~DenseLayerMeta();
private:
    int neuronCount;
    ActivationFunc activationFunction;
    DerivativeFunc derivativeFunc;
};

class DropoutLayerMeta : public LayerMeta {
public:
    DropoutLayerMeta(LayerMeta& _previous, float _ratio);
    
    virtual int getNeuronCount() const;
    
    virtual void accept(LayerMetaVisitor &);
    
    virtual ~DropoutLayerMeta();
private:
    float ratio;
    LayerMeta& previous;
};

class LayerMetaVisitor  {
public:
    virtual void visit(DenseLayerMeta& dense) = 0;
    virtual void visit(DropoutLayerMeta& dropout) = 0;
};

#endif /* LAYERMETA_H */


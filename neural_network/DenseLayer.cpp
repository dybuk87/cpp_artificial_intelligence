/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DenseLayer.cpp
 * Author: arturduch
 * 
 * Created on 9 listopada 2018, 11:21
 */

#include "DenseLayer.h"
#include <cmath>
#include <cstring>

DenseLayer::DenseLayer(ActivationFunc _activationFunction, DerivativeFunc _derivativeFunc, 
            int _inputCount, float* _input,
            float *_weights,
            int _outputCount, float* _sum, float* _output) : 
    Layer("Dense"),
    activationFunction(_activationFunction), derivativeFunc(_derivativeFunc),
        inputCount(_inputCount), input(_input), 
        weights(_weights),
        outputCount(_outputCount), sum(_sum), output(_output), oldWeights(new float[(_inputCount + 1) * _outputCount])
{
    for(int i=0; i<(_inputCount + 1) * _outputCount; i++) {
        weights[i] = ((float)(rand()%1000)) / 1000.0f;
    }
}

int DenseLayer::getOutputCount() const {
    return outputCount;
}

void DenseLayer::backprop(float roScale) {
    memcpy(oldWeights.get(), weights, (inputCount + 1) * outputCount * sizeof(float));
    
   // std::cout<<"LAYER"<<std::endl;
    
    for(int neuronIter = 0; neuronIter<outputCount; neuronIter++) {
          int weightOffset = (inputCount + 1) * neuronIter;
        
          // backup neuron weights

        float dErrordOut = output[neuronIter]; 
        float dOutdNet = derivativeFunc(sum[neuronIter]); 
        float dNetdBias = 1.0f;
        
        float dErrordBias = dErrordOut * dOutdNet * dNetdBias; 
        
        float ro = 0.5f * roScale;
        
     //   std::cout<<"TRAIN BIAS "<< neuronIter<< " : "<< dErrordOut <<" * " << dOutdNet <<" * "<<dNetdBias<<" = "<<dErrordBias<<std::endl;
    
      // train weights
        weights[weightOffset + 0] -= ro * dErrordBias; // bias fix  
        for(int inputIter = 0; inputIter < inputCount; inputIter++) {
            float dNetdWi = input[inputIter];
            float dErrordWi = dErrordOut * dOutdNet * dNetdWi;
            
      //      std::cout<<"TRAIN " << neuronIter<< " : "<< dErrordOut <<" * " << dOutdNet <<" * "<<dNetdWi<<" = "<<dErrordWi<<std::endl;
            
      //      std::cout<<"APPLY " << weights[weightOffset + 1 + inputIter] << " -  " << ro * dErrordWi << " = " << (weights[weightOffset + 1 + inputIter] - ro * dErrordWi)<<std::endl;
            
            weights[weightOffset + 1 + inputIter] -= ro * dErrordWi; 
        }
    }
          // push error
  //  std::cout<<"PUSH ERROR"<<std::endl;
    for(int inputIter = 0; inputIter < inputCount; inputIter++) {
        float sumVal = 0.0f;
        for(int neuronIter = 0; neuronIter<outputCount; neuronIter++) {
            float weight = oldWeights.get()[(inputCount + 1) * neuronIter + 1 + inputIter];
              float errCorrection = output[neuronIter] * derivativeFunc(sum[neuronIter]) * weight; 
             
         //    std::cout<<"ERROR:"<<output[neuronIter]<<" * "<<derivativeFunc(sum[neuronIter])<<" * "<<weight<<" = " <<errCorrection<<std::endl;
            
          
            sumVal += errCorrection;
        }
      //  std::cout<<"CALC INPUT "<<inputIter<<": " << sumVal<<std::endl;
        input[inputIter] = sumVal;
    }
}

 void DenseLayer::calculate() {
     for(int neuronIter = 0; neuronIter<outputCount; neuronIter++) {
    //     std::cout<<"NEURON "<< neuronIter <<": ";
         int weightOffset = (inputCount + 1) * neuronIter;
         float sumValue = weights[weightOffset + 0] * 1.0f; // bias
   //      std::cout<<"1.0 * " <<weights[weightOffset + 0];
         for(int inputIter = 0; inputIter < inputCount; inputIter++) {
      //       std::cout<<" + " << input[inputIter] <<" * " << weights[weightOffset + 1 + inputIter] ;
             sumValue += weights[weightOffset + 1 + inputIter] * input[inputIter]; 
         }
      //   std::cout<< std::endl;
         sum[neuronIter] = sumValue;
         output[neuronIter] = activationFunction(sumValue);
     }
 }

DenseLayer::~DenseLayer() {
  //  std::cout<<"Free layer"<<std::endl;
}
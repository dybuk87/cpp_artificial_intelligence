/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   activ_func.h
 * Author: arturduch
 *
 * Created on 8 listopada 2018, 13:31
 */

#ifndef ACTIV_FUNC_H
#define ACTIV_FUNC_H

typedef float (*ActivationFunc)(float value);
typedef float (*DerivativeFunc)(float value);


float identity(float value);
float didentity(float value);

float relu(float value);
float drelu(float value);

float logistic(float value);
float dlogistic(float value);


#endif /* ACTIV_FUNC_H */


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "activ_func.h"

#include <cmath>

float identity(float value) {
    return value;
}

float didentity(float value) {
    return 1.0f;
}

float relu(float value) {
    return value < 0.0f ? 0.0f : value;
}

float drelu(float value) {
    return value < 0.0f ? 0.0f : 1.0f;
}

float logistic(float value) {
    return (float)(1.0f / (1.0f + exp(-value)));
}

float dlogistic(float value) {
    return logistic(value)*(1.0f - logistic(value));
}
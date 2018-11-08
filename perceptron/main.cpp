/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: Duch
 *
 * Created on 4 listopada 2018, 11:17
 */

#include <cstdlib>
#include <iostream>
//#include <windows.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
//#include "core/ux_window.h"
//#include "core/ux_bitmap.h"

using namespace std;

const float BIAS = 1.0f;


const int learn_size = 1000;

const int validate_size = 3000;


const int w_width = 375;
const int w_height = 375;

typedef float (*ActivationFunc)(float value);
typedef float (*DerivativeFunc)(float value);

class Perceptron {
public:
    Perceptron(int size, ActivationFunc activationFunc, DerivativeFunc derivativeFunc);
    
    float sum(const float *input) const;
    
    float train(const float *low, const float* high, int count, float roScale);
    
    float train(const float *value, const float expected, float roScale);
    
    float calculate(const float *sum) const;
    
    ~Perceptron();
private:    
    float* const weight;
    const int size; 
    const ActivationFunc activationFunc;
    const DerivativeFunc derivativeFunc;
};

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

float testFunc(float x) {
    return 1.23 * x - 0.5;
}

float* generateHigh(float (*testFunc)(float), int count, float minX, float minY, float maxX, float maxY) { 
    float* result = new float[count * 2];
    int i = 0;
    
    while(i < count) {
        float rX = (float)(random()%1000)/1000.0f;
        float x = ((maxX - minX) * rX) + minX;  // x between minX and maxX;
        
        float funcY = testFunc(x);              // get function Y for this value;
        
        if (funcY <= maxY) {                     // if funcY >= minY we are ale to random value lower than f(x)
            float mY = max(funcY, minY);
            
            float rY = (float)(random()%1000)/1000.0f;
            float y = ((maxY - mY) * rY) + mY;  // y between minY and maxY;
            
            result[i * 2 + 0] = x;
            result[i * 2 + 1] = y;
            
            i++;
            
     //       printf("Rand high %6.3f %6.3f > %6.3f\n", x, y, funcY);
            
       
        }
    }
    
    return result;
}


float* generateLow(float (*testFunc)(float), int count, float minX, float minY, float maxX, float maxY) { 
    float* result = new float[count * 2];
    int i = 0;
    
    while(i < count) {
        float rX = (float)(random()%1000)/1000.0f;
        float x = ((maxX - minX) * rX) + minX;  // x between minX and maxX;
        
        float funcY = testFunc(x);              // get function Y for this value;
        
        if (funcY >= minY) {                     // if funcY >= minY we are ale to random value lower than f(x)
            float mY = min(funcY, maxY);
            
            float rY = (float)(random()%1000)/1000.0f;
            float y = ((mY - minY) * rY) + minY;  // y between minY and maxY;
            
            result[i * 2 + 0] = x;
            result[i * 2 + 1] = y;
            
            i++;
            
     //       printf("Rand low %6.3f %6.3f < %6.3f\n", x, y, funcY);
            
       
        }
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    
    float* low = generateLow(testFunc, learn_size, -1.0f, -1.0f, 1.0f, 1.0f);
    float* high = generateHigh(testFunc, learn_size, -1.0f, -1.0f, 1.0f, 1.0f);
    
    // validation set has bigger range - neural network should assume valid results
    float* validateSetLow = generateLow(testFunc, validate_size, -3.0f, -3.0f, 3.0f, 3.0f);
    float* validateSetHigh = generateHigh(testFunc, validate_size, -3.0f, -3.0f, 3.0f, 3.0f);
    
    Perceptron p(2, identity, didentity);
    
    float roScale = 1.0f;
    float prevError = 100000.0f;
    for(int i=0; i<200; i++) {
        float errorAvg = p.train(low, high, learn_size, roScale);
        roScale *= 0.99f;
        roScale = max(roScale, 0.001f);
        printf("ERROR(%d) : %8.4f   %8.5f\n", i, errorAvg, roScale);
    }

    int validCount  = 0;
    
    
    for(int i=0; i<validate_size; i++) {
        validCount += (p.calculate(validateSetLow  + i * 2) < 0.5);
        validCount += (p.calculate(validateSetHigh + i * 2) > 0.5);
    }
    
    printf("Valid percent  %8.3f\n", ((float)validCount / (validate_size * 2)));
    
    delete[] validateSetLow;
    delete[] validateSetHigh;
    
    delete[] low;
    delete[] high;
    
    return 0;
}

float Perceptron::sum(const float *input) const {
    float sum = BIAS * weight[0];
    for(int i=1; i<size; i++) {
        sum += weight[i] * input[i - 1];
    }
    return sum;
}

float Perceptron::calculate(const float *input) const { 
    return activationFunc(sum(input));
}

Perceptron::~Perceptron() {
    delete[] this->weight;
}

Perceptron::Perceptron(int _size, ActivationFunc _activationFunc, DerivativeFunc _derivativeFunc) : 
    activationFunc(_activationFunc), derivativeFunc(_derivativeFunc), size(_size + 1), weight(new float[_size + 1]) {  // +1 bias
    for(int i=0; i<size; i++) {
        this->weight[i] = 0.5f;
    }
}

    
float Perceptron::train(const float *value, const float expected, float roScale) {
    float testValue = calculate(value);  // expected 0.0f
    float error = 0.5 * (expected - testValue) * (expected - testValue);    // error
   //printf("p(%6.5f, %6.5f) = %6.5f expected %6.5f  - error : %6.5f\n",
     //              value[0], value[1], testValue, expected, error);
        
    float dErrordOut = -(expected - testValue);
    float dOutdNet = derivativeFunc(testValue);
    
    float dNetdBias = 1.0f;
    
    float dErrordBias = dErrordOut * dOutdNet * dNetdBias; 
    
    float ro = 0.5f * roScale;
    
    weight[0] -= ro * dErrordBias;
    
    for(int i=1; i<size; i++) {
        float dNetdWi = value[i - 1];
        float dErrordWi = dErrordOut * dOutdNet * dNetdWi; 
        weight[i] -= ro * dErrordWi;
    }
    return error;  
}

float Perceptron::train(const float *low, const float* high, int count, float roScale) {
    float dError = 0.0f;
    for(int i=0; i<count; i++, low += 2, high += 2) {
        dError += train(low, 0.0f, roScale);
        dError += train(high, 1.0f, roScale);
    }
    
    return dError / (2 * count);
}

/*

void onFrame(UxWindow* uxWindow) {
	
	//uxWindow->getBitmap().drawBitmap(xBitmap, xPos, yPos);
}



void init() {
    
    
    UxWindow* uxWindow1 = new UxWindow(mainEnvironment, "Test 1", w_width, w_height, onFrame);
    uxWindow1->show();
	
}

START_PROC(init())*/
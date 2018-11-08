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
#include "core/ux_window.h"
#include "core/ux_bitmap.h"

using namespace std;

const float BIAS = 1.0f;


const int learn_size = 500;

const int validate_size = 3000;


const int w_width = 800;
const int w_height = 800;

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

float logistic(float value) {
    return (float)(1.0f / (1.0f + exp(-value)));
}

float dlogistic(float value) {
    return logistic(value)*(1.0f - logistic(value));
}

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
    return  1.73 * x - 0.85;
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
    float sumVal = sum(value);
    float testValue = calculate(value);  // expected 0.0f
    float error = 0.5 * (expected - testValue) * (expected - testValue);    // error
   //printf("p(%6.5f, %6.5f) = %6.5f expected %6.5f  - error : %6.5f\n",
     //              value[0], value[1], testValue, expected, error);
        
    float dErrordOut = -(expected - testValue);
    float dOutdNet = derivativeFunc(sumVal);
    
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

UxBitmap bitmap(w_width, w_height, ARGB_8888);

void onFrame(UxWindow* uxWindow) {    
    
    uxWindow->getBitmap().drawBitmap(&bitmap, 0, 0);            
}

Perceptron p(2, logistic, dlogistic);

const int plotSize = 1;

void plot(UxBitmap& bitmap, int x, int y, int color) {   
    int sx = max(0, x - plotSize);
    int sy = max(0, y - plotSize);
    
    int ex = min(bitmap.getWidth(), x + plotSize + 1);
    int ey = min(bitmap.getHeight(), y + plotSize + 1);
          
    for(int yy = sy; yy<ey; yy++) {
        for(int xx = sx; xx<ex; xx++) {
            bitmap.asUInt32()[xx + yy * bitmap.getWidth()] = color;
        }
    }
}

void init() {
    
     float* low = generateLow(testFunc, learn_size, -1.0f, -1.0f, 1.0f, 1.0f);
    float* high = generateHigh(testFunc, learn_size, -1.0f, -1.0f, 1.0f, 1.0f);
    
    // validation set has bigger range - neural network should assume valid results
    float* validateSetLow = generateLow(testFunc, validate_size, -3.0f, -3.0f, 3.0f, 3.0f);
    float* validateSetHigh = generateHigh(testFunc, validate_size, -3.0f, -3.0f, 3.0f, 3.0f);
    
    
    
    float roScale = 1.0f;
    float prevError = 100000.0f;
    for(int i=0; i<2000; i++) {
        float errorAvg = p.train(low, high, learn_size, roScale);
        roScale *= 0.99999f;
        roScale = max(roScale, 0.001f);
       // printf("ERROR(%d) : %8.4f   %8.5f\n", i, errorAvg, roScale);
    }

        float errorAvg = p.train(low, high, learn_size, roScale);
        printf("ERROR : %8.4f   %8.5f\n", errorAvg, roScale);
    
    int validCount  = 0;
    
    
    for(int i=0; i<validate_size; i++) {
        validCount += (p.calculate(validateSetLow  + i * 2) < 0.5);
        validCount += (p.calculate(validateSetHigh + i * 2) > 0.5);
    }
    
    printf("Valid percent  %8.3f\n", ((float)validCount / (validate_size * 2)));
    
 
    
    uint32* data = bitmap.asUInt32();
    
    for(int y=0; y<bitmap.getHeight(); y++) {
        for(int x=0; x<bitmap.getWidth(); x++) {            
            
            float xx = x;
            float yy = y;
            
            xx /= bitmap.getWidth();  // 0 .. 1
            yy /= bitmap.getHeight();
            
            xx = xx * 6.0f - 3.0f;
            yy = yy * 6.0f - 3.0f;
            
            float in[] = { xx, yy };
            
            //std::cout<<"IN "<<xx<<", "<<yy<<" = "<<p.calculate(in)<<std::endl;
            
           plot(bitmap, x, y, p.calculate(in) > 0.5 ? 0xFF00FF00 : 0xFF000000);            
        }
    }    
    
    for(int i=0; i<learn_size; i++) {
        float xx = low[i * 2 + 0];
        float yy = low[i * 2 + 1];
        
        int x = (int)(bitmap.getWidth()  * (xx + 3.0f) / 6.0);
        int y = (int)(bitmap.getHeight() * (yy + 3.0f) / 6.0);
        
         plot(bitmap, x, y, 0xFFFF0000);
               
    }
    
    for(int i=0; i<learn_size; i++) {
        float xx = high[i * 2 + 0];
        float yy = high[i * 2 + 1];
        
        int x = (int)(bitmap.getWidth()  * (xx + 3.0f) / 6.0);
        int y = (int)(bitmap.getHeight() * (yy + 3.0f) / 6.0);       
        
        plot(bitmap, x, y, 0xFF0000FF);
    }
    
    for(int x=0; x<bitmap.getWidth(); x++) {
        float xx = (float)x  / bitmap.getWidth() ;
        xx = xx * 6.0f - 3.0f;
        
        float yy = testFunc(xx);
        int y = (int)(bitmap.getHeight() * (yy + 3.0f) / 6.0);
        
        y = min(max(0, y), bitmap.getHeight() - 1);
     
        
        plot(bitmap, x, y, 0xFFFFffff);
    }
    
    delete[] validateSetLow;
    delete[] validateSetHigh;
    
    delete[] low;
    delete[] high;   
    
    UxWindow* uxWindow1 = new UxWindow(mainEnvironment, "Test 1", w_width, w_height, onFrame);
    uxWindow1->show();
	
}

START_PROC(init())
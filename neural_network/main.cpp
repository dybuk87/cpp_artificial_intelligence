/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: arturduch
 *
 * Created on 5 listopada 2018, 16:02
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include "DropoutLayer.h"

#include "Network.h"
#include "NeuralNetBuilder.h"

using namespace std;

class Data {
public:
    Data(float x, float y, int classNo);
    
    float* getIn() const;
    
    float* getOut() const;
    
private:
    float in[2];
    float out[4];
};

Data::Data(float x, float y, int classNo) {
    in[0] = x;
    in[1] = y;
    
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    out[classNo] = 1.0f;
}

float* Data::getIn() const {
      return (float*)in;
}
    
float* Data::getOut() const {
        return (float*)out;
}

unique_ptr<Data>* generateData(int setSize, float minX, float minY, float maxX, float maxY, float (*f1)(float), float (*f2)(float));

void buildNetworkTest() {
    NeuralNetBuilder netBuilder(2);
    
    netBuilder.addDenseLayer(20, identity, didentity);
    netBuilder.addDropout(0.2);
    netBuilder.addDenseLayer(15, identity, didentity);
    netBuilder.addDenseLayer(8,  identity, didentity);
    netBuilder.addDenseLayer(4,  relu, drelu);
   
    netBuilder.summary();
    
    unique_ptr<Network> network(netBuilder.build());
    
    network->calculate();
    
    
    std::cout<<"RESULT: "<<network->getOutput()[0]<<std::endl;
}

void testNetwork(unique_ptr<Network>& network, float i1, float i2) {
    // TEST network result for input tensor
    network->getInput()[0] = 0.05f;
    network->getInput()[1] = 0.1f;                     
    network->calculate();
    std::cout<<std::endl;
    std::cout<<"RESULT TEACH: "<<network->getOutput()[0]<<"  "<<network->getOutput()[1]<<std::endl;
    
}

/**
 * Build dens network 2 input, 2 hidden 2 output
 * train network to match (0.05; 0.10) to (0.01; 0.99)
 * each layer use logistic activation function
 */
void testNetworkTraining() {
    
    NeuralNetBuilder netBuilder2(2);
    netBuilder2.addDenseLayer(2, logistic, dlogistic);
    netBuilder2.addDenseLayer(2, logistic, dlogistic);
    
    // summary before build
    netBuilder2.summary();
    
    // build network
    unique_ptr<Network> network2(netBuilder2.build());
    
    // set initial weight (this can be randomly set 
    float* w = network2->getWeights();
    w[0] = 0.35f; w[1] = 0.15f; w[2] = 0.20f;  // h1
    w[3] = 0.35f; w[4] = 0.25f; w[5] = 0.30f;  // h2
    
    w[6] = 0.60;  w[7] = 0.40f; w[8] = 0.45f;  // o1
    w[9] = 0.60f; w[10] = 0.50f; w[11] = 0.55f;  // o2
    
    
    // create train tensor
    unique_ptr<float> target(new float[2]);
    target.get()[0] = 0.01f;
    target.get()[1] = 0.99f;
    
    // 40 000 iterations
    for(int i=0; i<40000; i++) {
        // set input
        network2->getInput()[0] = 0.05f;
        network2->getInput()[1] = 0.1f;                     
        network2->calculate();
        // output is in network2->getOutput()[0] and network2->getOutput()[1]
        
        float mod = max(0.0f, (float)i - 30000);
        mod = 1.0f - (mod/20000);  // mod = learning speed, first 30'000 iters equal 1.0 after that mod will decrease
        
        // backpropagation
        network2->backprop(target.get(),  mod);
    }
    
    testNetwork(network2, 0.05f, 0.10f);
    std::cout.precision(15);
    std::cout<<"TOTAL ERROR: " <<std::fixed<< network2->totalError(target.get()) << std::endl<< std::endl;
  
    
    std::cout<<std::endl<<std::endl<<"WEIGHTS: "<<std::endl;
    
     w = network2->getWeights();
    
    for(int i=0; i<6; i++) {
        cout<<" "<<w[i];
    }
    cout<<endl;
    
    for(int i=6; i<12; i++) {
        cout<<" "<<w[i];
    }
    cout<<endl;
}

void testDropout() {
    std::unique_ptr<float[]> input(new float[10]);
    
    for(int i=0; i<10; i++) {
        input[i] = i + 1;
    }
    
    DropoutLayer dropout(0.4f, 10, input.get());
    dropout.calculate();
    
    std::cout.precision(4);
    std::cout<<"DROPOUT: " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout<<input[i]<<" ";
    }
    
    std::cout<<std::endl;
    
    
    
    for(int i=0; i<10; i++) {
        input[i] = i + 1;
    }
    dropout.calculate();
    
    std::cout.precision(4);
    std::cout<<"DROPOUT 2: " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout<<input[i]<<" ";
    }
    
    std::cout<<std::endl;
    
}

const int setSize = 3000;

float f1(float x) {
    return 1.32 * x + 0.5;
}

float f2(float x) {
    return -2.32 * x + 0.5;
}

float fabs(float x) {
    return x > 0.0f ? x : -x;
}

void networkTraining() {
    std::unique_ptr<std::unique_ptr<Data>[]> data(
        generateData(setSize, -1.0f, -1.0f, 1.0f, 1.0f, f1, f2)
    );
  /*  
    for(int i=0; i<setSize; i++) {
        std::cout<<"DATA: "<<i<<": "
                <<data[i]->getIn()[0]<<", " << data[i]->getIn()[1]
                << " = "
                <<data[i]->getOut()[0]<<", " << data[i]->getOut()[1]<<", " << data[i]->getOut()[2]<<", " << data[i]->getOut()[3]
                <<std::endl;
    }*/
    
    NeuralNetBuilder builder(2);
    builder.addDenseLayer(10, logistic, dlogistic);
    //builder.addDropout(0.4);
    builder.addDenseLayer(8, logistic, dlogistic);
   // builder.addDropout(0.2);
    builder.addDenseLayer(4, relu, drelu);
    
    builder.summary();
    
    std::unique_ptr<Network> network(builder.build());
  
    
    // 40 000 iterations
    for(int i=0; i<2000; i++) {
        float mod = max(0.0f, (float)i - 1500);
        mod = 1.0f - (mod/1000);  // mod = learning speed, first 30'000 iters equal 1.0 after that mod will decrease
     
        mod *= 0.05f;
        
        for(int j=0; j<setSize; j++) {
            memcpy(network->getInput(), data[j]->getIn(), 2 * sizeof(float));       
            
           // std::cout<<"IN : " << network->getInput()[0] << " " << network->getInput()[1] <<std::endl;
            network->calculate();
            // backpropagation
            network->backprop(data[j]->getOut(),  mod);
        }
        
        if (i % 100 == 0) {
            std::cout<<"ITER: " << i<<endl;
        }
    }
    
    float totalError = 0.0f;
    
    int validCount = 0;
    
    for(int j=0; j<setSize; j++) {
        memcpy(network->getInput(), data[j]->getIn(), 2 * sizeof(float));           
        network->calculate();
        
        int valid = 1;
        for(int i=0; i<4; i++) {
            if (network->getOutput()[i] > 0.1) {
                network->getOutput()[i] = 1.0;
            }
            
            if ( fabs(data[j]->getOut()[i] - network->getOutput()[i]) > 0.1) {
                valid = 0;
            }
        }
        
     /*   cout<<"DATA: " <<j << " : " << network->getOutput()[0] << " " << network->getOutput()[1] << " "<< network->getOutput()[2] << " " << network->getOutput()[3] << " | "
                   << data[j]->getOut()[0] <<" " << data[j]->getOut()[1] <<" " << data[j]->getOut()[2] <<" " << data[j]->getOut()[3] <<" " 
                << std::endl;*/
            // backpropagation
        totalError += network->totalError(data[j]->getOut());
        

        
        validCount += valid;
        
    }
    
    std::cout<<"TOTAL ERROR: " << (totalError/setSize) << std::endl;
    std::cout<<"Valid: " << validCount << " " << ((float)validCount/setSize) << "%"<< std::endl;
}

/*
 * 
 */
int main(int argc, char** argv) {
    cout<<"Neural network"<<endl;

    buildNetworkTest();
    testNetworkTraining();
    testDropout();
  
    
    networkTraining();
    
/*   */
    
    return 0;
}


unique_ptr<Data>* generateData(int setSize, float minX, float minY, float maxX, float maxY, float (*f1)(float), float (*f2)(float)) {
    unique_ptr<Data>* data = new unique_ptr<Data>[setSize];
    
    for(int i=0; i<setSize; i++) {
        float x = random()%1000;
        float y = random()%1000;
        
        x /= 1000.0f;
        y /= 1000.0f;
        
        x *= (maxX - minX);
        y *= (maxY - minY);
        
        x += minX;
        y += minY;
        
        int f1Side = (f1(x) - y) > 0 ? 1 : 0;
        int f2Side = (f2(x) - y) > 0 ? 1 : 0;
       
        int classNo = (f1Side << 1) | f2Side;
        
        data[i] = unique_ptr<Data>(new Data(x, y, classNo));
        
    }
    
    return data;
}
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

#include "Network.h"
#include "NeuralNetBuilder.h"

using namespace std;

void buildNetworkTest() {
    NeuralNetBuilder netBuilder(2);
    
    netBuilder.addDenseLayer(20, identity, didentity);
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

/*
 * 
 */
int main(int argc, char** argv) {
    cout<<"Neural network"<<endl;

    buildNetworkTest();
    testNetworkTraining();
  
    
/*   */
    
    return 0;
}


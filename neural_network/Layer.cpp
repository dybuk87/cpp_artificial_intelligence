/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Layer.cpp
 * Author: arturduch
 * 
 * Created on 8 listopada 2018, 13:37
 */

#include "Layer.h"

#include <cstring>


 Layer::Layer(const char *_name) : 
   name(std::unique_ptr<char[]>(new char[strlen(_name) + 1])) {
     memset(name.get(), 0, strlen(_name) + 1);
     memcpy(name.get(), _name, strlen(_name));
     
 }
 
const char*  Layer::getName() const {
    return name.get();
}
 
 Layer::~Layer() {
     
 }
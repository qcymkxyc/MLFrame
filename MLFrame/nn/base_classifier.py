#coding=utf8
'''
Created on 2018年2月24日

@author: Administrator
'''

from nn.base_estimator import BaseEstimator
import numpy as np
from basic.base_func import relu,relu_derivative
from basic.base_func import sigmoid,sigmoid_derivative

class BaseClassifier(BaseEstimator):
    """
                分类器的基类
    """
    def __init__(self,
                 hidden_units,
                 n_inputs,
                 n_classes,
                 learning_rate
                 ):
        
        self._learning_rate = learning_rate
        self._n_inputs = n_inputs
        self._n_classes = n_classes
        
        
        self._layer_list = self._build_layers(hidden_units)

    def _build_model(self,x,y):
        pred_y = self._forward_propagation(x)
        loss = self._call_loss(pred_y = pred_y, y = y)
        return loss
        self._call_optimizer(loss)
        
        
        
    def _build_layers(self,hidden_layers):
        BaseEstimator._build_layers(self, hidden_layers)
        if not isinstance(hidden_layers, list):
            raise TypeError("隐藏层必须为list类型")
        
        if self._n_classes == 2:
            hidden_layers.append(self._n_classes - 1)
        layers_list = []
        for i,current_units in enumerate(hidden_layers):
            #获取上一层的神经元数
            previous_units = self._n_inputs if i == 0 else hidden_layers[i - 1]
            
            weights = np.random.normal(
                                    loc = 0., 
                                    scale = 2. / previous_units,
                                    size = [previous_units,current_units])
            bias = np.random.normal(size = [current_units])
            
            weights_dict = {
                "weights" : weights,
                "bias" : bias
                };

            layers_list.append(weights_dict)
            
        return layers_list
            
            
    def _forward_propagation(self,x):
        BaseEstimator._forward_propagation(self,x)
        
        net = x
        for i,value in enumerate(self._layer_list):
            weights = value["weights"]
            bias = value["bias"]
            
            net = np.dot(net,weights) + bias
            
            if i == len(self._layer_list) - 1:
                break
            net = relu(net)
            
        net = sigmoid(net)
        
        return net
       
    
    def _call_loss(self, pred_y, y):
        BaseEstimator._call_loss(self, pred_y, y)
        
    def _backward_propagation(self, x, y, loss, layer_list):
        BaseEstimator._backward_propagation(self, x, y, loss, layer_list) 
        
            
            
        
        
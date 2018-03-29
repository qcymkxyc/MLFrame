#/usr/bin/python3
#coding=utf-8

import numpy as np
import tensorflow as tf


def discriminator_model_fn(features,labels,mode,params):
    """
        判别器的主体函数
    """
    net = tf.feature_column.input_layer(features,params["feature_columns"])
    for units in params["hidden_units"]:
        net = tf.layers.dense(net,units = units,activation = tf.nn.relu)
    
    logits = tf.layers.dense(net,1,activation = tf.nn.sigmoid);
    
    predict_class = tf.argmax(logits,1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predict_class[:, tf.newaxis],
            'probabilities': logits,
        }
        return tf.estimator.EstimatorSpec(mode,predictions = predictions)
        
    one_hot_label = tf.one_hot(indices = labels,
                               depth = 2,
                               name = "one_hot_label")
    loss = tf.reduce_mean(tf.concat([tf.log(logits + 1e-7),tf.log(1 - logits + 1e-7)],axis = 1) * one_hot_label
        )
    
        # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions = predict_class,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate = params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def generator_model_fn(features,labels,mode,params):
    """
        判别器的主体函数
    """
    net = tf.feature_column.input_layer(features,params["feature_columns"])
    for units in params["hidden_units"]:
        net = tf.layers.dense(net,units = units,activation = tf.nn.relu)
    
    logits = tf.layers.dense(net,params["label_dimension"],activation = None);
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': logits,
        }
        return tf.estimator.EstimatorSpec(mode,predictions = predictions)
        
    discriminator = params["discriminator"];
    
    discriminator_input_fn = tf.estimator.inputs.numpy_input_fn(
                                x = {"x" :logits},
                                batch_size = 1,
                                shuffle = False
    )
    
    disc_genr = discriminator.predict(discriminator_input_fn)
    
#     disc_pred = [v["probabilities"] for v in disc_genr]
    for i in disc_genr:
        print(i)
    
    loss = tf.reduce_mean(tf.log(1 - disc_genr))
    

    metrics = {'loss': loss}
#     tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate = params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class GANs(object):
    """
        基本的GAN类
    """
    def __init__(self,
                 sample_dim,
                 generator_input_dim = 30,
                 hidden_units = {
                   "discriminator" : [20,20],
                   "generator" : [20,20]  
                 },
                 dropout = {
                  "discriminator" : None,
                  "generator" : None   
                },
                 learning_rate = {
                  "discriminator" : 0.01,
                  "generator" : 0.01
                },
                 model_dir = "./log/"):
        """
            Args：
                
        """
        self.__sample_dim = sample_dim
        #定义判别器
        self.discriminator = tf.estimator.Estimator(
                model_fn = discriminator_model_fn,
                model_dir = model_dir + "discriminator",
               
                params = {
                    "feature_columns" :  tf.feature_column.numeric_column("x", (sample_dim)),
                    "dropout" : dropout["discriminator"],
                    "hidden_units" : hidden_units["discriminator"],
                    "learning_rate" : learning_rate["discriminator"]
                }
            )
        
        #定义生成器
        self.generator = tf.estimator.Estimator(
                model_fn = generator_model_fn,
                model_dir = model_dir + "generator",
                params = {
                    "feature_columns" : tf.feature_column.numeric_column("x",(generator_input_dim)),
                    "dropout" : dropout["generator"],
                    "hidden_units" : hidden_units["generator"],
                    "discriminator" : self.discriminator,
                    "label_dimension" : sample_dim

                }
            )
    
    def generate(self,generator_input_dim,count):
        """
            生成
        """
        generator_input_fn = tf.estimator.inputs.numpy_input_fn(
                                x = np.random.normal(size = [count,generator_input_dim]),
                                batch_size = 1,
                                shuffle = False
        )
        generated_data = self.generator.predict(
                            input_fn = generator_input_fn) 
        predict_data = [i for i in generated_data] 
        return np.array(predict_data)
      
    
    def _train_discriminator(self,x,y,steps,batch = 128):
        """
            训练识别器
        """
        discriminator_input_fn = tf.estimator.inputs.numpy_input_fn(
                                            x = {"x" : x},
                                            y = y,
                                            batch_size = batch,
                                            shuffle = True)
        self.discriminator.train(
                            input_fn = discriminator_input_fn,
                            steps = steps)
        
    def _train_generator(self,count):
        x = np.random.normal(size = [count,self.__sample_dim])
        generator_input_fn = tf.estimator.inputs.numpy_input_fn(
                                            x = {"x" : x},
                                            batch_size = count,
                                            shuffle = False
        )
        self.generator.train(
            input_fn = generator_input_fn,
            steps = 1
        )
    
    def _concat_data(self,generated_data,real_data):
        """
            合并成一个数据集
        """
        assert generated_data.shape == real_data.shape
        generated_y = [0] * generated_data.shape[0]
        real_y = [1] * real_data.shape[0]
        generated_y.extend(real_y)
        y = np.array(generated_y)
        
        data = np.concatenate([generated_data,real_data],axis = 0)
        return data,y
        
        
        
    def train(self,
              x,
              discriminator_repeat = 10,
              generator_input_dim = 30,
              batch = 128,
              steps = 1000):
        i = 0;
        while(i < steps):
            try:
                generate_data = self.generate(generator_input_dim = generator_input_dim,
                                              count = x.shape[0])
            except ValueError as e:
                generate_data = np.random.normal(size = x.shape)
                
            x,y = self._concat_data(generate_data, x)  
             
            self._train_discriminator(x, y, discriminator_repeat,batch)
            self._train_generator(x.shape[0])
            
            i += 1;


        
        
g = GANs(4)
g.train(x = np.arange(12).reshape(3,4))


    
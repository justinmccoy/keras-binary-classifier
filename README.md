# seeFOOD CNN a binary classifier written in Keras and converted to coreML


Walk you through how to use GPU hardware in the Cloud with Nimbix, to quickly train and deploy a Convolutional Neural Network Model that can tell you whether or not your lunchtime nutritional choice is the right one - all with the camera of the mobile phone in your pocket. All you need are some photos, descriptions of them, and you can be up and running with a model to stream video through in no time flat.

I'm sure you've seen the eposide of Silicon Valley, but to give you an idea of the amazing technology we are going to share with you today here's a clip:

[![SEEFOOD](https://img.youtube.com/vi/ACmydtFDTGs/0.jpg)](https://www.youtube.com/watch?v=ACmydtFDTGs)

So you want to identify hotdogs, great, summer is just around the corner, and you can never be too careful of what you're eating.  You too can develop an app that identifies **Hotdog** and the alternatives... **NotHotdog**


### Overview
This repo will walk you through the steps, and technologies to train a Deep Learning model using a Convolitional Netural Network, evaluate it's accuracy, and save it into a format that can be loaded on an iOS device. With a model converted to Apple's CoreML format we will load a `.mlmodel` into an opensource project: [Lumina](https://github.com/dokun1/lumina).  Within Lumina you can quickly import and activate your .mlmodel, and stream object predictions in real time from the camera feed... Let me repeat, you can stream object predictions from the camera feed in real time.

![Flow](images/flow.png)

### Technologies

#### Lumina 
*[Lumina](https://github.com/dokun1/lumina) is an iOS camera designed in Swift that can use any CoreML model for object recognition, as well as streaming video, images, and qr/bar codes.*

![Lumina](images/luminaLogo.jpg)

#### CoreMLTools
*[CoreMLTools](https://github.com/apple/coremltools) Integrates trained machine learning models into your iOS app*
![CoreML](images/coreml.jpg)

#### Keras 
*[Keras.io](https://keras.io/) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.*
![Keras](images/keras.png)

#### Nimbix
*[Nimbix](https://nimbix.net) provides super computing in the Cloud*
![NIMBIX](images/nimbix.png)

#### PowerAI
*[PowerAI](https://www.ibm.com/us-en/marketplace/deep-learning-platform/details#product-header-top)  takes advantage of the CPU:GPU NVLink interconnect (that's a fat pipe between CPU:GPU:Memory) to help support and load larger deep learning models than ever before. Train datasets that could never be trained before utilizing system memory without bottlenecks*
![]

### Steps

### Links
* [Free Trial to GPU Accelerated HW in the Cloud](https://developer.ibm.com/linuxonpower/cloud-resources/)


### License
[Apache 2.0](LICENSE)

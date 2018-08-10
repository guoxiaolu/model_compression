# model_compression
deep learning model compression based on keras

This is an application of "PRUNING FILTERS FOR EFFICIENT CONVNETS"[https://arxiv.org/abs/1608.08710].

This code is different in some places: 1. I don't compress the model layer by layer as time consuming. 2. Use Gaussian-like distribution to remove the layer channel instead fixed number/ratio.

It supports VGG-like or resnet-like model. You can modify the "std_times"(the lower this value is, the more compressed the model is). I have tested in resnet50 on imagenet, and std_times=1.0 is better (about 13% parameters is moved).

Inception-like model is not supported, as the 'concat' layer. You can take a try.

I have tested resnet50 on imagenet, if std_times=1.0, the top-1 error is 0.1% higher and top-5 error is 0.4% higher. However, I don't find the code evaluated in imagenet that can achieve the offical top-1 and top-5 error. If you know, please let me know. This is a hidden problem.(This problem is solved, see[https://github.com/guoxiaolu/model_compression/issues/1])

You can test lenet-5 on mnist, the evaluation result is almost same.

The "bak" file is other tried methods, like calculate gradients, compress the layer by fixed number.

Thank you for the project and blog[https://github.com/jacobgil/pytorch-pruning], it helps me a lot.

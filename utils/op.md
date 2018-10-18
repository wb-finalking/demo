# Tensorflow Custom Op Gradient

+ ### C++

We decided to simply start with a toy example: a copy-Layer.

------

**copy_op.cc**

```c++
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdio.h>

namespace tensorflow {



typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template<typename Device, typename T>
class MyCopyOp: public OpKernel {
public:
    explicit MyCopyOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        auto in_flat = input.flat<T>();

        printf("Debug MyCopyOp Features: %s \n",input.DebugString().c_str());

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, input.shape(), &output));

        auto out_flat = output->flat<T>();
        out_flat.setZero();

        for (int d = 0; d < input.dims(); ++d) {
            for (int i = 0; i < input.dim_size(d); ++i) {
                out_flat(d * input.dim_size(d) + i) = in_flat(
                        d * input.dim_size(d) + i);
            }
        }

        printf("Debug MyCopyOp Output: %s \n",output->DebugString().c_str());
    }

};


template<typename Device, typename T>
class MyCopyGradOp: public OpKernel {
public:
    explicit MyCopyGradOp(OpKernelConstruction* context) :
            OpKernel(context) {

    }

    void Compute(OpKernelContext* context) override {
        printf("called MyCopyGradOp.Compute() \n");
        const Tensor& gradients = context->input(0);
        const Tensor& features = context->input(1);
        printf("Debug MyCopyOpGrad Gradients: %s \n",gradients.DebugString().c_str());
        printf("Debug MyCopyOpGrad Features: %s \n",features.DebugString().c_str());

        TensorShape output_shape = features.shape();

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, output_shape, &output));
        output->flat<T>().setZero();

        const T* btm_ptr = gradients.flat<T>().data();
        T* top_ptr = output->flat<T>().data();

        for (int i = 0; i < gradients.NumElements(); ++i) {
            top_ptr[i] = btm_ptr[i];
        }

        printf("Debug MyCopyOpGrad Output: %s \n",output->DebugString().c_str());
        printf("---------------------------------- \n");
    }

};


REGISTER_OP("MyCopy")
.Input("features: T")
.Output("output: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
Copies all input values to the output
)doc");

REGISTER_OP("MyCopyGrad")
.Input("gradients: T")
.Input("features: T")
.Output("backprops: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
TODO!!
)doc");


#define REGISTER_MYCOPY_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("MyCopy").Device(DEVICE_CPU).TypeConstraint<type>("T"),              \
      MyCopyOp<Eigen::ThreadPoolDevice, type>);                                 \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("MyCopyGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      MyCopyGradOp<Eigen::ThreadPoolDevice, type>);                             //  \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("MyCopy").Device(DEVICE_GPU).TypeConstraint<type>("T"),              \
  //     MyCopyOp<Eigen::GpuDevice, type>);                                        \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("MyCopyGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
  //     MyCopyGradOp<Eigen::GpuDevice, type>);                                


REGISTER_MYCOPY_KERNELS(float); 
REGISTER_MYCOPY_KERNELS(int);
REGISTER_MYCOPY_KERNELS(double);


}
```

We used the simple MNIST example as the basis:

**layer_test.py**

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
from tensorflow.python.framework import ops
copy_op_module = tf.load_op_library('copy_op.so')

@ops.RegisterGradient("MyCopy")
def _CopyOpGrad(op, grad):
  return copy_op_module.my_copy_grad(grad,op.inputs[0])

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y1 = tf.nn.softmax(tf.matmul(x,W) + b)
y = copy_op_module.my_copy(y1)            //Here: MyCopy Layer is inserted

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(2):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

**compile**

```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared copy_op.cc -o copy_op.so -I $TF_INC -L $TF_LIB -fPIC -Wl,-rpath $TF_LIB
```
+ ### python

Gradient of `py_func` is `None` (just check `ops.get_gradient_function(y2.op)`). There's this [gist](https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342) by @harpone which shows how to use gradient override map for py_func.

Here's your example modified to use that recipe

```python
import numpy as np
import tensorflow as tf

def addone(x):
    # print(type(x)
    return x + 1

def addone_grad(op, grad):
    x = op.inputs[0]
    return x

from tensorflow.python.framework import ops
import numpy as np

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def pyfunc_test():

    # create data
    x_data = tf.placeholder(dtype=tf.float32, shape=[None])
    y_data = tf.placeholder(dtype=tf.float32, shape=[None])

    w = tf.Variable(tf.constant([0.5]))
    b = tf.Variable(tf.zeros([1]))

    y1 = tf.mul(w, x_data, name='y1')
    y2 = py_func(addone, [y1], [tf.float32], grad=addone_grad)[0]
    y = tf.add(y2, b)

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    print("Pyfunc grad", ops.get_gradient_function(y2.op))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10):
            #            ran = np.random.rand(115).astype(np.float32)
            ran = np.ones((115)).astype(np.float32)
            ans = ran * 1.5 + 3
            dic = {x_data: ran, y_data: ans}
            tt, yy, yy1= sess.run([train, y1, y2], feed_dict=dic)
            if step % 1 == 0:
                print('step {}'.format(step))
                print('{}, {}'.format(w.eval(), b.eval()))

        test = sess.run(y, feed_dict={x_data:[1]})
        print('test = {}'.format(test))


if __name__ == '__main__':
    pyfunc_test()
```
# Pkg.add.(["TensorFlow", "Distributions", "ProgressMeter", "MLLabelUtils", "MLDataUtils"])
# # Pkg.add("MLDatasets")  # not registered yet
# Pkg.clone("https://github.com/JuliaML/MLDatasets.jl.git")
# Pkg.checkout("MLDataUtils", "dev")
# using Conda
# # Conda.add("tensorflow")
# # using PyCall
# # @pyimport Conda
# # @show Conda.PYTHONDIR

using Base.Test
using Distributions
using MLLabelUtils
using MLDataUtils
using MLDatasets
using ProgressMeter
using TensorFlow

#Training Hyper Parameter
const learning_rate = 0.001
const training_iters = 2 #Just two, becuase I don't have anything to stop overfitting and I don't got all day
const batch_size = 256
const display_step = 100 #How often to display the

# Network Parameters
const n_input = 28 # MNIST data input (img shape: 28*28)
const n_steps = 28 # timesteps
const n_hidden = 128 # hidden layer num of features
const n_classes = 10; # MNIST total classes (0-9 digits)

const traindata_raw, trainlabels_raw = MNIST.traindata();
@show size(traindata_raw)
@show size(trainlabels_raw)

imshow(x) = join(mapslices(join, (x->x ? 'X': ' ').(x'.>0), 2), "\n") |> print
@show trainlabels_raw[8]
imshow(traindata_raw[:,:,8])


"""Makes 1 hot, row encoded labels."""
encode(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)),  LearnBase.ObsDim.First())

"""Prepares the data by encoding the labels and batching"""
prepared_batchs(data_raw, labels_raw) = eachbatch((data_raw, encode(labels_raw)), #Will zip these
                               batch_size,
                               (MLDataUtils.ObsDim.Last(), MLDataUtils.ObsDim.First())) #Slicing dimentions for each

@testset "data prep" begin

    @test encode([4,1,2,3,0]) == [0 0 0 0 1 0 0 0 0 0
                                  0 1 0 0 0 0 0 0 0 0
                                  0 0 1 0 0 0 0 0 0 0
                                  0 0 0 1 0 0 0 0 0 0
                                  1 0 0 0 0 0 0 0 0 0]

    data_b1, labels_b1 = first(prepared_batchs(traindata_raw, trainlabels_raw))
    @test size(data_b1) == (n_steps, n_input, batch_size)
    @test size(labels_b1) == (batch_size, n_classes)
end;


sess = Session(Graph())
X = placeholder(Float32, shape=[n_steps, n_input, batch_size])
Y_obs = placeholder(Float32, shape=[batch_size, n_classes])

variable_scope("model", initializer=Normal(0, 0.5)) do
    global W = get_variable("weights", [n_hidden, n_classes], Float32)
    global B = get_variable("bias", [n_classes], Float32)
end;


# Prepare data shape to match `rnn` function requirements
# Current data input shape: (n_steps, n_input, batch_size) from the way we declared X (and the way the data actually comes)
# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

x = transpose(X, Int32.([1, 3, 2].-1)) # Permuting batch_size and n_steps. (the -1 is to use 0 based indexing)
x = reshape(x, [n_steps*batch_size, n_input]) # Reshaping to (n_steps*batch_size, n_input)
x = split(1, n_steps, x) # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
@show get_shape.(x);


Hs, states = nn.rnn(nn.rnn_cell.LSTMCell(n_hidden), x; dtype=Float32);
Y_pred = nn.softmax(Hs[end]*W + B)

@show get_shape(Y_obs)
@show get_shape(Y_pred);


cost = reduce_mean(-reduce_sum(Y_obs.*log(Y_pred), reduction_indices=[1])) #cross entropy
@show get_shape(Y_obs.*log(Y_pred))
@show get_shape(cost) #Should be [] as it should be a scalar

optimizer = train.minimize(train.AdamOptimizer(learning_rate), cost)

correct_prediction = indmax(Y_obs, 2) .== indmax(Y_pred, 2)
@show get_shape(correct_prediction)
accuracy = reduce_mean(cast(correct_prediction, Float32));



run(sess, initialize_all_variables())

kk=0
for jj in 1:training_iters
    for (xs, ys) in prepared_batchs(traindata_raw, trainlabels_raw)
        run(sess, optimizer,  Dict(X=>xs, Y_obs=>ys))
        kk+=1
        if kk % display_step == 1
            train_accuracy, train_cost = run(sess, [accuracy, cost], Dict(X=>xs, Y_obs=>ys))
            info("step $(kk*batch_size), loss = $(train_cost),  accuracy $(train_accuracy)")
        end
    end
end

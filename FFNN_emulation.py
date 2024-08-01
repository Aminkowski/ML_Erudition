import torch
import torchvision
import torchvision.transforms as tfs
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# just importing modules. explanatory names, but black boxes nontheless.

trnsfrm = tfs.Compose([   # probably just function composition
    tfs.ToTensor(),     # first turns input into a tensor
    tfs.Normalize((0.5,), (0.5,))   # normalizes (elaborated below)
])
# so in this case we are giving an image as an input (torchvision)
# the resulting tensor is going to typically be a matrix of RGB values
# in our case it's black & white (95% sure) so
# due to compatibility reasons it's prolly gonna be a matrix of 1-tuples
# and they are going to be values between 0 and 1 (inferred)
# the Normalize() method takes in a tuple of means as it's first input 
# and a tuple of standard deviations as it's second input.
# so in the case of rgb we may have had (127, 127, 64) as the arg1 (127 = 255/2)
# and (25, 27, 20) for arg2 (tbh pulled these std's)
# oh yeah and here the values are being renormalized to -1 to 1
# as the formula is $(x - m) / s$ where x is the value, m mean, s std

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trnsfrm)
# the training data from mnist being loaded and put through our transform above.
testset = torchvision.datasets.MNIST(root='.data', train=False, download=True, transform=trnsfrm)
# this is the data that will be tested against.
# about 10^7 images in training set, and a million images as test

# ======== code to visualize image ========
def imshow(image):
    image = 0.5 + (image / 2)   # re-fucking-normalizing back to [0, 1]
    np_image = image.numpy()    # so... numpy-ifying the image. 
    # which is data from a module in torchvision. hmm. 
    # guess we just returning an np_array-ish thing
    plt.imshow(np_image[0], cmap='gray')
    plt.show()      # no idea about the details but boilerplate to "show" np_image

N = 127
sample_image, sample_label = trainset[N]
imshow(sample_image)
print(sample_label)
# ======= does nothing for the model ======

# Just loading the training / test data into torch module infrastructure
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
# not sure why the training data was shuffled but not the test.

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        # inserting layer
        self.fc1 = nn.Linear(28*28, 10)     # initializes internal module state
        # uuuh, but also applies "some" linear transformation
        # defining the "linear part" of the function (y = mx + b)
        # ML jargon for a linear function is a "fully connected layer". hence fc
        # images are 28x28 and there are 10 categories
        # first arg is the size of the input, second arg is size of output
        self.sigmoid = nn.Sigmoid()     # applies the sigmoid to each arg
        # isn't the init method supposed to just initialize attributes of the class?
        # what's up with these functions in here?
        # actually pretty sure we're essentially just providing aliases here.
        # if true, we could've...(?) just applied the nn.Linear and nn.Sigmoid
        # functions instead when the .fci and .sigmoid functions appear...
        # yeah... i think

    def forward(self, x):
        x = x.view(-1, 28*28)   # just flattens the image into a 784 diml vector
        # -1 because pytorch syntax. if 1 then it becomes a row vector
        x = self.sigmoid(self.fc1(x))   # yep. didn't seem necessary to define in init
        # so we took an image, returning a vector of probabilities
        return x

# **so here is the part where the model is being trained**
net = FeedForwardNN()   # defining an instance of the class. this is the model
criterion = nn.CrossEntropyLoss()   # our loss function defined here.
optimizer = optim.Adam(net.parameters(), lr=0.001)  # the thing that does the gradient descent
# Adam is an optimizer in pytorch. lr is the learning rate.
# note that optimizer takes the parameters of the net as args

# training the network on the data
for i, data in enumerate(trainloader, 0):   # recall, trainloader is a pytorch
    # data infrastructure of our training data. i is prolly just the iteration number
    inputs, labels = data   # self explanatory
    optimizer.zero_grad()   # "clears" / resets the gradient of the optimizer

    outputs = net(inputs)   # mother fucker used this fuckin notation
    # looks pleasing and intuitive but I would've been so confused if 
    # I hadn't already gone through the docs.
    # the syntax net(inputs) is "applying" net to the inputs
    # but you may say "wait... isn't net... like... the model? how do you apply a model?"
    # to which I say that "don't think about the code bruv, think about it intuitively"
    # which is to say that it applies all the changes that the model is supposd to do in a layer
    # what's going on here is that python classes / objects have a __call__ method
    # in which you tell the program what to do if the object is called.
    # for all sub-classes / sub-objects of nn.Module, this is essentially just
    # some error-handling and caling the hooks of the model.
    # what are hooks you may ask? essentially the functions you defined in your model
    # this can be more general than the layer transforms, but in our case it's
    # essentially just returning net.forward.
    # TL;DR: net(inputs) == net.forward(inputs)
    loss = criterion(outputs, labels)   # intuitively self-explanatory

    loss.backward()     # aleph: computes the dir of steepest descent of loss function
    # iow: returns d loss / d criteria vector (10 diml)
    optimizer.step()    # aleph: this updates the parameters in the dir of s d.
    # iow: probably som backpropagation thing. not that I have a good understanding of bp.
    # it appears also that optimizer modifies the model. weird.
    # thought you'd need to call a method of the object to do that
    # huh. so when you pass on net.parameters() in the optimizer definiton,
    # you're actually passing the memory locations, not just the values...?
    # I feel like I said something both profound and incredibly obvious lol

# now that the model has been trained, we evaluate
net.eval()      # just turns of "training mode" for evaluation. model no longer being modified
correct=0
total=0     # counters
for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)     # syntax a product of details of the pytorch infrastructure
    correct += (predicted == labels).sum().item()   # same... I guess...

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
# in this sample we got 89.11%

# now as a review, code something like the iteration 2 code from memory

# class AmiNN(nn.Module):
#     def __init__(self): #self?
#         super.AmiNN(self).__init__()
#             # super(AmiNN, self).__init__()
#         # something else here?
#         self.lin1 = nn.Linear(28*28, 14*14)     # bro forgot it's called fully connected layer smh
#         self.lin2 = nn.Linear(14*14, 11*11)     # maybe co-prime-ness is good
#         self.lin3 = nn.Linear(11*11, 10)
#         # going off my theory that these are just functions ==> multiple sigmoids are redundant
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         x = x.view(-1, 28*28)
#         x = self.sig(self.lin1(x))
#         x = self.sig(self.lin2(x))
#         x = self.sig(self.lin3(x))
#         return x
#
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 14*14)
        self.fc2 = nn.Linear(14*14, 11*11)
        self.fc3 = nn.Linear(11*11, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x

# newt = Aminn()
# loss = nn.CrossEntropyLoss()    # or was it torch?
# back = nn.Optimizer.Adam(newt.parameters(), lr = 0.001)
#     # opt = torch.optim.Adam(newt.parameters(), lr = 0.001)
#
fnet = FFNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fnet.parameters(), lr = 0.001)   # torch.optim.Adam

# for i, data in trainset:
#     # for i, data in enumerate(trainloader, 0):
#     images, labels = data   # images <-> inputs
#         # opt.zero_grad()
#     outputs = newt(images)
#     _, predictions = outputs(somethingsomething, 0)
#         # wrong section
#         # lossValue = loss(outputs, labels)
#     loss.dirofsd(predictions)
#         # lossValue.backward()
#     back.propagate()
#         # opt.step()
#     # pretty sure there was 1-3 more lines
#
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    outputs = fnet(inputs)
    loss = criterion(outputs, labels)

    loss.backward()     #loss.grad()
    optimizer.step()    #optimizer.backwards()

# newt.eval()
# correct = 0
# total = 0
# for data in testset:
#     inputs, labels = data
#     outputs = newt(images)
#     _, predictions = outputs(somethingsomething, 0)
#         # _, predictions = torch.max(outputs.data, 1)
#     total += someweirdwaytophraseit
#         # total += labels.size(0)
#     correct += someothershit
#         # correct += (predicted == labels).sum().item()
#     #at least 1-3 more lines
#         # wrong lol
#
# print("% of correct")
#
fnet.eval()
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = fnet(images)

    _, predictions = torch.max(outputs.data, 1)
    #_, predictions = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (predictions == labels).sum().item()

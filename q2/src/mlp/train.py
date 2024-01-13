from process import Net

model = Net(verbose=True, epoch=100, lr=0.002)
model.train(pretrained=False)

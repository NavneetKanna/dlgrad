from dlgrad.loss import crossentropy

def train(model, x_train, y_train):
    x = model.forward(x_train)
    loss = crossentropy(x, y_train)
    # loss.backward()
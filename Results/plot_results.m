data = load('ANN/Dropout.mat')

embedding = data.dropout;
train = data.train
val = data.val;
test = data.test;


semilogx(embedding, train)
hold on
semilogx(embedding, val)
semilogx(embedding, test)

legend('Train acc', 'Val acc', 'Test acc')

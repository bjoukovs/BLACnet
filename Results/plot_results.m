data = load('RNN/GRU2_Embedding.mat')

embedding = data.embedding;
train = data.train
val = data.val;
test = data.test;


plot(embedding, train)
hold on
plot(embedding, val)
plot(embedding, test)

legend('Train acc', 'Val acc', 'Test acc')


test = test';
train = train';
val = val';
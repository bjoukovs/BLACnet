clear all,
close all;

data = load('ANN/FINAL_EMBEDDING_2LAYERS.mat')

embedding = data.embedding;
train = data.train
val = data.val;
test = data.test;
accs = data.accuracy;


errorbar(embedding, mean(train, 2), std(train'))
hold on
errorbar(embedding,  mean(val, 2), std(val'))
semilogx(embedding, accs)

legend('Train acc', 'Val acc', 'Test acc')

%set(gca, 'XScale', 'log')



tm = mean(train, 2);
ts = std(train')';

vm = mean(val, 2);
vs = std(val')';

acc = accs';
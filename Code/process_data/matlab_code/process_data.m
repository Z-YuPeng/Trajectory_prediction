train_data = "train_data.mat";
train_target = "train_labels.mat";
test_data = "test_data.mat";

load(train_data);
load(train_target);
load(test_data);


train_data = train_x;
train_target = train_y;
test_data = test_x;

train_data = scaleForSVM(train_data,0,1);
test_data = scaleForSVM(test_data,0,1);

savepath = "MLEXPDatasets.mat";
 
save(savepath,'train_data','train_target','test_data');
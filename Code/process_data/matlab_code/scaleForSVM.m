function [dataset_scale,ps] = scaleForSVM(data,ymin,ymax)
if nargin < 3
    ymin = 0;
    ymax = 1;
end
[dataset_scale,ps] = mapminmax(data',ymin,ymax);
dataset_scale = dataset_scale';

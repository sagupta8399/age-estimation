function [ feat1 ] = make_data(net )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
feat=cell(1,1);
ind1=1;

tt=net.meta.normalization.averageImage;
temp=ones(224,224,3);

temp1(:,:,1)=temp(:,:,1)*tt(1);
temp1(:,:,2)=temp(:,:,2)*tt(2);
temp1(:,:,3)=temp(:,:,3)*tt(3);

d=dir('G:\wiki_crop\');

for index1=3:size(d,1)
    d1=dir(['G:\wiki_crop\',d(index1).name]);
    for index2=3:size(d1,1)
        temp=['G:\wiki_crop\',d(index1).name,'\',d1(index2).name]
        im=im2single(imread(temp));
        boxes = runObjectness(im,25);
        feat1=zeros(1,4096);
        ind=1;
        for index3=1:size(boxes,1)
            im1=im(boxes(index3,2):boxes(index3,4),boxes(index3,1):boxes(index3,3),:);
            im_ = imresize(im1, net.meta.normalization.imageSize(1:2)) ;
            im_ = im_ - single(temp1) ;
            res = vl_simplenn(net, im_) ;
            feat1(ind,:)=res(36).x;
            ind=ind+1;
        end
        feat{ind1}=feat1;
        ind1=ind1+1;
    end
end
    


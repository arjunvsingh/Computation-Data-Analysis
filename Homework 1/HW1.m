function [class,centroid] = HW1(pixels,K)

M=size(pixels,1);
class=zeros(K,1);
centroid=pixels(randsample(M,K),:);
%centroid=pixels(1:K,:);

iterno=100;
for iter = 1:iterno
    c2=sum(centroid.^2,2);
    c2=c2';
    tmpdiff=bsxfun(@minus,2*pixels*centroid',c2);
    [~,class1] = max(tmpdiff,[],2);       
    P=sparse(1:M,class1',1,M,K,M);
%recalculate the centroid
count=sum(P,1);
centroid=bsxfun(@rdivide,P'*pixels,count');
%decide to break the iteration
%choose the nearest node to be the centroid
for i = 1:K
    centroid_new=centroid(i,:);
    sum_distance=sum(bsxfun(@minus,centroid_new,pixels).^2,2);
    [~,index]=min(sum_distance);
    centroid(i,:)=pixels(index,:);
end
tf=isequal(class,class1);
    if tf == 0
      class=class1;
    else
        fprintf('--iteration end at %d\n', iter); 
        break     
    end
end


end



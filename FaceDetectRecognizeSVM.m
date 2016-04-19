%caner yildirim-21100818
traindata=csvread('train.csv');
testdata=csvread('test.csv');
[sizx,sizy]=size(traindata);
accuracy=zeros(35,1);
ind=1;

%------------------------linear
for cval=0.001:0.0005:0.015
model = svmtrain(traindata(:,1), traindata(:,2:sizy), strcat(['-q -t 0 -c ',num2str(cval)]));
predictions=svmpredict(testdata(:,1), testdata(:,2:sizy), model);
right=predictions==testdata(:,1);
accuracy(ind)=length(find(right==1))/40;
ind=ind+1;
end
figure;
x=0.001:0.0005:0.015;
plot(x,accuracy(1:1:length(x)));
xlabel('c');
ylabel('accuracy');

%------------------------rbf
accuracy2=zeros(20,20);
iND=1;
for cval=0.001:0.2:2.001
ind=1;    
    for gval= 0.001:0.003:0.025
model = svmtrain(traindata(:,1), traindata(:,2:sizy), strcat(['-q -t 2 -c  ',num2str(cval),' -g ',num2str(gval)]));
predictions = svmpredict(testdata(:,1), testdata(:,2:sizy), model);
right=predictions==testdata(:,1);
accuracy2(iND,ind)=length(find(right==1))/40;
ind=ind+1;
    end
iND=iND+1;    
figure(iND+1);
plot( 0.001:0.003:0.025,accuracy2(iND-1,1:1:(ind-1)));
xlabel(strcat(['cval ',num2str(cval),' gvals in x ']));
ylabel('accuracy');
end

%detection of web pages whether they belong to student or instructor etc. / Naive Bayes(with binomial or multinomial) with td-idf
smooth=input('Calculate with smoothing');
load('folds.txt');
dataset = importdata('data.txt');
data=dataset.data;
[site_no,vocab_no] = size(data);
rowheaders=dataset.rowheaders;
facultyinds=find(cellfun(@(rowheaders) strcmp(rowheaders,'faculty'), rowheaders));
studentinds= find(cellfun(@(rowheaders) strcmp(rowheaders,'student'), rowheaders));
facultysize=length(studentinds);
studentsize=site_no-facultysize;
labelledSet=[data folds];
smooth=1;
trainLabels = [2,3,4,5,6,7,8,9,10] ;
trainLabelSize=9;
testLabels = 1 ;
facultyno = 0 ;
studentno = 0 ;
documentno = 0 ;
xStudent = zeros(700,vocab_no+1);
xFaculty = zeros(700,vocab_no+1);
%TRAIN----------------------------------------------------
for k =1:site_no
    for trainLabelInd = 1:trainLabelSize
        if labelledSet(k,vocab_no+1)==trainLabels(trainLabelInd)
            if strcmp(rowheaders(k),'faculty')==1
                facultyno =facultyno+  1;
                documentno =documentno+  1;
                xFaculty(facultyno,:) =  labelledSet(k,:);
            end
            if strcmp(rowheaders(k),'student')==1
                studentno = studentno+ 1;
                documentno = documentno+ 1;
                xStudent(studentno,:) =labelledSet(k,:);
            end
        end
    end
end
%PRIOR PROBABILITIES-------------
Pfaculty = facultyno / documentno;
Pstudent = studentno / documentno;

disp ('Pfaculty-Pstudent');
disp(Pfaculty);
disp(Pstudent);
%Conditional Probobilities -------------------------
totalNumberOfWordsinStudents = sum(sum(xStudent));
totalNumberOfWordsinFaculties = sum(sum(xFaculty));

CondProStudent = zeros(1309,1);
CondProFaculty = zeros(1309,1);
%KEEP OUT 1 AND VOCAB_NO FOR PART A , OTHERWISE ITS PART B
if smooth==1
for k = 1:vocab_no
    CondProStudent(k) = (sum(xStudent(:,k)) + 1) / (totalNumberOfWordsinStudents + vocab_no);
end
for k = 1:vocab_no
    CondProFaculty(k) = (sum(xFaculty(:,k)) + 1) / (totalNumberOfWordsinFaculties + vocab_no);
end
end
if smooth==0
for k = 1:vocab_no
    CondProStudent(k) = (sum(xStudent(:,k)) ) / (totalNumberOfWordsinStudents );
end
for k = 1:vocab_no
    CondProFaculty(k) = (sum(xFaculty(:,k))) / (totalNumberOfWordsinFaculties);
end

end

    %TEST---------------------------------------------------------------------
predicted =zeros(1309,1) ;
for m = 1:site_no
    scoreStudent = 0 ;
    scoreFaculty = 0;
    if labelledSet(m,vocab_no+1)==1
        scoreStudent = log(Pstudent);
        scoreFaculty = log(Pfaculty);
        %Calculation MLE for student & faculty
        for k = 1:vocab_no
            if labelledSet(m,k)*CondProStudent(k) ~=0
                scoreStudent =scoreStudent + log(labelledSet(m,k)*CondProStudent(k)) ;
            end
            if labelledSet(m,k)*CondProFaculty(k) ~=0
                scoreFaculty =scoreFaculty + log(labelledSet(m,k)*CondProFaculty(k));
            end
        end
        %PREDICTION-------------------------
        if scoreFaculty > scoreStudent
            predicted(m)=1;
        end
        if scoreFaculty <= scoreStudent
            predicted(m)=0;
        end
    end
end

disp('Multinomial Predicted');
%disp (predicted);

%CONFUSION MATRIX------------------------------
index = 1 ;  	truePositive = 0; trueNegative = 0; falsePositive = 0; falseNegative = 0 ;
for r = 1:140
    if labelledSet(r,1310)==1
        if predicted(index)==1
            if strcmp(rowheaders(r),'faculty') ==1
                truePositive =truePositive+ 1;
                
            end
            if strcmp(rowheaders(r),'student')==1
                falsePositive =falsePositive+ 1;
            end
        end
        if predicted(index)==0
            if strcmp(rowheaders(r),'faculty')==1
                falseNegative =falseNegative+ 1;
            end
            if strcmp(rowheaders(r),'student')==1
                trueNegative =trueNegative+ 1;
            end
        end
        index =index+ 1;
    end
end
disp('Multinomial Reults');
disp('True Positive:');
disp(truePositive);
disp('True Negative:');
disp(trueNegative);
disp('False Positive:');
disp(falsePositive);
disp('False Negaitive:');
disp(falseNegative);
  %--------------------QUESTION 1.2------------------------------------
%------------------------ BINOMIAL-------------------------------
  %--------------------QUESTION 1.2-----------------------------------
xStudent2=(xStudent>0);
xFaculty2=(xFaculty>0);
%Conditional Probobilities BINOMIAL-------------------------
totalNumberOfWordsinStudents2 = sum(sum(xStudent2));
totalNumberOfWordsinFaculties2 = sum(sum(xFaculty2));

CondProStudent2 = zeros(1309,1);
CondProFaculty2 = zeros(1309,1);
%KEEP OUT 1 AND VOCAB_NO FOR PART A , OTHERWISE ITS PART B

if smooth==1
for k = 1:vocab_no
    CondProStudent2(k) = (sum(xStudent2(:,k))+1 ) / (totalNumberOfWordsinStudents2 +vocab_no);
end
for k = 1:vocab_no
    CondProFaculty2(k) = (sum(xFaculty2(:,k))+1) / (totalNumberOfWordsinFaculties2+vocab_no );
end
end

if smooth==0    
for k = 1:vocab_no
    CondProStudent2(k) = (sum(xStudent2(:,k)) ) / (totalNumberOfWordsinStudents2 );
end
for k = 1:vocab_no
    CondProFaculty2(k) = (sum(xFaculty2(:,k))) / (totalNumberOfWordsinFaculties2);
end
end
%TRAIN BINOMIAL---------------------------------------------------------------------
labelledSet2(:,1:1309)=labelledSet(:,1:1309)>0;
labelledSet2(:,1310)=labelledSet(:,1310);
predicted2 =zeros(1309,1) ;
for m = 1:site_no
    scoreStudent = 0 ;
    scoreFaculty = 0;
    if labelledSet2(m,vocab_no+1)==1
        scoreStudent = log(Pstudent);
        scoreFaculty = log(Pfaculty);
        %Calculation MLE for student & faculty
        for k = 1:vocab_no
            if labelledSet2(m,k)*CondProStudent2(k) ~=0
                scoreStudent =scoreStudent + log(labelledSet2(m,k)*CondProStudent2(k)) ;
            end
            if labelledSet2(m,k)*CondProFaculty2(k) ~=0
                scoreFaculty =scoreFaculty + log(labelledSet2(m,k)*CondProFaculty2(k));
            end
        end
        %PREDICTION BINOMIAL---------------------
        if scoreFaculty > scoreStudent
            predicted2(m)=1;
        end
        if scoreFaculty <= scoreStudent
            predicted2(m)=0;
        end
    end
end
disp('Predicted Binomial');

%CONFUSION MATRIX BINOMIAL-------------------------
index = 1 ;  	truePositive = 0; trueNegative = 0; falsePositive = 0; falseNegative = 0 ;
for r = 1:140
    if labelledSet2(r,1310)==1
        if predicted2(index)==1
            if strcmp(rowheaders(r),'faculty') ==1
                truePositive =truePositive+ 1;
                
            end
            if strcmp(rowheaders(r),'student')==1
                falsePositive =falsePositive+ 1;
            end
        end
        if predicted2(index)==0
            if strcmp(rowheaders(r),'faculty')==1
                falseNegative =falseNegative+ 1;
            end
            if strcmp(rowheaders(r),'student')==1
                trueNegative =trueNegative+ 1;
            end
        end
        index =index+ 1;
    end
end
disp('True Positive:');
disp(truePositive);
disp('True Negative:');
disp(trueNegative);
disp('False Positive:');
disp(falsePositive);
disp('False Negaitive:');
disp(falseNegative);
  %--------------------QUESTION 1.3------------------------------------
%---------TF-------------
  %--------------------QUESTION 1.3------------------------------------
N=facultyno+studentno;
tf=zeros(vocab_no,site_no);
i=1;
for d=1:site_no
    for trainLabelInd = 1:trainLabelSize
        if labelledSet(d,vocab_no+1)==trainLabels(trainLabelInd)
            for t=1:vocab_no
                
                if(data(d,t)~=0)
                    tf(t,i)=1+log(data(d,t));
                end
            end
            i=i+1;
        end
    end
end
%--------IDT---------------
idt=zeros(vocab_no,site_no);
for t=1:vocab_no
    cursum=0;
    for d=1:site_no
        for trainLabelInd = 1:trainLabelSize
            if labelledSet(d,vocab_no+1)==trainLabels(trainLabelInd)
                cursum= cursum+data(d,t);
            end
        end
    end
    idt(t,:)=log(N/(cursum+1));
end
%--------TD-IDF----------------
tdidf=tf.*idt;

data2=data.*tdidf';
xStudent3 = zeros(700,vocab_no);
xFaculty3 = zeros(700,vocab_no);
facultyno=0;studentno=0;documentno=0;
%TRAIN FOR TD-IDF-------------------------------------------------
for k =1:site_no
    for trainLabelInd = 1:trainLabelSize
        if folds(k)==trainLabels(trainLabelInd)
            if strcmp(rowheaders(k),'faculty')==1
                facultyno =facultyno+  1;
                documentno =documentno+  1;
                xFaculty3(facultyno,:) =  data2(k,:);
            end
            if strcmp(rowheaders(k),'student')==1
                studentno = studentno+ 1;
                documentno = documentno+ 1;
                xStudent3(studentno,:) =data2(k,:);
            end
        end
    end
end

%PRIOR PROBABILITIES FOR TD-IDF--------------------
Pfaculty = facultyno / documentno;
Pstudent = studentno / documentno;

disp ('Pfaculty-Pstudent');
disp(Pfaculty);
disp(Pstudent);
%Conditional Probobilities FOR TD-IDF --------------
totalNumberOfWordsinStudents3 = sum(sum(xStudent3));
totalNumberOfWordsinFaculties3 = sum(sum(xFaculty3));

CondProStudent3 = zeros(1309,1);
CondProFaculty3 = zeros(1309,1);

if smooth==1
for k = 1:vocab_no
    CondProStudent3(k) = (sum(xStudent3(:,k)) + 1) / (totalNumberOfWordsinStudents3 + vocab_no);
end
for k = 1:vocab_no
    CondProFaculty3(k) = (sum(xFaculty3(:,k)) + 1) / (totalNumberOfWordsinFaculties3 + vocab_no);
end
end

if smooth==0
for k = 1:vocab_no
    CondProStudent3(k) = (sum(xStudent3(:,k)) ) / (totalNumberOfWordsinStudents3 );
end
for k = 1:vocab_no
    CondProFaculty3(k) = (sum(xFaculty3(:,k)) ) / (totalNumberOfWordsinFaculties3 );
end
end
%TEST FOR TD-IDF---------------------------------------------------------------------
predicted3 =zeros(1309,1) ;
for m = 1:site_no
    scoreStudent = 0 ;
    scoreFaculty = 0;
    if folds(m)==1
        scoreStudent = log(Pstudent);
        scoreFaculty = log(Pfaculty);
        %Calculation MLE for student & faculty
        for k = 1:vocab_no
            if data2(m,k)*CondProStudent3(k) ~=0
                scoreStudent =scoreStudent + log(data2(m,k)*CondProStudent3(k)) ;
            end
            if data2(m,k)*CondProFaculty3(k) ~=0
                scoreFaculty =scoreFaculty + log(data2(m,k)*CondProFaculty3(k));
            end
        end
        %PREDICTION FOR TD-IDF-------------------------
        if scoreFaculty > scoreStudent
            predicted3(m)=1;
        end
        if scoreFaculty <= scoreStudent
            predicted3(m)=0;
        end
    end
end

disp('TD-IDT Multinomial Reslts');

%CONFUSION MATRIX FOR TD-IDF------------------------------
index = 1 ;  	truePositive = 0; trueNegative = 0; falsePositive = 0; falseNegative = 0;
for r = 1:140
    if labelledSet(r,1310)==1
        if predicted3(index)==1
            if strcmp(rowheaders(r),'faculty') ==1
                truePositive =truePositive+ 1;
                
            end
            if strcmp(rowheaders(r),'student')==1
                falsePositive =falsePositive+ 1;
            end
        end
        if predicted3(index)==0
            if strcmp(rowheaders(r),'faculty')==1
                falseNegative =falseNegative+ 1;
            end
            if strcmp(rowheaders(r),'student')==1
                trueNegative =trueNegative+ 1;
            end
        end
        index =index+ 1;
    end
end
    
    disp('True Positive:');
    disp(truePositive);
    disp('True Negative:');
    disp(trueNegative);
    disp('False Positive:');
    disp(falsePositive);
    disp('False Negaitive:');
    disp(falseNegative);
      %--------------------QUESTION 1.4------------------------------------
    %--------------------QUESTION 1.4------------------------------------
      %--------------------QUESTION 1.4------------------------------------
    amounts=[1,2,5,10,20];
    trPos=zeros(1,length(amounts));
    trNeg=zeros(1,length(amounts));
    flsPos=zeros(1,length(amounts));
    flsNeg=zeros(1,length(amounts));
    vocno=vocab_no;
    for indamount=1:length(amounts)
        vocab_no=amounts(indamount)+vocno;
        data3=[data repmat(data(:,668),1,amounts(indamount)) folds];
        facultyno = 0 ;
studentno = 0 ;
documentno = 0 ;
xStudent = zeros(700,vocab_no+1);
xFaculty = zeros(700,vocab_no+1);
%TRAIN----------------------------------------------------
for k =1:site_no
    for trainLabelInd = 1:trainLabelSize
        if data3(k,vocab_no+1)==trainLabels(trainLabelInd)
            if strcmp(rowheaders(k),'faculty')==1
                facultyno =facultyno+  1;
                documentno =documentno+  1;
                xFaculty(facultyno,:) =  data3(k,:);
            end
            if strcmp(rowheaders(k),'student')==1
                studentno = studentno+ 1;
                documentno = documentno+ 1;
                xStudent(studentno,:) =data3(k,:);
            end
        end
    end
end
%PRIOR PROBABILITIES-------------
Pfaculty = facultyno / documentno;
Pstudent = studentno / documentno;

disp ('Pfaculty-Pstudent');
disp(Pfaculty);
disp(Pstudent);
%Conditional Probobilities -------------------------
totalNumberOfWordsinStudents = sum(sum(xStudent));
totalNumberOfWordsinFaculties = sum(sum(xFaculty));

CondProStudent = zeros(1309,1);
CondProFaculty = zeros(1309,1);
%KEEP OUT 1 AND VOCAB_NO FOR PART A , OTHERWISE ITS PART B
if smooth==1
for k = 1:vocab_no
    CondProStudent(k) = (sum(xStudent(:,k)) + 1) / (totalNumberOfWordsinStudents + vocab_no);
end
for k = 1:vocab_no
    CondProFaculty(k) = (sum(xFaculty(:,k)) + 1) / (totalNumberOfWordsinFaculties + vocab_no);
end
end
if smooth==0
for k = 1:vocab_no
    CondProStudent(k) = (sum(xStudent(:,k)) ) / (totalNumberOfWordsinStudents );
end
for k = 1:vocab_no
    CondProFaculty(k) = (sum(xFaculty(:,k))) / (totalNumberOfWordsinFaculties);
end

end

    %TEST---------------------------------------------------------------------
predicted =zeros(1309,1) ;
for m = 1:site_no
    scoreStudent = 0 ;
    scoreFaculty = 0;
    if data3(m,vocab_no+1)==1
        scoreStudent = log(Pstudent);
        scoreFaculty = log(Pfaculty);
        %Calculation MLE for student & faculty
        for k = 1:vocab_no
            if data3(m,k)*CondProStudent(k) ~=0
                scoreStudent =scoreStudent + log(data3(m,k)*CondProStudent(k)) ;
            end
            if data3(m,k)*CondProFaculty(k) ~=0
                scoreFaculty =scoreFaculty + log(data3(m,k)*CondProFaculty(k));
            end
        end
        %PREDICTION-------------------------
        if scoreFaculty > scoreStudent
            predicted(m)=1;
        end
        if scoreFaculty <= scoreStudent
            predicted(m)=0;
        end
    end
end

disp('Multinomial Predicted');
%disp (predicted);

%CONFUSION MATRIX------------------------------
index = 1 ;  	truePositive = 0; trueNegative = 0; falsePositive = 0; falseNegative = 0 ;
for r = 1:140
    if data3(r,vocab_no+1)==1
        if predicted(index)==1
            if strcmp(rowheaders(r),'faculty') ==1
                truePositive =truePositive+ 1;
                
            end
            if strcmp(rowheaders(r),'student')==1
                falsePositive =falsePositive+ 1;
            end
        end
        if predicted(index)==0
            if strcmp(rowheaders(r),'faculty')==1
                falseNegative =falseNegative+ 1;
            end
            if strcmp(rowheaders(r),'student')==1
                trueNegative =trueNegative+ 1;
            end
        end
        index =index+ 1;
    end
end
disp('Reslts for N repeation');
disp(amounts(indamount));
disp('True Positive:');
disp(truePositive);
trPos(indamount)=truePositive;
disp('True Negative:');
disp(trueNegative);
trNeg(indamount)=trueNegative;
disp('False Positive:');
disp(falsePositive);
flsPos(indamount)=falsePositive;
disp('False Negaitive:');
disp(falseNegative);
flsNeg(indamount)=falseNegative;
    end
    figure;
    
    
    hold on;
    plot(1:length(amounts),flsNeg,'o');
    plot(1:length(amounts),flsPos,'.');
        legend('False Negative','False Positive');
    

classdef CNNmodel
    properties
        NumConvlayer=1;
        NumFullCon=1;
        kernels=cell(1,1);
        poolSize;
        KernelSize;
        alpha=0.1;
        W=cell(1,1);
    end
    methods
        function obj=CNNmodel(inputlayerSize,numkernel,poolSize,KernelSize,fullConSize)
%            inputlayerSize
            obj.kernels{1}=rand(KernelSize(1,1),KernelSize(1,2),inputlayerSize(3),numkernel(1))*0.2-0.1;
            obj.kernels{2}=rand(KernelSize(2,1),KernelSize(2,2),numkernel(1),numkernel(2))*0.2-0.1;
            obj.poolSize=poolSize;
            obj.KernelSize=KernelSize;
            obj.W{1}=rand(fullConSize(1),floor((floor((inputlayerSize(1)-KernelSize(1,1)+1)/poolSize(1))-KernelSize(2,1)+1)/poolSize(1))*floor((floor((inputlayerSize(2)-KernelSize(1,2)+1)/poolSize(2))-KernelSize(2,2)+1)/poolSize(1))*numkernel(2)+1)*0.2-0.1;
            obj.W{2}=rand(fullConSize(2),fullConSize(1)+1)*0.2-0.1;
            obj.W{3}=rand(1,fullConSize(2)+1)*0.2-0.1;
        end
        function [output,Clayer]=testmodel(obj,inputlayer)
            n=size(inputlayer,4);
            Clayer=cell(2,1);
            Clayer{1}=struct('conv','relu','pool','out');
            Clayer{1}.conv=obj.conv2d(inputlayer,obj.kernels{1});
            Clayer{1}.relu=obj.relu(Clayer{1}.conv);
            Clayer{1}.pool=obj.pool(Clayer{1}.relu);
%            Clayer{1}.out=reshape(Clayer{1}.pool,size(obj.W,1),n)';
            
            
            Clayer{2}.conv=obj.conv2d(Clayer{1}.pool,obj.kernels{2});
            Clayer{2}.relu=obj.relu(Clayer{2}.conv);
            Clayer{2}.pool=obj.pool(Clayer{2}.relu);
            out=reshape(Clayer{2}.pool,size(obj.W{1},2)-1,n)';
            Clayer{2}.out=out;
            
            Clayer{3}.conv=obj.fullCon([ones(n,1) Clayer{2}.out],obj.W{1});
            Clayer{3}.out=obj.relu(Clayer{3}.conv);
            Clayer{4}.conv=obj.fullCon([ones(n,1) Clayer{3}.out],obj.W{2});
            Clayer{4}.out=obj.relu(Clayer{4}.conv);
            output=obj.fullCon([ones(n,1) Clayer{4}.out],obj.W{3});
        end
        function [out]=conv2d(obj,input,kernel)
            n=size(input,4);
            [a,b,c,d]=size(kernel);
            out=zeros(size(input,1)-a+1,size(input,2)-b+1,d,n);
                for k=1:d
                    out(:,:,k,:)=convn(input,kernel(:,:,:,k),'valid');
                end
        end
        function [out]=relu(obj,input)
            out=input;
            %Ids=find(out<0);
            out(out<0)=0; %out(Ids)=obj.alpha*out(Ids);
        end
        function [out]=pool(obj,input)
            out=zeros(floor(size(input)./[obj.poolSize,1,1]));
            for i=1:size(out,1)
                for j=1:size(out,2)
                    out(i,j,:,:)=max(max(input((i-1)*obj.poolSize(1)+1:i*obj.poolSize(1),(j-1)*obj.poolSize(2)+1:j*obj.poolSize(2),:,:)));
                end
            end
        end
        function [out]=fullCon(obj,input,W)
            out=input*W';
        end
        function obj=Train(obj,inputlayer,output,testI,testO,lrRate,batchSize,iteration)
            obj.W{1}=rand(size(obj.W{1}))*0.2-0.1;  % initialize full connected layer weights
            obj.W{2}=rand(size(obj.W{2}))*0.2-0.1;  % initialize full connected layer weights
            obj.W{3}=rand(size(obj.W{3}))*0.2-0.1;
            obj.kernels{1}=rand(size(obj.kernels{1}))*0.2-0.1;% initialize conv layer kernels
            obj.kernels{2}=rand(size(obj.kernels{2}))*0.2-0.1;% initialize conv layer kernels
            N=size(inputlayer,4); % training ex
            batch=randperm(N); % batch
            eps=1e-8;
            B1=0.9;
            B2=0.999;
            B1pow=1;
            B2pow=1;
            VW1=zeros(size(obj.W{1}));
            VW2=zeros(size(obj.W{2}));
            VW3=zeros(size(obj.W{3}));
            VK1=zeros(size(obj.kernels{1}));
            VK2=zeros(size(obj.kernels{2}));
            SW1=zeros(size(obj.W{1}));
            SW2=zeros(size(obj.W{2}));
            SW3=zeros(size(obj.W{3}));
            SK1=zeros(size(obj.kernels{1}));
            SK2=zeros(size(obj.kernels{2}));
            
            Traincost=zeros(1,iteration);
            testcost=zeros(1,iteration);
            
            for it=1:iteration
                tic;
                for start=1:batchSize:N
                    range=start+batchSize-1;
                    if range>N
                        range=N;
                    end
                    X=inputlayer(:,:,:,batch(start:range)); % batch input
                    Y=output(batch(start:range)); % batch output
                    [Out,Clayer]=obj.testmodel(X); % forward pass
                    err=(Out-Y)'*(Out-Y)/batchSize; % MS error
                    cost=err;
                    gredFC3=(Out-Y); % gredient w.r.t. full-conn layer weights
                    gredW3=gredFC3'*[ones(size(Clayer{4}.out,1),1) Clayer{4}.out]/size(Y,1);
                    gredFC2=obj.gredRelu(gredFC3*obj.W{3}(:,2:end),Clayer{4}); % gredient w.r.t. full-conn layer weights
                    gredW2=gredFC2'*[ones(size(Clayer{3}.out,1),1) Clayer{3}.out]/size(Y,1);
                    gredFC1=obj.gredRelu(gredFC2*obj.W{2}(:,2:end),Clayer{3});
                    gredW1=gredFC1'*[ones(size(Clayer{2}.out,1),1) Clayer{2}.out]/size(Y,1); % gredient w.r.t. full-conn layer weights
                    
                    gredP2=gredFC1*obj.W{1}(:,2:end); % gredient w.r.t. input of full-conn
                    gredP2=reshape(gredP2',size(Clayer{2}.pool)); % gredient w.r.t. output of pool
                    
                    gredC2=obj.gredPool(gredP2,Clayer{2});
                    gredC2=obj.gredRelu(gredC2,Clayer{2});
                    DK2=obj.gredConvKernel(Clayer{1}.pool,gredC2,obj.kernels{2});
                    
                    gredP1=obj.gredConv(Clayer{1}.pool,gredC2,obj.kernels{2});
                    gredC1=obj.gredPool(gredP1,Clayer{1});
                    gredC1=obj.gredRelu(gredC1,Clayer{1});
                    DK1=obj.gredConvKernel(X,gredC1,obj.kernels{1});
                    
                    % update:::::
                    %obj.kernels{1}=obj.kernels{1}-lrRate*DK1;
                    %obj.kernels{2}=obj.kernels{2}-lrRate*DK2;
                    %obj.W{1}=obj.W{1}-lrRate*gredW1;
                    %obj.W{2}=obj.W{2}-lrRate*gredW2;
        
                    VW1=B1*VW1+(1-B1)*gredW1;
                    VW2=B1*VW2+(1-B1)*gredW2;
                    VW3=B1*VW3+(1-B1)*gredW3;
                    VK1=B1*VK1+(1-B1)*DK1;
                    VK2=B1*VK2+(1-B1)*DK2;
                    
                    SW1=B2*SW1+(1-B2)*(gredW1.^2);
                    SW2=B2*SW2+(1-B2)*(gredW2.^2);
                    SW3=B2*SW3+(1-B2)*(gredW3.^2);
                    SK1=B2*SK1+(1-B2)*(DK1.^2);
                    SK2=B2*SK2+(1-B2)*(DK2.^2);
                    
                    B1pow=B1pow*B1;
                    B2pow=B2pow*B2;
                    
                    
                    VCW1=VW1/(1-B1pow);
                    VCW2=VW2/(1-B1pow);
                    VCW3=VW3/(1-B1pow);
                    VCK1=VK1/(1-B1pow);
                    VCK2=VK2/(1-B1pow);
                    
                    SCW1=SW1/(1-B2pow);
                    SCW2=SW2/(1-B2pow);
                    SCW3=SW3/(1-B2pow);
                    SCK1=SK1/(1-B2pow);
                    SCK2=SK2/(1-B2pow);
                    
                    obj.kernels{1}=obj.kernels{1}-lrRate*VCK1./(SCK1.^0.5+eps);
                    obj.kernels{2}=obj.kernels{2}-lrRate*VCK2./(SCK2.^0.5+eps);
                    obj.W{1}=obj.W{1}-lrRate*VCW1./(SCW1.^0.5+eps);
                    obj.W{2}=obj.W{2}-lrRate*VCW2./(SCW2.^0.5+eps);
                    obj.W{3}=obj.W{3}-lrRate*VCW3./(SCW3.^0.5+eps);
                    
                    
                    
                end
                
                for i=1:batchSize:N-batchSize+1
                    [Out,~]=obj.testmodel(inputlayer(:,:,:,i:i+batchSize-1));
                    Traincost(it)=Traincost(it)+(Out-output(i:i+batchSize-1))'*(Out-output(i:i+batchSize-1))/N;
                end
                N=length(testO);
                for i=1:batchSize:N-batchSize+1
                    [Out,~]=obj.testmodel(testI(:,:,:,i:i+batchSize-1));
                    testcost(it)=testcost(it)+(Out-testO(i:i+batchSize-1))'*(Out-testO(i:i+batchSize-1))/N;
                end
                N=length(output);
                fprintf('iteration:%d train cost:%f test cost:%f\n',it,Traincost(it),testcost(it));
                
                toc;
            end
        end
        
        function [gredin]=gredPool(obj,gredout,Clayer)
			[a,b,~,~]=size(gredout); % a*b image dim. of pool output image
			gredin=zeros(size(Clayer.relu)); % gredient w.r.t. output of relu will be calc.
			for  i=1:a
				for j=1:b % (i,j) a single pixel of pool
					A=reshape(Clayer.relu((i-1)*obj.poolSize(1)+1:i*obj.poolSize(1),...
					    (j-1)*obj.poolSize(2)+1:j*obj.poolSize(2),:,:),obj.poolSize(1)*...
					    obj.poolSize(2),size(Clayer.relu,3),size(Clayer.relu,4)); % A=2d patch corr. to pool(i,j,:,:)
					[~,Mj]=max(A); % id of max(A) which goes to pool output
					for k=1:size(Mj,2)
					    for l=1:size(Mj,3)
					        gredin((i-1)*obj.poolSize(1)+mod((Mj(1,k,l)-1),obj.poolSize(1))+1,...
					            (j-1)*obj.poolSize(2)+floor((Mj(1,k,l)-1)/obj.poolSize(1))+1,k,l)...
					            =gredout(i,j,k,l); % gred w.r.t. to patch-max in relu output(pool input)
					    end
					end
				end
			end
        end
        
        function [gredin]=gredConv(obj,input,gredout,kernel)
            gredin=zeros(size(input));
            kernel=flip(kernel,3);
            for m=1:size(gredout,3)
                for n=1:size(gredin,3)
                    gredin(:,:,n,:)=gredin(:,:,n,:)+convn(gredout(:,:,m,:),kernel(:,:,n,m),'full'); % gredient w.r.t. kernel
                end
            end
        end

        function [gredin]=gredConvKernel(obj,input,gredout,kernel)
            gredin=zeros(size(kernel));
            input=input(end:-1:1,end:-1:1,end:-1:1,end:-1:1);
            for l=1:size(gredin,4)
                gredin(:,:,:,l)=convn(input,gredout(:,:,l,:),'valid')/size(gredout,4); % gredient w.r.t. kernel
            end
        end

        
        function [gredout]=gredRelu(obj,gredout,Clayer)
            gredout(Clayer.conv<0)=0; % gred w.r.t. input of relu(conv. output)        
        end
    end
end

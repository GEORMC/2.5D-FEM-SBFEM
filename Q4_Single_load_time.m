  clear all;
  
t=-.45:.01:.45;
dt=t(2)-t(1);
TH=zeros(size(t)); pos0=find(t==0); TH(pos0)=1/dt;
Freq=2:2:512 
% U=fftshift(fft(TH))/length(t);
% Amplitude=fftshift(fft(TH))/length(t); figure;subplot(2,1,1);hold on;plot(t,TH); subplot(2,1,2);hold on;plot(Freq,abs(Amplitude));

%__________________________________________________________________________        
%% Time to Frequency Domain Transformation. +<<>><<>><<>><<>><<>><<>><<>>+

% OMEGA is circular Frequency [rad/sec]
  OMEGA = 2*pi*Freq;
  
  


%%
fide = fopen('12march2014zdirection1.fdneut');
tliner = fgets(fide);
t=0;
while ischar(tliner)
    
    t=t+1;
    tliner = fgets(fide);
% Some Initial Datas ! ---------------------------------------------------
 
 if strncmpi(tliner,'   NO. OF NODES',9)
    a = dlmread('12march2014zdirection1.fdneut', '',[t+1,0,t+1,0]);
 end
% Nodal Coordinates ------------------------------------------------------
 
 if strncmpi(tliner,'NODAL COORDINATES',9)
    NODCplatformhorzintal = dlmread('12march2014zdirection1.fdneut', '',[t+1,1,t+a(1),3]);% NODC is nodal coor.
 end
% NODE ELEMENT or Node Connectivity MaTriX ------------------------------
 
 if strncmpi(tliner,'GROUP',5)

 ii = (tliner(1:1,26:36));
 jj= (tliner(1:1,43:56));
 ii=str2num(ii);
 jj=str2num(jj);

 end

 if strncmpi(tliner,'ENTITY NAME:   fluid',16)
    Node_Elementplatformhorzintal = dlmread('12march2014zdirection1.fdneut', '',[t+1,1,t+ii,jj]);
   
 end
    

end
%% Input Mesh Data form Gambit 
  ffttj=0;  


fide = fopen('MESH.fdneut');
tliner = fgets(fide);
t=0;

while ischar(tliner)
    
    t=t+1;
    tliner = fgets(fide);

%% Some Initial Datas ! ---------------------------------------------------
 
 if strncmpi(tliner,'   NO. OF NODES',9)
    a = dlmread('MESH.fdneut', '',[t+1,0,t+1,0]);
 end
 
%% Nodal Coordinates ------------------------------------------------------
 
 if strncmpi(tliner,'NODAL COORDINATES',9)
    NODC = dlmread('MESH.fdneut', '',[t+1,1,t+a(1),2]);% NODC is nodal coor.
 end
 
%% NODE ELEMENT or Node Connectivity MaTriX ------------------------------
 
 if strncmpi(tliner,'GROUP',5)

 ii = (tliner(1:1,26:36));
 jj= (tliner(1:1,43:56));
 ii=str2num(ii);
 jj=str2num(jj);

 end

 if strncmpi(tliner,'ENTITY NAME:   fluid',16)
    Node_Element = dlmread('MESH.fdneut', '',[t+1,1,t+ii,jj]);
   
 end
 
%% Up side 1 **************************************************************

   if strncmpi(tliner,'GROUP:        2 ELEMENTS',24)
       
       ii = (tliner(1:1,26:36));
       jj = (tliner(1:1,43:56));
       ii=str2num(ii);
       jj=str2num(jj);
   end
       
   if strncmpi(tliner,'GROUP:        2 ELEMENTS',24)
       us1 = dlmread('MESH.fdneut', '',[t+2,1,t+1+ii,jj]); 
   end
   
%% Up side 2 **************************************************************
   
   if strncmpi(tliner,'GROUP:        3 ELEMENTS',24)
       
       ii = (tliner(1:1,26:36));
       jj = (tliner(1:1,43:56));
       ii=str2num(ii);
       jj=str2num(jj);
   end
       
   if strncmpi(tliner,'GROUP:        3 ELEMENTS',24)
       us2 = dlmread('MESH.fdneut', '',[t+2,1,t+1+ii,jj]); 
   end
   
%% Right side 1 ***********************************************************   

   if strncmpi(tliner,'GROUP:        4 ELEMENTS',24)
       
       ii = (tliner(1:1,26:36));
       jj = (tliner(1:1,43:56));
       ii=str2num(ii);
       jj=str2num(jj);
   end
       
   if strncmpi(tliner,'GROUP:        4 ELEMENTS',24)
       rs1 = dlmread('MESH.fdneut', '',[t+2,1,t+1+ii,jj]); 
   end
   
%% Right side 2 ***********************************************************
   if strncmpi(tliner,'GROUP:        5 ELEMENTS',24)
       
       ii = (tliner(1:1,26:36));
       jj = (tliner(1:1,43:56));
       ii=str2num(ii);
       jj=str2num(jj);
   end
       
   if strncmpi(tliner,'GROUP:        5 ELEMENTS',24)
       rs2 = dlmread('MESH.fdneut', '',[t+2,1,t+1+ii,jj]); 
   end 
%    
%% Left side 1 ************************************************************   

   if strncmpi(tliner,'GROUP:        6 ELEMENTS',24)
       
       ii = (tliner(1:1,26:36));
       jj = (tliner(1:1,43:56));
       ii=str2num(ii);
       jj=str2num(jj);
   end
       
   if strncmpi(tliner,'GROUP:        6 ELEMENTS',24)
       ls1 = dlmread('MESH.fdneut', '',[t+2,1,t+1+ii,jj]); 
   end
   

%% Left side 2 ***********************************************************
     if strncmpi(tliner,'GROUP:        7 ELEMENTS',24)
       
       ii = (tliner(1:1,26:36));
       jj = (tliner(1:1,43:56));
       ii=str2num(ii);
       jj=str2num(jj);
    end
       
   if strncmpi(tliner,'GROUP:        7 ELEMENTS',24)
       ls2 = dlmread('MESH.fdneut', '',[t+2,1,t+1+ii,jj]); 
   end

end

%% Assemble Boundaries ...

us = [us1;us2];
rs = [rs1;rs2];
% ds = [ds1];
ls = [ls1;ls2];

%% Data for Preproceassor Part --------------------------------------------

% Define # of Elements & # of Nodes in near field.
ne_FE = size(Node_Element,1);
nn_FE = size(NODC,1);

% Define Nodal coordinates X & Y.
x = NODC(:,1);
y = NODC(:,2);

X = x';
Y = y';

%% Define NODE ELEMENT or Node Connectivity MaTriX.
nee = Node_Element';

%% Creating MaTriX nddc, Node degree of freedom MaTriX.

nddc = zeros(3,(nn_FE));

% According to related example page 255, left side of the model is fixed in
% X direction, down & right sides are fixed in both X Y direction.
 nddc(1,ls(:))=1;
% nddc(1:2,ds(:))=1;
% nddc(1:2,rs(:))=1;

%% Creating Matrix nd   nd = Node Degree of freedom
nd = zeros(3,nn_FE);
k=1;
for i=1:nn_FE;
    for j=1:3;
        if nddc(j,i)==1;
            nd(j,i)=0;
        else
            nd(j,i)=k;
            k=k+1;
        end
    end
end
NOd=nd;

%% Creating Matrix cn   cn = node connectivity
cn = zeros(12,ne_FE);
for i=1:ne_FE;
    cn(1,i)=nd(1,nee(1,i));
    cn(2,i)=nd(2,nee(1,i));
    cn(3,i)=nd(3,nee(1,i));
    %----------------------
    cn(4,i)=nd(1,nee(2,i));
    cn(5,i)=nd(2,nee(2,i));
    cn(6,i)=nd(3,nee(2,i));
    %----------------------
    cn(7,i)=nd(1,nee(3,i));
    cn(8,i)=nd(2,nee(3,i));
    cn(9,i)=nd(3,nee(3,i));
    %----------------------
    cn(10,i)=nd(1,nee(4,i));
    cn(11,i)=nd(2,nee(4,i));
    cn(12,i)=nd(3,nee(4,i));
%     %----------------------
%     cn(13,i)=nd(1,nee(5,i));
%     cn(14,i)=nd(2,nee(5,i));
%     cn(15,i)=nd(3,nee(5,i));
%     %----------------------
%     cn(16,i)=nd(1,nee(6,i));
%     cn(17,i)=nd(2,nee(6,i));
%     cn(18,i)=nd(3,nee(6,i));
%     %----------------------
%     cn(19,i)=nd(1,nee(7,i));
%     cn(20,i)=nd(2,nee(7,i));
%     cn(21,i)=nd(3,nee(7,i));
%     %----------------------
%     cn(22,i)=nd(1,nee(8,i));
%     cn(23,i)=nd(2,nee(8,i));
%     cn(24,i)=nd(3,nee(8,i));
end
for fftt= 1:.5*length(OMEGA)+1;

   nd=NOd;
    omega =OMEGA(fftt+1);

clearvars -except omgh ffttj aaaa fftt aaat OMEGA Time Spectrum Amplitude Tinc st ft t tt TH omega DELTA cn nd nn_FE NOd nddc nee X Y nn_FE ne_FE NODC
%% Define concentrated load

Global_load = zeros(max(max(nd)),1);
LoAD = 3.14;%0.5*Amplitude(fftt+1)

% nd(i,j), i related to loading direction (1=X 2=Y 3=Z) & j related to node. 
Global_load(nd(2,1)) = LoAD;
%%
% *Properties*

nIPnts=2;

ndn=3;

%% Define the material properties.

% Elasticity modulus(KN/m2)
el = (5e7);

% Poisson ratio
nu = 0.25;
% Shear modulus :
G_shear = el/(2*(1+nu));
% Viscous Damping :
  Viscous = 0;
  IMAG = complex(0,1);
  elv =complex(1,0.1) ;
        
 %% Elasticity Matrix E is [6x6]
E = (el/((1+nu)*(1-2*nu)))*[1-nu nu nu 0 0 0;
                             nu 1-nu nu 0 0 0;
                             nu nu 1-nu 0 0 0;
                             0 0 0 (1-2*nu)/2 0 0;
                             0 0 0 0 (1-2*nu)/2 0;
                             0 0 0 0 0 (1-2*nu)/2];

%% Moving Load data. $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%####################******************************************************
%% Define parameter K.
% omega is excitation frequency.
%   omega = 201.1;
% omega0 is self oscillation frequency of the moving load.  
  omega0 = 0;
% C is moving load velocity (m/s).
   C =120;
% K
node_disp = zeros(2,nn_FE);

    
    ffttj=ffttj+1;
    clear LocalStiff_FE MASS_local_FE
nd=NOd;
  K =(omega - omega0)/C;
%% Gauss Points & Weights.

% Gauss Quadrature (3 x 3) < 9 Points >
% Gauss Pionts are : -0.77459  0      0.77459
% Gauss Weights are : 0.556    0.889  0.556
% Gauss Points => gpt = [XI ETA]
% gpt = [ XI         ETA    ]
  gpt = [-sqrt(0.6)   -sqrt(0.6);
          0           -sqrt(0.6);
          sqrt(0.6)   -sqrt(0.6);
          sqrt(0.6)    0;
          sqrt(0.6)    sqrt(0.6);
          0            sqrt(0.6);
         -sqrt(0.6)    sqrt(0.6);
         -sqrt(0.6)    0;
          0            0];
      
XI = gpt(:,1);
ETA = gpt(:,2);

% Gauss Weights => gwt 
  gwt = [5/9 5/9;
         8/9 5/9;
         5/9 5/9;
         5/9 8/9;
         5/9 5/9;
         8/9 5/9;
         5/9 5/9;
         5/9 8/9;
         8/9 8/9];

%% Near Field Stiffness Matrix ********************************************
LOCALSTIFF_each_elem_FE = zeros(12,12,ne_FE);
% Start ii looping through the Q8 element.
for ii=1:ne_FE;
     
    LocalStiff_FE = 0;

%% Gauss Quadrature Integration 
% Gauss Quadrature (3 x 3) <9 Points>
% Start looping over 9 (3x3) quadrature points.
    for i=1:9;
%% Shape functions for Q4 Element

N1 = 0.25 * (1-(XI(i))) * (1-(ETA(i)));
N2 = 0.25 * (1+(XI(i))) * (1-(ETA(i)));
N3 = 0.25 * (1+(XI(i))) * (1+(ETA(i)));
N4 = 0.25 * (1-(XI(i))) * (1+(ETA(i)));

%% Gradient of shape functions
% Calculate the derivation of Ni on XI & ETA.

% Ni_XI (i=1,...,4)
N1_XI = -0.25*(1-(ETA(i)));
N2_XI =  0.25*(1-(ETA(i)));
N3_XI =  0.25*(1+(ETA(i)));
N4_XI = -0.25*(1+(ETA(i)));

% Ni_ETA (i=1,...,4)
N1_ETA = -0.25*(1-(XI(i)));
N2_ETA = -0.25*(1+(XI(i)));
N3_ETA =  0.25*(1+(XI(i)));
N4_ETA =  0.25*(1-(XI(i)));

% First column is N,XI & Second column is N,ETA 
% Compute gradient @ ith. quadrature piont (XI, ETA)
grad_N = [N1_XI    N1_ETA;
          N2_XI    N2_ETA;
          N3_XI    N3_ETA;
          N4_XI    N4_ETA];

%% Define Coordinate Matrix of each element => XY
XY = [X(nee(1,ii)) X(nee(2,ii)) X(nee(3,ii)) X(nee(4,ii));
      Y(nee(1,ii)) Y(nee(2,ii)) Y(nee(3,ii)) Y(nee(4,ii))];
   
%% Compute Jacobian @ ith. quadrature point
   J =        XY    *   grad_N;
%  [2 x 2]  [2 x 4]    [4 x 2]

Jacobian = J';
inv_J = inv(Jacobian);

%% Matrix B                         
% G is [9x3n]; which n is # of nodes in the element.
% for Q8 element, n is 4, so G is [9x12]. ;)
IMAG = complex(0,1);
G = [grad_N(1,1) 0 0 grad_N(2,1) 0 0 grad_N(3,1) 0 0 grad_N(4,1) 0 0;
     grad_N(1,2) 0 0 grad_N(2,2) 0 0 grad_N(3,2) 0 0 grad_N(4,2) 0 0;
     -IMAG*K*N1  0 0 -IMAG*K*N2  0 0 -IMAG*K*N3  0 0 -IMAG*K*N4  0 0;
     0 grad_N(1,1) 0 0 grad_N(2,1) 0 0 grad_N(3,1) 0 0 grad_N(4,1) 0;
     0 grad_N(1,2) 0 0 grad_N(2,2) 0 0 grad_N(3,2) 0 0 grad_N(4,2) 0;
     0 -IMAG*K*N1  0 0 -IMAG*K*N2  0 0 -IMAG*K*N3  0 0 -IMAG*K*N4  0;
     0 0 grad_N(1,1) 0 0 grad_N(2,1) 0 0 grad_N(3,1) 0 0 grad_N(4,1);
     0 0 grad_N(1,2) 0 0 grad_N(2,2) 0 0 grad_N(3,2) 0 0 grad_N(4,2);
     0 0 -IMAG*K*N1  0 0 -IMAG*K*N2  0 0 -IMAG*K*N3  0 0 -IMAG*K*N4];


                        
% Matrix A is [6x9].
% In the formulation of matrix A, the firs matrix is [6x9] & the second one is [9x9].
A = [1 0 0 0 0 0 0 0 0;
     0 0 0 0 1 0 0 0 0;
     0 0 0 0 0 0 0 0 1;
     0 0 0 0 0 1 0 1 0;
     0 0 1 0 0 0 1 0 0;
     0 1 0 1 0 0 0 0 0]*[inv_J(1,1) inv_J(1,2) 0 0 0 0 0 0 0;
                         inv_J(2,1) inv_J(2,2) 0 0 0 0 0 0 0;
                         0          0          1 0 0 0 0 0 0;
                         0 0 0 inv_J(1,1) inv_J(1,2) 0 0 0 0;
                         0 0 0 inv_J(2,1) inv_J(2,2) 0 0 0 0;
                         0 0 0 0          0          1 0 0 0;
                         0 0 0 0 0 0 inv_J(1,1) inv_J(1,2) 0;
                         0 0 0 0 0 0 inv_J(2,1) inv_J(2,2) 0;
                         0 0 0 0 0 0 0          0          1];
                     
% Matrix B is [6x3n], which for Q4 is [6x12].
%    B     =     A    *    G 
% [6 x 12] =  [6 x 9]   [9 x 12]
     B     =     A    *    G;

% ..::.. Calculation of Element Stiffness Matrix ..::.. %

% In this formulation B is complex conjugate (Hermitian), instead of complex symmetric.
% LocalStiff_FE = LocalStiff_FE + (gwt(i,1) * gwt(i,2) * B.' * E * conj(B) * det(Jacobian));
LocalStiff_FE = LocalStiff_FE + (gwt(i,1) * gwt(i,2) * B' * E * B * det(Jacobian));
    end
    LOCALSTIFF_each_elem_FE(:,:,ii) = LocalStiff_FE;
end

%% ---------Assembling of Global Near Field Stiffness MaTriX---------------

STIFFNESS_Global_FE = zeros(max(max(nd)),max(max(nd)));

for i=1:ne_FE;
    xxx=1;                              
    for j=1:12;
        if cn(j,i)~=0;
            r(xxx)=cn(j,i);
            l(xxx)=j;
            xxx=xxx+1;
        end
    end
    
    for myi=1:length(r);
        for myj=1:length(r);
            STIFFNESS_Global_FE(r(myi),r(myj))=STIFFNESS_Global_FE(r(myi),r(myj)) + LOCALSTIFF_each_elem_FE(l(myi),l(myj),i);
        end
    end
    clear r
    clear l
end

%% Near Field Mass Matrix *************************************************
MASS_local_FE = zeros(12,12,ne_FE);
% Define the value of Density
RO = 2000;


% Start ii looping through the Q8 element.
for ii=1:ne_FE;
    
     Mass = 0;

%% Define Coordinate Matrix of each element => XY
XY = [X(nee(1,ii)) X(nee(2,ii)) X(nee(3,ii)) X(nee(4,ii));
      Y(nee(1,ii)) Y(nee(2,ii)) Y(nee(3,ii)) Y(nee(4,ii))];

%% Gauss Quadrature Integration
% Gauss Quadrature (3 x 3) <9 Points>
% Start looping over 9 (3x3) quadrature points.
     for i=1:9;
%% Gradient of shape functions
% Calculate the derivation of Ni on XI & ETA.

% Ni_XI (i=1,...,4)
N1_XI = -0.25*(1-(ETA(i)));
N2_XI =  0.25*(1-(ETA(i)));
N3_XI =  0.25*(1+(ETA(i)));
N4_XI = -0.25*(1+(ETA(i)));

% Ni_ETA (i=1,...,4)
N1_ETA = -0.25*(1-(XI(i)));
N2_ETA = -0.25*(1+(XI(i)));
N3_ETA =  0.25*(1+(XI(i)));
N4_ETA =  0.25*(1-(XI(i)));

% First column is N,XI & Second column is N,ETA 
% Compute gradient @ ith. quadrature piont (XI, ETA)
grad_N = [N1_XI    N1_ETA;
          N2_XI    N2_ETA;
          N3_XI    N3_ETA;
          N4_XI    N4_ETA];

% Compute Jacobian @ ith. quadrature point
   J =        XY    *   grad_N;
%  [2 x 2]  [2 x 4]     [4 x 2]

Jacobian = J';
inv_J = inv(Jacobian);

% Shape Functions of Q4 Element.

N1 = 0.25 * (1-(XI(i))) * (1-(ETA(i)));
N2 = 0.25 * (1+(XI(i))) * (1-(ETA(i)));
N3 = 0.25 * (1+(XI(i))) * (1+(ETA(i)));
N4 = 0.25 * (1-(XI(i))) * (1+(ETA(i)));

% Calculation Matrix N.
% Size of Mtrix N in Q4 Element is [3 x 12].

N = [N1 0 0 N2 0 0 N3 0 0 N4 0 0;
     0 N1 0 0 N2 0 0 N3 0 0 N4 0;
     0 0 N1 0 0 N2 0 0 N3 0 0 N4];
 
 Mass = Mass + (RO * N' * N * det(Jacobian) * gwt(i,1) * gwt(i,2));
     
     end;
     MASS_local_FE(:,:,ii) = Mass;
end;

%% --------------Assembling of Global Near Field Mass MaTriX---------------

MASS_Global_FE = zeros(max(max(nd)),max(max(nd)));

for ii=1:ne_FE;
    xxx=1;                              
    for j=1:12;
        if cn(j,ii)~=0
            r(xxx)=cn(j,ii);
            l(xxx)=j;
            xxx=xxx+1;
        end
    end
    
    for myi=1:length(r);
        for myj=1:length(r);
            MASS_Global_FE(r(myi),r(myj))= MASS_Global_FE(r(myi),r(myj)) + MASS_local_FE(l(myi),l(myj),ii);
        end
    end
    clear r
    clear l
end

%% ------------------------------------------------------------------------

STIFFNESS_Global_Total = (STIFFNESS_Global_FE);

MASS_Global_Total = (MASS_Global_FE);

%% Calculation of [K - Mw^2] in Frequency domain :
% S is Impedance function [K - Mw^2]

W = ((omega)^2);
% S = ( ((1+2*IMAG*HYSTERETIC)*(STIFFNESS_Global_Total)) - W * (MASS_Global_Total) );
Sb = (((STIFFNESS_Global_Total)*elv) - W * (MASS_Global_Total));

%% Calculation of DELTA, {f} = [S] x {DELTA}
% Huge attention :
% GLOBAL-STIFFNESS-MATRIX is summation of FE\IE stiff. matrix.
% FORCE-VECTOR comes from previous section (Creating external force vector)

%

%% Creating Node Deformation Matrix <<ndd shows disp. of all nodes>>



% node_con_IE is node connectivity matrix for quarter hole problem.
NC_LS = load('Node Connectivity Left.txt');
NC_DS = load('Node Connectivity Down.txt');
NC_RS = load('Node Connectivity Right.txt');

node_con_IE = [NC_LS;NC_DS;NC_RS];
           
Node_Element_IE = (node_con_IE)';

%% Read Nodal Coordinates from MESH2.fdneut 
 %ss = load('IE_Nodal_Coordinate.txt');

% adding IE nodal coordinate to main nodal coordinate matrix. 
NODC_IE = [NODC];

% Define # of Elements & # of Nodes in far field. this part complete
% manually
ne_IE = size(node_con_IE,1);
nn_IE = size(ss,1);

% Define Nodal coordinates X & Y.
x_x = NODC_IE(:,1);
y_y = NODC_IE(:,2);

X_IE = x_x';
Y_IE = y_y';

%% Creating Matrix node_dof_IE   node_dof_IE = node degree of freedom
k=1;
for i=1:nn_FE;
    for j=1:2;
        if nddc(j,i)==1;
            node_dof_IE(j,i)=0;
        else
            node_dof_IE(j,i)=k;
            k=k+1;
        end
            
    end
end
%% Creating Matrix node_conectivity_IE  node_conectivity_IE = node connectivity
ennodes=2;
for i=1:ne_IE;
    for j=1:ennodes
    node_conectivity_IE(2*j-1,i)=node_dof_IE(1,Node_Element_IE(j,i));
    node_conectivity_IE(2*j,i)=node_dof_IE(2,Node_Element_IE(j,i));
    end
end

% %% Far Filed Data.
% %=========================================================================%
Node_Element_IE=Node_Element_IE';
%% Far Field Stiffness Matrix *********************************************
% Start ii looping through the Q8 element.

ndn=3;
iv1=zeros(nn_FE,1);
iasy=1;
  elv =complex(1,0.1) ;
si=0;
is=15;
% omega=201.1;
for ii=1:ne_IE
    sd(1).elem(ii).NODE=Node_Element_IE(ii,:);
    for i=1:ennodes
        if Node_Element_IE(ii,i)~=0;
            iv1(Node_Element_IE(ii,i),1)=1;
        end
    end
end
j=0;
sd(1).np=sum(iv1(:,1));
for ii=1:nn_FE

        if iv1(ii)~=0;
           j=j+1;
           sd(1).sdnode(j).Gid=ii;
           sd(1).sdnode(j).coord=NODC(ii,:);
            sd(1).sdnode(j).dof(1:3)=1;
        end
    
end

for ii=1:ne_IE
    for j=1:ennodes;
        ts=sd(1).elem(ii).NODE(j);
        if ts==0
        globalNodeNo = 0;
        else
        k=1;
          while ts~=sd(1).sdnode(k).Gid && k<sd(1).np 
          k=k+1;
          end 
        if k==1 && ts~=sd(1).sdnode(1).Gid ;
            k=-1;
        end
        if  k==sd(1).np && ts~=sd(1).sdnode(sd(1).np).Gid 
            k=-1;
        end
        globalNodeNo = k;
        end
       sd(1).elem(ii).NODE(j)=globalNodeNo;
    end  
    
end
ND=0;
for ii=1:sd(1).np
    
    for i=1:ndn
        if sd(1).sdnode(ii).dof(i)~=0;
            ND=ND+1;
            sd(1).sdnode(ii).dof(i)=ND;
        end
    end
end
sd(1).ndof=ND;

            for i = 1:ne_IE
             sd(1).elem(i).nNodalDofs=ennodes*ndn;
	 
            m = 0;
            for j = 1: ennodes
            for k = 1:ndn
             m = m + 1;
             sd(1).elem(i).nodalDof(m) = sd(1).sdnode(sd(1).elem(i).NODE(j)).dof(k);
            end  
           end  
           end  



[ Gauss_tbl ] =initialize_Gauss_table();

 sd(1).coefmtx(sd(1).ndof,sd(1).ndof,1:6)=0;
 sd(1).m0(sd(1).ndof,sd(1).ndof)=0;
for ii=1:ne_IE;
   e= sd(1).elem(ii);
   

    LocalStiff_IE = 0;
    SCALING_CENTER=[0;0];
    xe=[];
   for sci=1:ennodes
        k=sd(1).elem(ii).NODE(sci);
        if k~=0
     xe(:,sci)=(sd(1).sdnode(k).coord)'-SCALING_CENTER;
        end
   end
   clear sci k

                  estf(1:ennodes*ndn,1:ennodes*ndn,1)=0;
   estf(1:ennodes*ndn,1:ennodes*ndn,2)=0;
   estf(1:ennodes*ndn,1:ennodes*ndn,3)=0;
   estf(1:ennodes*ndn,1:ennodes*ndn,4)=0;
   estf(1:ennodes*ndn,1:ennodes*ndn,5)=0;
   estf(1:ennodes*ndn,1:ennodes*ndn,6)=0;
   efm(1:ennodes*ndn,1:ennodes*ndn)=0;
    for i=1:nIPnts;

   
   lclxy=Gauss_tbl(nIPnts).xi(i);
   wgt=Gauss_tbl(nIPnts).wgt(i);
    [shp ] = shapeFunction1d(ennodes ,lclxy);
    xpg=0;
    for iisc=1:ennodes
                xpg=shp(:,iisc)*xe(:,iisc)'+xpg;
    end
    jdet=det(xpg);
    xpg=inv(xpg);
    if jdet  <0
        disp 'ill-conditionaed element'
    end
    b=0;
    
    %     loop over each node 
        for iy=1:ennodes
       
        jy=3*(iy-1);
%       [B1]  
        a1=xpg(1,1)*shp(1,iy);
        b(1,jy+1,1)=complex(a1,0.d0);
        b(6,jy+2,1)=complex(a1,0.d0);
        b(5,jy+3,1)=complex(a1,0.d0);
        a1=xpg(2,1)*shp(1,iy);
        b(2,jy+2,1)=complex(a1,0.d0);
        b(4,jy+3,1)=complex(a1,0.d0);
        b(6,jy+1,1)=complex(a1,0.d0);
%       [B2]  
        a1=xpg(1,2)*shp(2,iy);
        b(1,jy+1,2)=complex(a1,0.d0);
        b(6,jy+2,2)=complex(a1,0.d0);
        b(5,jy+3,2)=complex(a1,0.d0);
        a1=xpg(2,2)*shp(2,iy);
        b(2,jy+2,2)=complex(a1,0.d0);
        b(4,jy+3,2)=complex(a1,0.d0);
        b(6,jy+1,2)=complex(a1,0.d0);
%       [B3]
             
        b(3,jy+3,3)=complex(0.d0,-K*shp(1,iy));
        b(4,jy+2,3)=complex(0.d0,-K*shp(1,iy));
        b(5,jy+1,3)=complex(0.d0,-K*shp(1,iy));
        end
     w=wgt*jdet;




  

   %% Elasticity Matrix E is [6x6]
E = (el/((1+nu)*(1-2*nu)))*[1-nu nu nu 0 0 0;
                             nu 1-nu nu 0 0 0;
                             nu nu 1-nu 0 0 0;
                             0 0 0 (1-2*nu)/2 0 0;
                             0 0 0 0 (1-2*nu)/2 0;
                             0 0 0 0 0 (1-2*nu)/2];
    estf(:,:,1)=b(:,:,1)'*E*b(:,:,1)*w +estf(:,:,1);   
    estf(:,:,2)=b(:,:,2)'*E*b(:,:,1)*w +estf(:,:,2);
    estf(:,:,3)=b(:,:,2)'*E*b(:,:,2)*w +estf(:,:,3);
    estf(:,:,4)=b(:,:,3)'*E*b(:,:,1)*w +estf(:,:,4);
    estf(:,:,5)=b(:,:,3)'*E*b(:,:,2)*w +estf(:,:,5);
    estf(:,:,6)=b(:,:,3)'*E*b(:,:,3)*w +estf(:,:,6);
%c    Transponse[N].<den>.[N]*|J|*(Gauss integration weight) 

    for iy=1:ennodes
        for jy=1:ennodes
           tmpb=shp(1,iy)*shp(1,jy)*w;
           for k=1:ndn
            efm(ndn*(iy-1)+k,ndn*(jy-1)+k)=...
                       efm(ndn*(iy-1)+k,ndn*(jy-1)+k)+tmpb*RO;
           end               
        end
    end
    
    
    im=1;
    end
    
        for j=1:e.nNodalDofs             
          k=e.nodalDof(j);
          if(k~=0) 
%            for l=1,nedf
            for l=1:e.nNodalDofs
              m=e.nodalDof(l);
              if(m~=0) 
%                 if ( s.sparseFormat .eq. 0) then
                   sd(1).coefmtx(k,m,1:6)=sd(1).coefmtx(k,m,1:6)+estf(j,l,1:6);
                   if(im~=0) 
                       sd(1).m0(k,m)=sd(1).m0(k,m)+efm(j,l);
                   end
%                 end
%                 if (s.sparseFormat.eq.2) then
%                  ipos = FindArrayIndex(s.ia,s%idiag(m),s%idiag(m+1)-1,k) !ltx column major
%                 else
%                  ipos = FindArrayIndex(s%ia,s%idiag(k),s%idiag(k+1)-1,m) !ltx row major
%                 end 
%                 s.e0p(ipos) = s.e0p(ipos)+estf(j,l,1)
%                 s.e1p(ipos) = s.e1p(ipos)+estf(j,l,2)
%                 s.e2p(ipos) = s.e2p(ipos)+estf(j,l,3)
%                 s.e3p(ipos) = s.e3p(ipos)+estf(j,l,4)
%                 s.e4p(ipos) = s.e4p(ipos)+estf(j,l,5)
%                 s.e5p(ipos) = s.e5p(ipos)+estf(j,l,6)
%                 if(im~=0) 
%                     s.m0p(ipos)=s.m0p(ipos)+efm(j,l)
%                 end
              end 
            end 
          end 
        end 
    
    

%% Weight Factors for Integration (similar to Newton-Cotes Method)
% Bettess and Zienkiewicz proposed this method (1977) 



end


if omega==0


    e0=sd(1).coefmtx(:,:,1) ;
    e1=sd(1).coefmtx(:,:,2) ;
    e2=sd(1).coefmtx(:,:,3) ;

    
    
    
    nd=sd(1).ndof;
    
    nd2=nd*2;
%_______________________________ Riccati__________________________

 bbb(nd+1:2*nd,1:nd)=(e1/e0)*e1'-e2;
 bbb(1:nd,1:nd)=-e0\(e1');
  bbb(nd+1:2*nd,nd+1:2*nd)=(e1)/e0;
   bbb(1:nd,nd+1:2*nd)=-eye(nd,nd)/e0;
  S=zRiccati(nd,bbb);
  
    R=-1*eye(nd,nd);
   R=R^(-1);
% S=care(-e0\(e1'),-eye(nd,nd)/e0,-((e1/e0)*e1'-e2),R);
S= Alriccati( bbb );
  

else
    
        zq=sd(1).m0;
   zqinv= sd(1).coefmtx(:,:,1);
    nd=sd(1).ndof;
    wr(nd )=0;
    mtmp=zeros(nd*3,1); 

    D = lapack('dsygv',1,'V','U',nd,zq,nd,zqinv,nd,wr,mtmp,nd*3,i);
    zq=D{([5])};
    wr=D{([9])};
    e0=zq'*sd(1).coefmtx(:,:,1)*zq ;
    e1=zq'*sd(1).coefmtx(:,:,2)*zq ;
    e2=zq'*sd(1).coefmtx(:,:,3)*zq ;
    e3=zq'*sd(1).coefmtx(:,:,4)*zq ;
    e4=zq'*sd(1).coefmtx(:,:,5)*zq ;
    e5=zq'*sd(1).coefmtx(:,:,6)*zq ;
%     m0=zq'*sd(1).m0*zq*elv;
for j=1:nd
    m0(j,j)=wr(j);
end

    nd2=nd*2;
%_______________________________ Riccati__________________________

  bbb(nd+1:2*nd,1:nd)=(-(-e3*e3'+e5+omega^2*m0));
  bbb(1:nd,1:nd)=(e3'*complex(0,1));
   bbb(nd+1:2*nd,nd+1:2*nd)=(-(e3)'*complex(0,1));
   bbb(1:nd,nd+1:2*nd)=-eye(nd,nd);
   CINF=zRiccati(nd,bbb);


%  bbb(nd+1:2*nd,1:nd)=real(-(e3*e3.'-e5+omega^2*m0));
%  bbb(1:nd,1:nd)=imag(e3');
%   bbb(nd+1:2*nd,nd+1:2*nd)=imag(-(e3));
%    bbb(1:nd,nd+1:2*nd)=-eye(nd,nd);
%   CINF=zRiccati(nd,bbb);

%   CINF = Alriccati( bbb );
%    CINF=(CINF+CINF')/2
  R=eye(nd,nd);
%   R=R^(-1);
      CINF=care((e3'*complex(0,1)),R,(-e3*e3'+e5+omega^2*m0));

%       CINF=care((e3.'),R,(e3*e3'+e5+omega^2*m0));
%     
%       CINF = newton_care(e3.',R,(e3*e3'+e5+omega^2*m0),CINF);
% X=NEWTON_CARE(A,B,C,X0) solves the CARE C + XA + A'X - XBX = 0
%    CINF=CINF*complex(0,1)
error=-e3*e3'+e5+omega^2*m0-e3*complex(0,1)*CINF+CINF*complex(0,1)*e3'-CINF*CINF;
%  CINF=(CINF+CINF')/2;
%  [CINF]= sign_care(e3',-eye(nd,nd),e3*e3'-e5+omega^2*m0)
%  error=e3*e3'-e5+omega^2*m0+e3*CINF+CINF*e3'+CINF*CINF;
% CINF=(CINF+CINF')/2;
%________________________________LYAPUNOV___________________________
%  KINF =CINF*0;
% R=chol(e1*(CINF'+e3')+(CINF+e3)*e1'-e4-e4'-CINF);
% X=lyapchol(e3+CINF,R)
% KINF=X*X';
%   CINF=CINF*complex(0,1);
 KINF=lyapunov( (-e3+CINF*complex(0,1)), (e3'+CINF*complex(0,1)), (e1*(CINF*complex(0,1)+e3')+(CINF*complex(0,1)-e3)*e1'-e4+e4'-CINF*complex(0,1))*(-1) ,nd);
%   KINF = lyap(e3+CINF,(e3'+CINF'),e1*(CINF'+e3')+(CINF+e3)*e1'-e4-e4'-CINF) ;
 error=(-e3+CINF*complex(0,1))*KINF+KINF*(e3'+CINF*complex(0,1))+e1*(CINF*complex(0,1)+e3')+(CINF*complex(0,1)-e3)*e1'-e4+e4'-CINF*complex(0,1);
%  CINF=CINF*(complex(1,.1))^.5;
%  KINF=KINF*(complex(1,.1));
%  KINF=(KINF+KINF')/2;
% A1=lyap(e3+CINF,(CINF'+e3'),(KINF+e1)*(KINF'+e1')-e2) ;
  A=lyapunov((-e3+CINF*complex(0,1))/complex(0,1),(CINF*complex(0,1)+e3')/complex(0,1),((KINF+e1)*(KINF+e1')-e2)*(-1),nd) ;
% % 
  error=(-e3+CINF*complex(0,1))*A/complex(0,1)+A/complex(0,1)*(e3'+CINF*complex(0,1))-(KINF+e1)*(KINF+e1')+e2;
% A=(A+A')/2;
% A=A1;
% CINF=CINF*complex(0,1)
 CINF=(zq')\CINF/(zq);
 KINF=(zq')\KINF/(zq);
 A=(zq')\A/(zq);
 if iasy>=2
    
     
     
     A(:,:,2)=(KINF+e1)*A(:,:,1)+A(:,:,1)*(KINF+e1')-A(:,:,1);
     A(:,:,2)=lyapunov(e3+CINF,e3'+CINF, A(:,:,2)*(-1),nd) ;
       error=(e3+CINF)*A(:,:,2)+A(:,:,2)*(e3'+CINF)+ (KINF+e1)*A(:,:,1)+A(:,:,1)*(KINF+e1')-A(:,:,1);
       for i = 2:iasy-1
           AA=(KINF+e1)*A(:,:,i)+A(:,:,i)*(KINF+e1')+(-1)^j*factorial(j)*A(:,:,i);
                   for j = 1: (i-1)
                       AA=A(:,:,j)*A(:,:,i-j)+AA;
                   end 
                      A(:,:,i+1)=lyapunov(e3+CINF,e3'+CINF, AA*(-1),nd) ;
       error=(e3+CINF)*A(:,:,i+1)+A(:,:,i+1)*(e3'+CINF)+ AA;
       end
     
 end



 
S=CINF*is*complex(0,1)*elv^.5+KINF*elv+A/complex(0,1)/is*elv^1.5; %CINF*is
%                    for j = 1:iasy
%                       S=A(:,:,j)/is^(j)+S;
%                    end 
SE=CINF*complex(0,1)*elv^.5-A/complex(0,1)/is^2*elv^.5*1.5;
%                    for j = 1:iasy
%                       SE=(-1)^j*factorial(j)*A(:,:,j)+SE;
%                    end 
%         AA=(KINF+e1)*A(:,:,1)+A(:,:,1)*(KINF+e1')-A(:,:,1);           
%        for i = 2:iasy
%            AA=(KINF+e1)*A(:,:,i)+A(:,:,i)*(KINF+e1')-A(:,:,i);
%                    for j = 1: (i-1)
%                        AA=A(:,:,j)*A(:,:,i-j)+AA;
%                    end 
% 
%        end

% ERROR=(e3*is+e1+S)*(e3'*is+e1'+S)-e5*is*is-e4'*is-e4*is-e2+omega^2*m0*is*is-SE-((KINF+e1)*A(:,:,1)+A(:,:,1)*(KINF+e1')+A(:,:,1)+A(:,:,1)*A(:,:,1));

SEE=(e3+e1+S)*(e1'+e3'+S)-e5-e4'-e4-e2+omega^2*m0;
%options = odeset('RelTol',1e-4,'AbsTol',[1e-4 1e-4 1e-5]);

for i=1:nd
    for j=1:nd
       IC((i-1)*nd+j,1)=S(j,i);
    end
end

[SI,SS] = ode113(@(si,S) Kdyn(si,S,(zq')\e0/(zq)*elv,(zq')\e1/(zq)*elv,(zq')\e2/(zq)*elv,(zq')\e3/(zq)*elv,(zq')\e4/(zq)*elv,(zq')\e5/(zq)*elv,(zq')\m0/(zq),omega),[is 1],IC); % Solve ODe
S=[];
[a,~]=size(SS);
for i=1:nd
    for j=1:nd
       S(j,i)=(SS(a,(i-1)*nd+j));
    end
end


%  S=(zq')\S/(zq);
end


 clear SS e0 e1 e2 e3 e4 e5 m0 estf b IC SI bbb
 
ss=max(max(NOd));
SINF=zeros(ss,ss);
s=0;
dfdf=0
for i=1:sd(1).np
    for k=1:sd(1).np
    for j=1:ndn;
        for t=1:ndn;
        if NOd(j,sd(1).sdnode(1,i).Gid(1))~=0 & NOd(t,sd(1).sdnode(1,k).Gid(1))~=0;
           SINF(NOd(j,sd(1).sdnode(1,i).Gid(1)),NOd(t,sd(1).sdnode(1,k).Gid(1)))=S(sd(1).sdnode(1,i).dof(1,j),(sd(1).sdnode(1,k).dof(1,t)));
           s=s+1;
        else
            dfdf=dfdf+1
        end
        end
    end
    end
end
sum(sum(S))
sum(sum(SINF))

ST=Sb+SINF;



    DELTA(:,fftt+1 ) = ((ST)\(Global_load));
    DELTA(:,length(OMEGA)+1-fftt) = conj(DELTA(:,fftt+1)); 
    clear ST Sb SINF CINF SE   S ss IC KINF
end

%% 

TimeHistory = zeros(max(max(NOd)),length(OMEGA));

for i = 1:(max(max(NOd)));
    TimeHistory(i,:) = ifftshift(ifft(DELTA(i,:)));
end

TransDELTA = zeros(3,nn_FE,length(OMEGA));
node_disp = zeros(3,nn_FE);

for oo = 1:length(OMEGA);
    node_disp = zeros(3,nn_FE);
    DeLtA = real(TimeHistory(:,oo,1)); % or abs(TimeHistory) !!! <<<<<<<<<<<<
    for i=1:3;
        for j=1:nn_FE;
            if NOd(i,j)~=0;
            node_disp(i,j)=DeLtA(NOd(i,j));
            end
        end
    end
    TransDELTA(:,:,oo) = node_disp;
end




%% 


%% 


figure
subplot(2,1,1)
plot((4*pi*G_shear)*real(TimeHistory(NOd(2,10),:,1)),'Linewidth',2)


xlabel('Time (s)','fontweight','b')
ylabel('V_y','fontweight','b','rotation',0)

subplot(2,1,2)
plot((4*pi*G_shear)*real(TimeHistory(NOd(3,10),:,1)),'Linewidth',2)
title('Time-domain Displacements W_y, point (0,1,0 m)')
xlabel('Time (s)','fontweight','b')
ylabel('W_y','fontweight','b','rotation',0)




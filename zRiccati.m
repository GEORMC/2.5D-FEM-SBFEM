function [ zst ] = zRiccari( n,a )
%UNTITLED2 Summary of this function goes here
%select='zselect'

[U,T] = schur(a);
[z,TS] = ordschur(U,T,'lhp');




indx=zeros(n*2,1,'int32');

%     
%       for i=1:n
%        for j=1:n
%         zst(j,i)=z(n+i,j);
% 
%        end  
%       end  
%       zz(1:n,1:n)=z(1:n,1:n);
%  D = lapack('ZGETRF',n,n,zz,n*2,indx,i)
%   z=D{([3])}
%  C = lapack( 'ZGETRS','T',n,n,z,n*2,indx,zst,n,i)    
% z=C{([7])}


  zst=z(n+1:n*2,1:n)/(z(1:n,1:n));%^(-1);
end


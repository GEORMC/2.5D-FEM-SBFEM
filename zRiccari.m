function [ zst ] = zRiccati( n,a )
%UNTITLED2 Summary of this function goes here

 n2=n+n
 D = lapack('ZGEES(h,h,l,s,c,s,s,c,s,c,c,c,c,i)','V','S',zselect,n2,a,n2,1i,eign,z,n2,tmpm,n2*max(4,n2),tmpmm,bw,info)
        z=D{([9])}
    
      for i=1:n
       for j=1:n
        zst(j,i)=z(n+i,j)

       end  
      end  
 C = lapack('ZGETRF',n,n,z,n2,indx,i)
  z=D{([3])}
 C = lapack( 'ZGETRS','T',n,n,z,n2,indx,zst,n,i)    
z=C{([7])}
end


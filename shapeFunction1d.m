function [shp ] = shapeFunction1d(ennodes, c1 )
% shapeFunction1d Summary of this function goes here
%   Detailed explanation goes here








%
%     SHAPE FUNCTIONS [N] AND THEIR DERIVATIVES [N],eta
% C     FOR 2-4 NODE LINE ELEMENT
% C
% c  shp(1,j):  shape function Nj(eta) of j-th node
% c  shp(2,j):  derivative of shape function Nj(eta),eta  
% c  iep:       number of nodes of element
% c  c1:        local coordinate eta
% C
      
      if ennodes==2
%     2-node line element
       shp(1,1)=0.5d0*(1.d0-c1);
       shp(1,2)=0.5d0*(1.d0+c1);
       shp(2,1)=-0.5d0;
       shp(2,2)=0.5d0;
      elseif ennodes==3
%      3-node line element
       shp(1,3)=1.d0-c1*c1;
       shp(2,3)=-c1-c1;
       shp(1,1)=0.5d0*(1.d0-c1)-0.5d0*shp(1,3);
       shp(1,2)=0.5d0*(1.d0+c1)-0.5d0*shp(1,3);
       shp(2,1)=-0.5d0-0.5d0*shp(2,3);
       shp(2,2)=0.5d0-0.5d0*shp(2,3);
      elseif ennodes==4

% !cbgnltx
% ! Shape functions of 4-node line element
% !  \begin{align*}
% !   N_1(\xi) = & -\dfrac{1}{16}(1-\xi)(1-9\xi^2) =\dfrac{1}{16}(-1+(1+(9-9\xi)\xi)\xi)\\
% !   N_2(\xi) = & -\dfrac{1}{16}(1+\xi)(1-9\xi^2) =\dfrac{1}{16}(-1+(-1+( 9+ 9\xi)\xi)\xi)\\
% !   N_3(\xi) = &  \dfrac{9}{16}(1-\xi^2)(1-3\xi) =\dfrac{1}{16}(9+(-27+(-9+27\xi)\xi)\xi)\\
% !   N_4(\xi) = &  \dfrac{9}{16}(1-\xi^2)(1+3\xi) =\dfrac{1}{16}(9+(27+(-9-27\xi)\xi)\xi)\\
% !  \end{align*}
% ! Derivatives of shape functions
% !  \begin{align*}
% !   N_1(\xi),_{\xi} = & \dfrac{1}{16}(1+18\xi-27\xi^2) \\
% !   N_2(\xi),_{\xi} = & \dfrac{1}{16}(-1+18\xi+27\xi^2) \\
% !   N_3(\xi),_{\xi} = & \dfrac{1}{16}(-27-18\xi+81\xi^2) \\
% !   N_4(\xi),_{\xi} = & \dfrac{1}{16}( 27-18\xi-81\xi^2) \\
% !  \end{align*}
% ! Integrations of shape functions
% !  \begin{align*}
% !   \int^{+1}_{-1}N_1(\xi)\mathrm{d}\xi = &  \dfrac{1}{16}(-2+9\times\dfrac{2}{3}) = \dfrac{1}{4}\\
% !   \int^{+1}_{-1}N_2(\xi)\mathrm{d}\xi = &  \dfrac{3}{4} \\
% !   \int^{+1}_{-1}N_3(\xi)\mathrm{d}\xi = & \dfrac{3}{4} \\
% !   \int^{+1}_{-1}N_4(\xi)\mathrm{d}\xi = & \dfrac{1}{4} \\
% !  \end{align*}
% !cendltx
       shp(1,1)=(-1.d0+(1.d0+(9.d0-9.d0*c1)*c1)*c1)/16.d0;
       shp(1,2)=(-1.d0+(-1.d0+(9.d0+9.d0*c1)*c1)*c1)/16.d0;
       shp(1,3)=(9.d0+(-27.d0+(-9.d0+27.d0*c1)*c1)*c1)/16.d0;
       shp(1,4)=(9.d0+(27.d0+(-9.d0-27.d0*c1)*c1)*c1)/16.d0;
       shp(2,1)=(1.d0+(18.d0-27.d0*c1)*c1)/16.d0;
       shp(2,2)=(-1.d0+(18.d0+27.d0*c1)*c1)/16.d0;
       shp(2,3)=(-27.d0+(-18.d0+81.d0*c1)*c1)/16.d0;
       shp(2,4)=(27.d0+(-18.d0-81.d0*c1)*c1)/16.d0;
      else
       pause
       disp 'element type not supported'
      end























end


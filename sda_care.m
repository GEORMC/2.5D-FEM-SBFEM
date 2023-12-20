function [X,Y] = sda_care(A,B,C,g)
% [X,Y]=SDA_CARE(A,B,C,g) solves the CARE C + XA + A'X - XBX = 0
% by means of SDA based on the Cayley transform
%    A, B, C: matrix coefficients
%    g: scalar gamma used by the Cayley transform
%    X: solution of the CARE
%    Y: solution of the dual CARE
n = size(A,1);
tol = 1e-37;
kmax = 90;
% Initialization
Ai = inv(A+g*eye(n));
R = B*Ai'*C;
S1 = inv(A+g*eye(n)+R);
E = S1*(A - g*eye(n) + R);
R = eye(n) - Ai*(A - g*eye(n));
G = S1*B*R';
P = -S1'*C*R;
% SDA step
err = 1;
k = 0;
while err > tol && k < kmax
    IGP = eye(n) - G*P;
    Z = [E;P']/IGP;
    E1 = Z(1:n,:);
    P1 = Z(n+1:end,:);
    G = G + E1*G*E';
    P = P + E'*P1'*E;
    E = E1*E;
    err = norm(E,1);
    k = k + 1;
end
X = P; Y = G;
if k == kmax
    disp('Warning: reached the maximum number of iterations')
end

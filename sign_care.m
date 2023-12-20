function X = sign_care(A,B,C)
% X=SIGN_CARE(A,B,C) solves the CARE C + XA + A'X - XBX = 0
% by means of the matrix sign function
%    A, B, C: matrix coefficients
%    X : solution of the CARE
n  = size(B,1);
H = [A,-B;-C,-A'];
W = matrix_sign(H) + eye(2*n);
X = -W(1:2*n,n+1:2*n)\W(1:2*n,1:n);

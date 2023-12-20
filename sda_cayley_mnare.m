function [X,Y] = sda_cayley_mnare(A,B,C,D)
% [X,Y]=SDA_CAYLEY_MNARE(A,B,C,D) solves the M-NARE C + XA + DX - XBX = 0
% by means of SDA based on the Cayley transform
%    A, B, C, D: matrix coefficients
%    X : solution of the M-NARE
%    Y : solution of the dual M-NARE
g = max(max(diag(A)),max(diag(D)));
n = size(A,1); m = size(D,1);
% Initialization by means of Cayley transform
U = [A+g*eye(n),-B;C,D+g*eye(m)];
V = [A-g*eye(n),-B;C,D-g*eye(m)];
W = U\V;
E = W(1:n,1:n);
G = -W(1:n,n+1:n+m);
P = -W(n+1:n+m,1:n);
F = W(n+1:n+m,n+1:n+m);
% SDA step
[X,Y] = sda(E,F,G,P);

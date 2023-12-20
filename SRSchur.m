function [Q,R,ap] = SRSchur(Q,R,z,b);

% SYNTAX: [Q,R,ap] = SRSchur(Q,R,z,b)
%
% INPUT: orthogonal real Q and quasi-triangular real R such that AQ=QR and a 
% target z in the complex plane. The fourth parameter b determines the length  
% of the ordering with respect to z to be produced:
%
% if b < 0 then -b blocks will be sorted,
% if b > 0 then  b or b+1 eigenvalues will be sorted, depending on the sizes 
% of the blocks,
% if b = 0 then the whole Schur form will be sorted.
%
% OUTPUT: orthogonal real Q and quasi-triangular real R such that AQ=QR with 
% the diagonal blocks ordered with respect to the target z. The number of
% ordered blocks/eigenvalues is determined by the parameter b.
% A vector ap warns for inaccuracy of the solution if an entry of ap exceeds
% one. 
% 
% SUBFUNCTIONS: normalize.m, swaplist.m, select.m, swap.m, lu_complpiv.m
%
% SEE ALSO: schur.m, rsf2csf.m

r = find(abs(diag(R,-1)) > 100*eps);
s = 1:size(R,1)+1;
s(r+1) = [];

for k=1:length(s)-1;
  sk = s(k);
  if s(k+1)-sk == 2
    [Q,R] = normalize(Q,R,sk:s(k+1)-1);
    p(k)  = R(sk,sk)+sqrt(R(sk+1,sk)*R(sk,sk+1));
  else
    p(k)  = R(s(k),s(k));
  end
end

for k = swaplist(p,s,z,b); 
  v      = s(k):s(k+1)-1; 
  w      = s(k+1):s(k+2)-1;
  nrA    = norm(R([v,w],[v,w]),inf);
  [Q,R]  = swap(Q,R,v,w); 
  s(k+1) = s(k)+s(k+2)-s(k+1);
  v      = s(k):s(k+1)-1; 
  w      = s(k+1):s(k+2)-1;
  if length(v)==2
    [Q,R] = normalize(Q,R,v);
  end
  if length(w)==2
    [Q,R] = normalize(Q,R,w);
  end
  ap(k)  = norm(R(w,v),inf)/(10*eps*nrA);
end

R = R - tril(R,-2);
for j=2:length(s)-1; R(s(j),s(j)-1)=0; end


% ----------------------------------------------%

  function [U,S] = normalize(U,S,v);
  n  = size(S,1);
  Q  = rot(S(v,v));
  S(:,v) = S(:,v)*Q;
  S(v,:) = Q'*S(v,:); 
  U(:,v) = U(:,v)*Q;

  % ----------------------------------------------%

  function Q = rot(X);
  c = 1; s = 0;
  if X(1,1)~=X(2,2);
    tau   = (X(1,2)+X(2,1))/(X(1,1)-X(2,2));
    off   = sqrt(tau^2+1);
    v     = [tau - off, tau + off];
    [d,w] = min(abs(v));
    c     = 1/sqrt(1+v(w)^2);
    s     = v(w)*c;
  end
  Q = [c -s;s c];

% ----------------------------------------------%

  function v = swaplist(p,s,z,b);
  n = length(p);
  k = 0; v = [];
  srtd = 0; 
  q = diff(s);
  fini = 0;
  while ~fini
    k        = k+1;
    [dum,j]  = select(p(k:n),z);
    p(k:n+1) = [p(j+k-1) p(k:n)];
    p(j+k)   = [];
    q(k:n+1) = [q(j+k-1) q(k:n)];
    q(j+k)   = [];
    v        = [v,j+k-2:-1:k];
    srtd     = srtd + q(k);
    fini     = (k==n-1)|(k==-b)|(srtd==b)|((srtd==b+1)&(b~=0));
  end

% ----------------------------------------------%

  function [val,pos] = select(p,z);
  y = real(z)+abs(imag(z))*i;
  [val pos] = min(abs(p-y));

% -----------------------------------------------%

  function [U,S] = swap(U,S,v,w);  
  [p,q] = size(S(v,w)); Ip = eye(p); Iq = eye(q);
  r = [];
  for j=1:q
    r = [r;S(v,w(j))];
  end 
  K = kron(Iq,S(v,v))-kron(S(w,w)',Ip);
  [L,H,P,Q] = lu_complpiv(K);
  e = min(abs(diag(H)));
  sigp = 1:p*q;
  for k = 1:p*q-1;
    sigp([k,P(k)]) = sigp([P(k),k]);
  end 
  r = e*r(sigp); 
  x = (H\(L\r));
  sigq = 1:p*q;
  for k = 1:p*q-1;
    sigq([k,Q(k)]) = sigq([Q(k),k]);
  end
  x(sigq) = x;
  X = [];
  for j=1:q
    X = [X,x((j-1)*p+1:j*p)];
  end
  [Q,R]      = qr([-X;e*Iq]);
  S(:,[v,w]) = S(:,[v,w])*Q;
  S([v,w],:) = Q'*S([v,w],:); 
  U(:,[v,w]) = U(:,[v,w])*Q;

  % -----------------------------------------------%

    function [L,U,P,Q] = lu_complpiv(A);
    P = []; Q = []; n = size(A,1);
    for k=1:n-1;
      [a,r] = max(abs(A(k:n,k:n)));
      [dummy,c] = max(abs(a));
      cl  = c+k-1;
      rw  = r(c)+k-1;
      A([k,rw],:) = A([rw,k],:);
      A(:,[k,cl]) = A(:,[cl,k]);
      P(k) = rw; Q(k) = cl;
      if A(k,k) ~= 0;
        rs = k+1:n;
        A(rs,k)  = A(rs,k)/A(k,k);
        A(rs,rs) = A(rs,rs)-A(rs,k)*A(k,rs);
      end
    end
    U = tril(A')'; L = tril(A,-1) + eye(n);
   
  

  
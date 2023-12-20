function out = zselect( x )

out=zeros(1,1,int32)
if (abs(x)<2)
    out(1)=1
else
    out(1)=2
end



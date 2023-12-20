   

for fftt= 1:.5*length(OMEGA)+1;
DELTA(:,fftt+1 ) = ((ST)\(Global_load));
    DELTA(:,length(OMEGA)+1-fftt) = conj(DELTA(:,fftt+1));
end
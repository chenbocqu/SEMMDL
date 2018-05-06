function [ out ] = plot_dict( mdl )

P = mdl.P;
B = mdl.B;

%%
ImB         = displayPatches(B);      % 
ImPB         = displayPatches(P'*B);      % 

close ;
figure;
%%
subplot(121); imagesc(ImB);  colormap('gray');xlabel('µÍÎ¬×ÖµäB');
subplot(122); imagesc(ImPB); colormap('gray');xlabel('ÖØ¹¹¸ßÎ¬×ÖµäP^TB');

end


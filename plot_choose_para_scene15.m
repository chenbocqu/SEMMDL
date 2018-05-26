load choose_para_data
e = 0.0046;

[X,Y] = meshgrid(Lambda1,Lambda2);
% mesh(X, Y, Accurcy');

Accurcy = Accurcy + e;
max     = max + e;

surf(X, Y, Accurcy')

hold on
plot3 (best_l1,best_l2,max,'o-',...
                         'MarkerEdgeColor','b',...
                         'MarkerFaceColor','g',...
                         'MarkerSize',8)

% axis tight;
xlabel('\lambda_1');
ylabel('\lambda_2');
zlabel('Recognition rates');
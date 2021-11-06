clear
close all
clc
d_pre = data_loader_class('evaluation_data_pre.mat');
d_post = data_loader_class('evaluation_data_post.mat');
d_Sohrabi = data_loader_class('evaluation_data_Sohrabi_cpu.mat');

titles = {'After stage 1 of training', 'After stage 2 of training', 'Sohrabi'};
markers = {'bs-', 'ro-', 'black*-'};
Ylabels = {'$\overline{C}_{\phi}$', '$\overline{C}_{\phi}$', '$\overline{C}_{\phi}$'};
array_of_data_objects = [d_pre, d_post, d_Sohrabi];
n_plots = length(array_of_data_objects) + 2;
for i = 1:n_plots-2
    figure(1)
    subplot(1,n_plots,i)
    boxchart(array_of_data_objects(i).C_samples_x_OFDM_index,'MarkerStyle','.')
    hold on
    plot(mean(array_of_data_objects(i).C_samples_x_OFDM_index,1) , 'r.-' , 'linewidth',1)
    plot(1:array_of_data_objects(i).s(2), mean(mean(array_of_data_objects(i).C_samples_x_OFDM_index))+0.*[1:array_of_data_objects(i).s(2)],'b','linewidth',1)
    %         plot(1:array_of_data_objects(i).s(2)+1, 10.9 +0.*[1:array_of_data_objects(i).s(2)+1],'Cyan','linewidth',1)
    xticks(categorical(1:24:array_of_data_objects(i).s(2)))
%     xticks([1 25 50])
    xlabel('OFDM symbol index')
    ylabel(char(Ylabels(i)),'Interpreter','latex','fontsize',16)
    title(titles(i))
    ylim([0 10])
    if (i==3)
        legend( 'stats of Capacity_{symbol}' , 'mean of Capacity_{symbol}', 'Capacity_{Frame}', 'Capacity_{PHN-free}', 'Orientation', 'horizontal')
    end
    
    subplot(1,n_plots,4)
    plot(mean(array_of_data_objects(i).C_samples_x_OFDM_index,1),char(markers(i)) , 'linewidth',1)
    ylabel('$\overline{C}_{\phi}$','Interpreter','latex','fontsize',16)
    xlabel('OFDM symbol index')
    title('Capacity of OFDM symbols')
    legend(titles)
    ylim([0 10])
    hold on
    xticks([1 25 50])
    
        %% R_X
%         subplot(3,n_plots ,i+n_plots)
%         boxchart(squeeze(mean(array_of_data_objects(i).RX_forall_k_forall_OFDMs_forall_samples,3)) ,'MarkerStyle','.')
%         hold on
%         xticks(categorical(1:array_of_data_objects(i).s(2)))
%         xlabel('OFDM symbol index \times 10')
%         ylabel(' $\frac{1}{K}\sum_k \mathbf{R}_{\mathbf{X}}[k]$ ','Interpreter','latex')
%     %     ylim([ 0 1])
%         %% R_Q
%         subplot(3,n_plots ,i+2*n_plots )
%         boxchart(squeeze(mean(array_of_data_objects(i).RQ_forall_k_forall_OFDMs_forall_samples,3)) ,'MarkerStyle','.')
%         hold on
%         xticks(categorical(1:array_of_data_objects(i).s(2)))
%         xlabel('OFDM symbol index \times 10')
%         ylabel(' $\frac{1}{K}\sum_k \mathbf{R}_{\mathbf{Q}}[k]$ ','Interpreter','latex')
    %     ylim([ 0 .02])
end


%% ploting losses
%
load('loss_data.mat')
len_tr = length(tr_loss);
len_ts = length(ts_loss);
figure(1)
subplot(1,n_plots,5)
plot(1:len_tr,tr_loss,char(markers(1)), 'linewidth', 1)
hold on
plot(2:len_ts+1,ts_loss,char(markers(2)), 'linewidth', 1)
plot(2:len_ts+1,metric,char(markers(3)), 'linewidth', 1)
legend('train loss', 'test loss', 'real capacity', 'Sohrabi method in absence of phase noise')
title('loss vs epoch')
xlabel('epoch')
ylabel('loss')
grid on

%% parallel coordinate data
load('training_metadata_records.mat')
[num_rows, num_cols] = size(training_metadata)

recorded_items = {
    'bench?',
    'train dataset sizePre',
    'train Dataset Size Post',
    'width',
    'Batch Size',
    'L rate',
    'gradient norm clipper',
    'LR decay rate',
    'LR patience',
    'min lr',
    'dropout rate',
    'conv kernels',
    'conv filters',
    'conv strides',
    'n params',
    'n layers',
    'epochs pre',
    'epochs post',
    'loss tr pre',
    'loss ts pre',
    'capacity pre',
    'loss tr post',
    'loss ts post',
    'capacity post'};


varTypes = {
    'string',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double',
    'double'};

varNames = recorded_items;
sz = [num_rows, num_cols];
T = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
for i = 1 : num_rows
    T(i,2:num_cols+1) = num2cell(training_metadata(i,:));
    T(i,1) = {'NO'};
end
T(1,1) = {'YES'};
figure(3)
p=parallelplot(T, 'GroupVariable','capacity post', 'Jitter', 0.0, 'LineAlpha', .3,'LineStyle', '-', 'LineWidth', 3, 'MarkerStyle', '+' )





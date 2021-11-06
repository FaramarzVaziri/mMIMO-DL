classdef data_loader_class
    properties
        C_samples_x_OFDM_index, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples, s, L
    end
    methods
        function obj = data_loader_class(val)
            y = load(val) ;
%             disp(y)
            obj.C_samples_x_OFDM_index = y.C_samples_x_OFDM_index;
            obj.s = size(y.C_samples_x_OFDM_index);
            obj.L = y.L;
            obj.RX_forall_k_forall_OFDMs_forall_samples = y.RX_forall_k_forall_OFDMs_forall_samples;
            obj.RQ_forall_k_forall_OFDMs_forall_samples = y.RQ_forall_k_forall_OFDMs_forall_samples;
        end
    end
end

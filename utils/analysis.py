import torch
import numpy as np

def process_outputs(args, model, data, output):
    Y_train, Y_test, Y_train_om, train_ts, test_ts, train_om_ts, train_oms, test_oms = data.values()
    train_responses, train_responses_om, test_responses, test_om_indcs, train_om_indcs = output.values()

    # change responses
    change_responses = get_change_responses(args, train_responses, test_responses, train_ts, test_ts, test_oms, model.perception_only)

    # omission responses
    omission_responses = get_omission_responses(args, train_responses_om, test_responses, train_oms, test_oms, train_om_indcs, test_om_indcs)

    return change_responses, omission_responses


def get_change_responses(args, train_responses, test_responses,
                         train_ts, test_ts,
                         test_oms, perception_only=True):

    z_train, z_test = train_responses['z'], test_responses['z']
    sigmap_train, sigmap_test = train_responses['sigma_p'], test_responses['sigma_p']
    sigmaq_train, sigmaq_test = train_responses['sigma_q'], test_responses['sigma_q']
    mup_train, mup_test = train_responses['mu_p'], test_responses['mu_p']
    theta_train, theta_test = train_responses['theta'], test_responses['theta']
    h_train, h_test = train_responses['h'], test_responses['h']
    terror_train, terror_test = (train_responses['mu_p'] - train_responses['z'])**2, (test_responses['mu_p'] - test_responses['z'])**2

    if not perception_only:
        value_train, value_test = train_responses['lick_value'], test_responses['lick_value']
    
    test_no_om_indcs = np.where(np.array(test_oms) <= 0)[0]
    num_test_no_omission = len(test_no_om_indcs)

    half_blank = args.blank_ts // 2
    trial_dur = 2 * half_blank + args.blank_ts + 2 * args.img_ts + 1

    z_train_change = torch.zeros(z_train.shape[0], trial_dur, z_train.shape[-1])
    z_test_change = torch.zeros(num_test_no_omission, trial_dur, z_test.shape[-1])
        
    sigmap_train_change = torch.zeros(sigmap_train.shape[0], trial_dur, sigmap_train.shape[-1])
    sigmap_test_change = torch.zeros(num_test_no_omission, trial_dur, sigmap_test.shape[-1])

    sigmaq_train_change = torch.zeros(sigmaq_train.shape[0], trial_dur, sigmaq_train.shape[-1])
    sigmaq_test_change = torch.zeros(num_test_no_omission, trial_dur, sigmaq_test.shape[-1])

    theta_train_change = torch.zeros(theta_train.shape[0], trial_dur, theta_train.shape[-1])
    theta_test_change = torch.zeros(num_test_no_omission, trial_dur, theta_test.shape[-1])

    h_train_change = torch.zeros(h_train.shape[0], trial_dur, h_train.shape[-1])
    h_test_change = torch.zeros(num_test_no_omission, trial_dur, h_test.shape[-1])

    terror_train_change = torch.zeros(terror_train.shape[0], trial_dur, terror_train.shape[-1])
    terror_test_change = torch.zeros(num_test_no_omission, trial_dur, terror_test.shape[-1])

    mup_train_change = torch.zeros(mup_train.shape[0], trial_dur, mup_train.shape[-1])
    mup_test_change = torch.zeros(num_test_no_omission, trial_dur, mup_test.shape[-1])

    if not perception_only:
        value_train_change = torch.zeros(value_train.shape[0], trial_dur)
        value_test_change = torch.zeros(num_test_no_omission, trial_dur)

    for s in range(z_train_change.shape[0]):
        start = train_ts[s]['before'][0][-1] - half_blank
        end = train_ts[s]['after'][1][0] + half_blank + 1
        z_train_change[s] = z_train[s, start:end, :]
        sigmaq_train_change[s] = sigmaq_train[s, start:end, :]
        sigmap_train_change[s] = sigmap_train[s, start:end, :]
        mup_train_change[s] = mup_train[s, start:end, :]
        theta_train_change[s] = theta_train[s, start:end, :]
        h_train_change[s] = h_train[s, start:end, :]
        terror_train_change[s] = terror_train[s, start:end]

        if not perception_only:
            value_train_change[s] = value_train[s, start:end]

    for si in range(num_test_no_omission):
        s = test_no_om_indcs[si]
        start = test_ts[s]['before'][0][-1] - half_blank
        end = test_ts[s]['after'][1][0] + half_blank + 1
        z_test_change[si] = z_test[s, start:end, :]
        sigmaq_test_change[si] = sigmaq_test[s, start:end, :]
        sigmap_test_change[si] = sigmap_test[s, start:end, :]
        mup_test_change[si] = mup_test[s, start:end, :]
        theta_test_change[si] = theta_test[s, start:end, :]
        h_test_change[si] = h_test[s, start:end, :]
        terror_test_change[si] = terror_test[s, start:end, :]

        if not perception_only:
            value_test_change[si] = value_test[s, start:end]

    # PV activity is the inverse of sigma_q but we assume zero still gets mapped to zero
    pv_train_change = torch.where(sigmaq_train_change != 0, 1. / sigmaq_train_change, torch.zeros_like(sigmaq_train_change))
    pv_test_change = torch.where(sigmaq_test_change != 0, 1. / sigmaq_test_change, torch.zeros_like(sigmaq_test_change))
    
    # fraction of active exc neurons
    z_active_train = torch.count_nonzero(z_train_change, -1).unsqueeze(-1) / args.latent_dim
    z_active_test = torch.count_nonzero(z_test_change, -1).unsqueeze(-1) / args.latent_dim

    change_responses = {
        "familiar": {
            "exc": z_train_change,
            "frac_active": z_active_train,
            "pv": pv_train_change,
            "vip": sigmap_train_change,
            "sst_theta": theta_train_change,
            "sst_mup": mup_train_change,
            "h": h_train_change,
            "temp_pred_error": terror_train_change,
        },

        "novel": {
            "exc": z_test_change,
            "frac_active": z_active_test,
            "pv": pv_test_change,
            "vip": sigmap_test_change,
            "sst_theta": theta_test_change,
            "sst_mup": mup_test_change,
            "h": h_test_change,
            "temp_pred_error": terror_test_change,
        }
    }

    if not perception_only:
        change_responses["familiar"].update({
            "value": value_train_change.unsqueeze(-1)
        })

        change_responses["novel"].update({
            "value": value_test_change.unsqueeze(-1)
        })
    

    return change_responses

def get_omission_responses(args, train_responses_om, test_responses, train_oms, test_oms, train_om_indcs, test_om_indcs):

    sigmap_train_om, h_train_om, z_train_om, mup_train_om, theta_train_om = train_responses_om['sigma_p'], train_responses_om['h'], train_responses_om['z'], train_responses_om['mu_p'], train_responses_om['theta']
    sigmap_test, h_test, z_test, mup_test, theta_test = test_responses['sigma_p'], test_responses['h'], test_responses['z'], test_responses['mu_p'], test_responses['theta']

    om_trial_dur = 2 * args.blank_ts + 2 * (args.blank_ts // 2) + 3 * args.img_ts

    sigmap_om_train = torch.zeros(len(train_om_indcs), om_trial_dur, sigmap_train_om.shape[-1])
    h_om_train = torch.zeros(len(train_om_indcs), om_trial_dur, h_train_om.shape[-1])
    z_om_train = torch.zeros(len(train_om_indcs), om_trial_dur, z_train_om.shape[-1])
    mup_om_train = torch.zeros(len(train_om_indcs), om_trial_dur, mup_train_om.shape[-1])
    theta_om_train = torch.zeros(len(train_om_indcs), om_trial_dur, theta_train_om.shape[-1])

    sigmap_om_test = torch.zeros(len(test_om_indcs), om_trial_dur, sigmap_test.shape[-1])
    h_om_test = torch.zeros(len(test_om_indcs), om_trial_dur, h_test.shape[-1])
    z_om_test = torch.zeros(len(test_om_indcs), om_trial_dur, z_test.shape[-1])
    mup_om_test = torch.zeros(len(test_om_indcs), om_trial_dur, mup_test.shape[-1])
    theta_om_test = torch.zeros(len(test_om_indcs), om_trial_dur, theta_test.shape[-1])

    for si in range(len(train_om_indcs)):
        
        s = train_om_indcs[si]
        om_ind = train_oms[s]
        start = om_ind - (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = om_ind + (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = min(end, sigmap_train_om.shape[1])
        dur = end - start
        
        h_om_train[si, :dur] = h_train_om[s, start:end]
        sigmap_om_train[si, :dur] = sigmap_train_om[s, start:end]
        z_om_train[si, :dur] = z_train_om[s, start:end]
        mup_om_train[si, :dur] = mup_train_om[s, start:end]
        theta_om_train[si, :dur] = theta_train_om[s, start:end]
        

    for si in range(len(test_om_indcs)):
        
        s = test_om_indcs[si]
        om_ind = test_oms[s]
        start = om_ind - (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = om_ind + (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = min(end, sigmap_test.shape[1])
        dur = end - start
        
        h_om_test[si, :dur] = h_test[s, start:end]
        sigmap_om_test[si, :dur] = sigmap_test[s, start:end]
        z_om_test[si, :dur] = z_test[s, start:end]
        mup_om_test[si, :dur] = mup_test[s, start:end]
        theta_om_test[si, :dur] = theta_test[s, start:end]
    
    omission_responses = {
        "familiar": {
            "exc": z_om_train,
            "vip": sigmap_om_train,
            "sst_theta": theta_om_train,
            "sst_mup": mup_om_train,
            "h": h_om_train
        },

        "novel": {
            "exc": z_om_test,
            "vip": sigmap_om_test,
            "sst_theta": theta_om_test,
            "sst_mup": mup_om_test,
            "h": h_om_test
        }
    }

    return omission_responses


    
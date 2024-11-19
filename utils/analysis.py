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
    
    num_train = train_responses['z'].shape[0]
    
    test_no_om_indcs = np.where(np.array(test_oms) <= 0)[0]
    num_test_no_omission = len(test_no_om_indcs)

    half_blank = args.blank_ts // 2
    trial_dur = 2 * half_blank + args.blank_ts + 2 * args.img_ts + 1
    
    familiar_resp = {k: torch.zeros(num_train, trial_dur, v.shape[-1]) for (k, v) in train_responses.items()}
    novel_resp = {k: torch.zeros(num_test_no_omission, trial_dur, v.shape[-1]) for (k, v) in test_responses.items()}

    for s in range(num_train):
        start = train_ts[s]['before'][0][-1] - half_blank
        end = train_ts[s]['after'][1][0] + half_blank + 1
        for k in familiar_resp.keys():
            familiar_resp[k][s] = train_responses[k][s, start:end]

    for si in range(num_test_no_omission):
        s = test_no_om_indcs[si]
        start = test_ts[s]['before'][0][-1] - half_blank
        end = test_ts[s]['after'][1][0] + half_blank + 1
        for k in novel_resp.keys():
            novel_resp[k][si] = test_responses[k][s, start:end]

    # PV activity is the inverse of sigma_q but we assume zero still gets mapped to zero
    familiar_resp['sigma_q'] = torch.where(familiar_resp['sigma_q'] != 0, 1. / familiar_resp['sigma_q'], torch.zeros_like(familiar_resp['sigma_q']))
    novel_resp['sigma_q'] = torch.where(novel_resp['sigma_q'] != 0, 1. / novel_resp['sigma_q'], torch.zeros_like(novel_resp['sigma_q']))
    
    # fraction of active exc neurons
    familiar_resp['frac_active'] = torch.count_nonzero(familiar_resp['z'], -1).unsqueeze(-1) / args.latent_dim
    novel_resp['frac_active'] = torch.count_nonzero(novel_resp['z'], -1).unsqueeze(-1) / args.latent_dim

    change_responses = {"familiar": familiar_resp, "novel": novel_resp}    

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


    
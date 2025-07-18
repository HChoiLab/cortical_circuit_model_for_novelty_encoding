import torch
import numpy as np
import scipy.stats as stats

def process_outputs(args, model, data, output, abridged=False):
    Y_train, Y_test, Y_train_om, train_ts, test_ts, train_om_ts, train_oms, test_oms = data.values()
    train_responses, train_responses_om, test_responses, test_om_indcs, train_om_indcs = output.values()

    # change responses
    change_responses = get_change_responses(args, train_responses, test_responses, train_ts, test_ts, test_oms, abridged=abridged)

    # omission responses
    omission_responses = get_omission_responses(args, train_responses_om, test_responses, train_oms, test_oms, train_om_indcs, test_om_indcs, abridged=abridged)

    return change_responses, omission_responses


def get_change_responses(args, train_responses, test_responses,
                         train_ts, test_ts,
                         test_oms,
                         abridged=False):
    
    if args.adaptation_model:
        abridged_pops = ['E', 'VIP', 'SST']
    else:
        abridged_pops = ['z', 'sigma_q', 'sigma_p', 'theta', 'mu_p', 'temp_error']
    abridged_fn = lambda s: (s in abridged_pops) if abridged else True
    
    num_train = train_responses[abridged_pops[0]].shape[0]
    
    test_no_om_indcs = np.where(np.array(test_oms) <= 0)[0]
    num_test_no_omission = len(test_no_om_indcs)

    half_blank = args.blank_ts // 2
    trial_dur = 2 * half_blank + args.blank_ts + 2 * args.img_ts + 1
    
    familiar_resp = {k: torch.zeros(num_train, trial_dur, v.shape[-1]) for (k, v) in train_responses.items() if abridged_fn(k)}
    novel_resp = {k: torch.zeros(num_test_no_omission, trial_dur, v.shape[-1]) for (k, v) in test_responses.items() if abridged_fn(k)}

    for s in range(num_train):
        pre_on = train_ts[s]['before'][0][-1]
        change_off = train_ts[s]['after'][1][0]
        
        # trial start and end times
        start = pre_on - half_blank
        end = change_off + half_blank + 1
        for k in familiar_resp.keys():
            familiar_resp[k][s] = train_responses[k][s, start:end]

    for si in range(num_test_no_omission):
        s = test_no_om_indcs[si]
        
        pre_on = test_ts[s]['before'][0][-1]
        change_off = test_ts[s]['after'][1][0]
        
        # trial start and end times
        start = pre_on - half_blank
        end = change_off + half_blank + 1
        for k in novel_resp.keys():
            novel_resp[k][si] = test_responses[k][s, start:end]

    if not args.adaptation_model:
        # PV activity is the inverse of sigma_q but we assume zero still gets mapped to zero
        familiar_resp['sigma_q'] = torch.where(familiar_resp['sigma_q'] != 0, 1. / familiar_resp['sigma_q'], torch.zeros_like(familiar_resp['sigma_q']))
        novel_resp['sigma_q'] = torch.where(novel_resp['sigma_q'] != 0, 1. / novel_resp['sigma_q'], torch.zeros_like(novel_resp['sigma_q']))
        
        # fraction of active exc neurons
        familiar_resp['frac_active'] = torch.count_nonzero(familiar_resp['z'], -1).unsqueeze(-1) / args.latent_dim
        novel_resp['frac_active'] = torch.count_nonzero(novel_resp['z'], -1).unsqueeze(-1) / args.latent_dim

    change_responses = {"familiar": familiar_resp, "novel": novel_resp}
    
    familiar_means = {k: torch.zeros(v.shape[0], v.shape[-1], 2) for (k, v) in familiar_resp.items()}
    novel_means = {k: torch.zeros(v.shape[0], v.shape[-1], 2) for (k, v) in novel_resp.items()}

    for k in change_responses['familiar'].keys():
        pre_start = half_blank
        change_start = half_blank + args.img_ts + args.blank_ts
        familiar_means[k][:, :, 0] = change_responses["familiar"][k][:, pre_start:pre_start+4].mean(1)
        familiar_means[k][:, :, 1] = change_responses["familiar"][k][:, change_start:change_start+4].mean(1)
        novel_means[k][:, :, 0] = change_responses["novel"][k][:, pre_start:pre_start+4].mean(1)
        novel_means[k][:, :, 1] = change_responses["novel"][k][:, change_start:change_start+4].mean(1)
    
    change_responses["familiar_means"] = familiar_means
    change_responses["novel_means"] = novel_means

    return change_responses

def get_omission_responses(args, train_responses_om, test_responses, train_oms, test_oms, train_om_indcs, test_om_indcs, abridged=True):

    if args.adaptation_model:
        abridged_pops = ['E', 'VIP', 'SST']
    else:
        abridged_pops = ['z', 'sigma_q', 'sigma_p', 'theta', 'mu_p', 'temp_error']
    abridged_fn = lambda s: (s in abridged_pops) if abridged else True

    num_train = len(train_om_indcs)
    num_test = len(test_om_indcs)

    om_trial_dur = 2 * args.blank_ts + 2 * (args.blank_ts // 2) + 3 * args.img_ts
    seq_len_train, seq_len_test = train_responses_om[abridged_pops[0]].shape[1], test_responses[abridged_pops[0]].shape[1]

    familiar_resp = {k: torch.zeros(num_train, om_trial_dur, v.shape[-1]) for (k, v) in train_responses_om.items() if abridged_fn(k)}
    novel_resp = {k: torch.zeros(num_test, om_trial_dur, v.shape[-1]) for (k, v) in test_responses.items() if abridged_fn(k)}

    for si in range(num_train):
        
        s = train_om_indcs[si]
        om_ind = train_oms[s]
        start = om_ind - (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = om_ind + (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = min(end, seq_len_train)
        dur = end - start

        for k in familiar_resp.keys():
            familiar_resp[k][si, :dur] = train_responses_om[k][s, start:end]
    
    for si in range(num_test):

        s = test_om_indcs[si]
        om_ind = test_oms[s]
        start = om_ind - (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = om_ind + (args.blank_ts + args.blank_ts // 2 + args.img_ts)
        end = min(end, seq_len_test)
        dur = end - start

        for k in novel_resp.keys():
            novel_resp[k][si, :dur] = test_responses[k][s, start:end]
    
    omission_responses = {"familiar": familiar_resp, "novel": novel_resp}

    return omission_responses


def compute_dprime(ts, actions, response_window):
    
    num_catch, fas = 0, 0
    num_go, hits = len(ts), 0

    for si in range(len(ts)):

        s = ts[si]

        num_catch += len(s['before'][0]) + len(s['after'][0]) - 1

        # find false alarms during catch trials
        for tr in s['before'][0] + s['after'][0][1:]:
            tr_actions = actions[si][tr+response_window[0]:tr+response_window[1]]
            if torch.any(tr_actions).item():
                fas += 1
        
        # find hits during go trial
        go_ts = s['after'][0][0]
        go_actions = actions[si][go_ts+response_window[0]:go_ts+response_window[1]]
        if torch.any(go_actions).item():
            hits += 1
    
    hit_rate = hits / num_go
    fa_rate = fas / num_catch

    hit_rate = min(max(hit_rate, 1 / (2 * num_go)), 1 - 1 / (2 * num_go))
    fa_rate = min(max(fa_rate, 1 / (2 * num_catch)), 1 - 1 / (2 * num_catch))

    dprime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)

    return dprime

def compute_population_stats(responses, alpha=0.05):

    # Sample size, mean, and standard error of the mean
    n = len(responses)
    mean_val = np.mean(responses)
    std_val = np.std(responses, ddof=1)
    sem = std_val / np.sqrt(n)
    
    # Two-tailed t-value for CI = 100 * (1 - alpha) %
    # degrees of freedom = n-1
    t_val = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Confidence interval
    error_margin = t_val * sem
    return mean_val, error_margin
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scienceplots as scp
import torch


def plot_trial_responses(args, ax, familiar_responses, novel_responses, trial_mode='change', labels=None, clrs=None, sem=True, normalize=True):
    
    if labels is None:
        labels = ["Familiar", "Novel"]
    if clrs is None:
        clrs = ["darkorange", "darkblue"]
    
    familiar_mean = familiar_responses.mean([0, -1]).detach()
    novel_mean = novel_responses.mean([0, -1]).detach()
    
    # define the event onset
    half_blank = args.blank_ts // 2
    event_onset = args.blank_ts // 2 + args.img_ts + args.blank_ts
    
    # normalize averages
    if normalize:
        min_all = torch.min(torch.cat([familiar_mean[:event_onset], novel_mean[:event_onset]]))
        max_all = torch.max(torch.cat([familiar_mean[:event_onset], novel_mean[:event_onset]]))
        familiar_mean = (familiar_mean - min_all) / (max_all - min_all)
        novel_mean = (novel_mean - min_all) / (max_all - min_all)
    
    # calculate std     
    familiar_std = familiar_responses.mean(-1).std(0).detach() / np.sqrt(familiar_responses.shape[0])
    novel_std = novel_responses.mean(-1).std(0).detach() / np.sqrt(novel_responses.shape[0])
    
    # plot image presentations
    if trial_mode == 'change':
        ax.axvspan(half_blank, half_blank + args.img_ts, color="r", alpha=0.05)
        ax.axvspan(half_blank + args.blank_ts + args.img_ts, half_blank + args.blank_ts + 2 * args.img_ts, color="b", alpha=0.05)
    
    elif trial_mode == 'omission':
        # first image
        ax.axvspan(half_blank, half_blank + args.img_ts, color='magenta', alpha=0.05)

        # omitted image
        ax.axvline(args.blank_ts + half_blank + args.img_ts, linestyle="--", color='magenta', linewidth=2.5)
        ax.axvline(args.blank_ts + half_blank + 2 * args.img_ts, linestyle="--", color='magenta', linewidth=2.5)

        # last image
        ax.axvspan(2 * args.blank_ts + half_blank + 2 * args.img_ts,
                   2 * args.blank_ts + half_blank + 3 * args.img_ts, color='magenta', alpha=0.05)
    else:
        raise
        
    ax.plot(familiar_mean.numpy(), label=labels[0], color=clrs[0], linewidth=3.0)
    ax.plot(novel_mean.numpy(), label=labels[1], color=clrs[1], linewidth=3.0)
    if sem:
        ax.fill_between(np.arange(familiar_responses.shape[1]),
                        familiar_mean - familiar_std,
                        familiar_mean + familiar_std,
                        color=clrs[0], alpha=0.25)
        ax.fill_between(np.arange(novel_responses.shape[1]),
                        novel_mean - novel_std,
                        novel_mean + novel_std,
                        color=clrs[1], alpha=0.25)

def plot_change_responses(args, ax, responses, label, clr, sem=True):
    response_mean = responses.mean([0, -1]).detach()
    response_std = responses.mean(-1).std(0).detach() / np.sqrt(responses.shape[0])
    
    half_blank = args.blank_ts // 2
    
    ax.axvspan(half_blank, half_blank + args.img_ts, color="r", alpha=0.05)
    ax.axvspan(half_blank + args.blank_ts + args.img_ts, half_blank + args.blank_ts + 2 * args.img_ts, color="b", alpha=0.05)
    
    ax.plot(response_mean.numpy(), label=label, color=clr, linewidth=3.0)
    if sem:
        ax.fill_between(np.arange(responses.shape[1]),
                        response_mean - response_std,
                        response_mean + response_std,
                        color=clr, alpha=0.25)
        
def plot_omission_responses(args, ax, responses, label, image_clr, trace_clr, sem=True):
    response_mean = responses.mean([0, -1]).detach()
    response_std = responses.mean(-1).std(0).detach() / np.sqrt(responses.shape[0])
    
    # first image
    ax.axvspan(args.blank_ts // 2, args.blank_ts // 2 + args.img_ts, color=image_clr, alpha=0.05)
    
    # omitted image
    ax.axvline(args.blank_ts + args.blank_ts // 2 + args.img_ts, linestyle="--", color=image_clr, linewidth=2.5)
    ax.axvline(args.blank_ts + args.blank_ts // 2 + 2 * args.img_ts, linestyle="--", color=image_clr, linewidth=2.5)
    
    # last image
    ax.axvspan(2 * args.blank_ts + args.blank_ts // 2 + 2 * args.img_ts,
               2 * args.blank_ts + args.blank_ts // 2 + 3 * args.img_ts, color=image_clr, alpha=0.05)
    
    ax.plot(response_mean.numpy(), label=label, color=trace_clr, linewidth=3.0)
    if sem:
        ax.fill_between(np.arange(responses.shape[1]),
                        response_mean - response_std,
                        response_mean + response_std,
                        color=trace_clr, alpha=0.25)
        
        
def raincloud_plot(ax, familiar_responses, novel_responses):

    # Create a list of colors for the boxplots based on the number of features you have
    boxplots_colors = ['darkorange', 'darkblue']
    median_colors = ['orangered', 'navy']

    # Boxplot data
    data = [familiar_responses, novel_responses]
    bp = ax.boxplot(data, patch_artist = True, vert = True, showmeans=True, showfliers=False,
                   meanprops={'markersize': 10, 'markerfacecolor': 'darkgreen'})

    # Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
        patch.set_linewidth(2.0)
        patch.set_linestyle('solid')
    
    for patch, color in zip(bp['medians'], median_colors):
        patch.set_color(color)
        patch.set_linewidth(2.5)

    # Create a list of colors for the violin plots based on the number of features you have
    violin_colors = ['orange', 'cornflowerblue']

    # Violinplot data
    vp = ax.violinplot(data, points=500, 
                   showmeans=False, showextrema=False, showmedians=False, vert=True)
    
    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
        # Change to the desired color
        b.set_color(violin_colors[idx])
        b.set_alpha(0.3)
    

    # Create a list of colors for the scatter plots based on the number of features you have
    scatter_colors = ['darkorange', 'darkblue']

    # Scatterplot data
    for idx, features in enumerate(data):
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        plt.scatter(y, features, s=10, c=scatter_colors[idx])

    plt.xticks(np.arange(1,3,1), ['Familiar', 'Novel'])  # Set text labels.
    plt.ylabel('Average response')


def plot_sequence_response(responses, timestamps, seq_idx=0, pop_avg=False, perception_only=True):

    z, vip, sst_theta, sst_mup = responses['z'][seq_idx], responses['sigma_p'][seq_idx], responses['theta'][seq_idx], responses['mu_p'][seq_idx]

    if not perception_only:
        licks = responses['action']

    if not pop_avg:
        z = z[..., randint(z.shape[-1])]
        vip = vip[..., randint(vip.shape[-1])]
        sst_theta = sst_theta[..., randint(sst_theta.shape[-1])]
        sst_mup = sst_mup[..., randint(sst_mup.shape[-1])]
    
    else:
        z, vip, sst_theta, sst_mup = z.mean(-1), vip.mean(-1), sst_theta.mean(-1), sst_mup.mean(-1)
        
    _ = plt.figure(figsize=(15, 10))

    with plt.style.context(['nature', 'notebook']):

        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)

        for bf_on, bf_off in zip(*timestamps[seq_idx]['before']):
            ax1.axvspan(bf_on, bf_off, color="r", alpha=0.09)
            ax2.axvspan(bf_on, bf_off, color="r", alpha=0.09)
            ax3.axvspan(bf_on, bf_off, color="r", alpha=0.09)
            

        for af_on, af_off in zip(*timestamps[seq_idx]['after']):
            ax1.axvspan(af_on, af_off, color="b", alpha=0.09)
            ax2.axvspan(af_on, af_off, color="b", alpha=0.09)
            ax3.axvspan(af_on, af_off, color="b", alpha=0.09)

        ax1.plot(z.cpu().detach().numpy(), c='firebrick', label="Excitatory", linewidth=3.5)
        ax2.plot(vip.cpu().detach().numpy(), c='darkgreen', label="VIP", linewidth=3.5)
        ax3.plot(sst_theta.cpu().detach().numpy(), c='darkmagenta', label="SST (Theta)", linewidth=3.5)
        
        ax3.set_xlabel("Timestep")
        
        #ax1.set_title("Excitatory")
        #ax2.set_title("VIP")
        #ax3.set_title("SST (Theta)")
        
        ax1.tick_params('x', which='both', top=False, labelbottom=False)
        ax2.tick_params('x', which='both', top=False, labelbottom=False)
        ax3.tick_params('x', which='both', top=False)
        ax1.tick_params('y', which='both', right=False)
        ax2.tick_params('y', which='both', right=False)
        ax3.tick_params('y', which='both', right=False)
        
        ax1.locator_params('x', nbins=4)
        ax1.locator_params('y', nbins=2)
        ax2.locator_params('x', nbins=4)
        ax2.locator_params('y', nbins=2)
        ax3.locator_params('x', nbins=4)
        ax3.locator_params('y', nbins=2)
        
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # plot licks on the first axis
        if not perception_only:
            
            licks_inds = licks[seq_idx].nonzero()

            ymin, ymax = ax1.get_ylim()
            vertical_center = (ymin + ymax) / 2
            ax1.plot(licks_inds.cpu().numpy(), [vertical_center]*len(licks_inds), 'bo', markersize=12, alpha=0.4)
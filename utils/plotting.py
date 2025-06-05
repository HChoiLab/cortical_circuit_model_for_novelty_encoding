import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import scienceplots as scp
import torch
from utils.schedules import ramp_schedule, decreasing_ramp_schedule, stepLR_schedule
from task import get_reward_sequence
from utils.analysis import compute_population_stats

# GLOBAL COLOR PARAMETERS
PRE_CLR = sns.color_palette('pastel')[-3]
CHANGE_CLR = sns.color_palette('husl', 9)[-2]

FAM_CLR = 'darkorange'
NOV_CLR = 'darkblue'

VIP_CLR = 'royalblue'
SST_CLR = 'forestgreen'
EXC_CLR = 'firebrick'


def plot_trial_responses(args, ax, familiar_responses, novel_responses,
                         trial_mode='change', labels=None, clrs=None,
                         sem=True, normalize=True, lw=3.0):
    
    if labels is None:
        labels = ["Familiar", "Novel"]
    if clrs is None:
        clrs = [FAM_CLR, NOV_CLR]
    
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

    # use time relative to event onset
    half_blank -= event_onset
    
    # plot image presentations
    if trial_mode == 'change':
        ax.axvspan(half_blank, half_blank + args.img_ts, color=PRE_CLR, alpha=0.25, edgecolor="none", linewidth=0, zorder=1)
        ax.axvspan(half_blank + args.blank_ts + args.img_ts, half_blank + args.blank_ts + 2 * args.img_ts, color=CHANGE_CLR, alpha=0.2,
                   edgecolor="none", linewidth=0, zorder=1)
    
    elif trial_mode == 'omission':
        # first image
        ax.axvspan(half_blank, half_blank + args.img_ts, color=PRE_CLR, alpha=0.25)

        # omitted image
        ax.axvline(args.blank_ts + half_blank + args.img_ts, linestyle="--", color='steelblue', linewidth=2.5)
        ax.axvline(args.blank_ts + half_blank + 2 * args.img_ts, linestyle="--", color='steelblue', linewidth=2.5)

        # last image
        ax.axvspan(2 * args.blank_ts + half_blank + 2 * args.img_ts,
                   2 * args.blank_ts + half_blank + 3 * args.img_ts, color=PRE_CLR, alpha=0.25)
    else:
        raise
        
    x_range = np.arange(len(familiar_mean)) - event_onset
    ax.plot(x_range, familiar_mean.numpy(), label=labels[0], color=clrs[0], linewidth=lw)
    ax.plot(x_range, novel_mean.numpy(), label=labels[1], color=clrs[1], linewidth=lw)
    if sem:
        ax.fill_between(x_range,
                        familiar_mean - familiar_std,
                        familiar_mean + familiar_std,
                        color=clrs[0], alpha=0.4)
        ax.fill_between(x_range,
                        novel_mean - novel_std,
                        novel_mean + novel_std,
                        color=clrs[1], alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=14)

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
    ax.axvspan(args.blank_ts // 2, args.blank_ts // 2 + args.img_ts, color=PRE_CLR, alpha=0.05)
    
    # omitted image
    ax.axvline(args.blank_ts + args.blank_ts // 2 + args.img_ts, linestyle="--", color=PRE_CLR, linewidth=2.5)
    ax.axvline(args.blank_ts + args.blank_ts // 2 + 2 * args.img_ts, linestyle="--", color=PRE_CLR, linewidth=2.5)
    
    # last image
    ax.axvspan(2 * args.blank_ts + args.blank_ts // 2 + 2 * args.img_ts,
               2 * args.blank_ts + args.blank_ts // 2 + 3 * args.img_ts, color=PRE_CLR, alpha=0.05)
    
    ax.plot(response_mean.numpy(), label=label, color=trace_clr, linewidth=3.0)
    if sem:
        ax.fill_between(np.arange(responses.shape[1]),
                        response_mean - response_std,
                        response_mean + response_std,
                        color=trace_clr, alpha=0.25)
        
        
def raincloud_plot(ax, familiar_responses, novel_responses, xlabels=None, marker_sz=10., color_scheme=2):
    # color schemes: 0 -> familiar, familiar, 1 -> novel, novel, 2 -> familiar, novel

    if xlabels is None:
        xlabels = ['Familiar', 'Novel']

    # Create a list of colors for each component of the raincloud
    if color_scheme == 0:
        boxplots_colors = ['darkorange']*2
        median_colors = ['orangered']*2
        violin_colors = ['orange']*2
        scatter_colors = ['darkorange']*2
    elif color_scheme == 1:
        boxplots_colors = ['darkblue']*2
        median_colors = ['navy']*2
        violin_colors = ['cornflowerblue']*2
        scatter_colors = ['darkblue']*2
    else:
        boxplots_colors = ['darkorange', 'darkblue']
        median_colors = ['orangered', 'navy']
        violin_colors = ['orange', 'cornflowerblue']
        scatter_colors = ['darkorange', 'darkblue']

    # Boxplot data
    data = [familiar_responses, novel_responses]
    bp = ax.boxplot(data, patch_artist = True, vert = True, showmeans=True, showfliers=False,
                   meanprops={'markersize': marker_sz*2, 'markerfacecolor': 'teal'})

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

    # Scatterplot data
    for idx, features in enumerate(data):
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        ax.scatter(y, features, s=marker_sz, c=scatter_colors[idx])

    ax.set_xticks(np.arange(1,3,1), xlabels)  # Set text labels.
    ax.set_ylabel('Average response')


def plot_sequence_responses(
    responses, timestamps, seq_idx=0, pop_avg=False, perception_only=True
):
    """
    Plots all keys in the `responses` dict on separate subplots.
    Each subplot is titled with the key name.

    The function:
      - Selects the last 4 'before' intervals and the first 4 'after' intervals,
      - Includes 2 extra timesteps before the earliest 'before' interval,
      - Uses the final 'before' interval as time=0 (the image change boundary),
      - Either plots a single random unit if not pop_avg, or the mean across units if pop_avg,
      - If `perception_only=False`, also plots licks (i.e. `responses["action"]`) on the first subplot.
    """

    # Responses to show
    include_responses = ["z", "temp_error", "sigma_p", "mu_p", "theta"]
    resp_clrs = [EXC_CLR, EXC_CLR, VIP_CLR, SST_CLR, SST_CLR]

    # --- Extract 'before'/'after' intervals for this sequence ---
    before_on, before_off = timestamps[seq_idx]['before']
    after_on, after_off   = timestamps[seq_idx]['after']

    # Keep only 4 intervals from 'before' & 4 intervals from 'after'
    #before_on, before_off = before_on[-3:], before_off[-3:]
    #after_on, after_off   = after_on[:3], after_off[:3]

    # Compute start & end indices:
    #  - 2 timesteps before the earliest 'before' interval
    #  - the last 'after_off'
    start_idx = int(before_on[0]) - 1
    end_idx   = int(after_off[-1])

    # Define the image change time as the end of the 4th 'before' interval
    change_time = after_on[0]

    # Create as many subplots as there are keys
    fig, axes = plt.subplots(
        nrows=len(include_responses), ncols=1, figsize=(8, 1.5 * len(include_responses)), sharex=True
    )

    # If there's only one key, 'axes' is a single Axes object, convert it to a list
    if len(include_responses) == 1:
        axes = [axes]

    # Build the x-values for plotting (relative to change_time)
    full_indices = np.arange(start_idx, end_idx)
    x_vals_rel   = full_indices - change_time  # shift so change_time is zero

    # --- Optionally handle licks ---
    # We'll plot them on the first axis, if `perception_only=False` and 'action' in responses.
    plot_licks = (not perception_only) and ('action' in responses)
    if plot_licks:
        # Extract licks for this sequence
        licks = responses['action'][seq_idx]
        # Slice it
        licks = licks[start_idx:end_idx]
        # Find nonzero indices
        licks_inds = licks.nonzero(as_tuple=True)[0]
        # CPU + numpy conversion if necessary
        if hasattr(licks_inds, 'cpu'):
            licks_inds = licks_inds.cpu().detach().numpy()
        else:
            licks_inds = np.asarray(licks_inds)
        # Corresponding x-values
        licks_xvals = x_vals_rel[licks_inds]

    # --- Loop through each key and plot ---
    for i, key in enumerate(include_responses):
        ax = axes[i]
        ax.set_title(key)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i < len(axes)-1:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            

        # Extract data for this key and sequence
        data = responses[key][seq_idx]  # shape could be [time] or [time, units]

        # 1) If data is 2D: [time, units], handle pop_avg or single-unit
        if data.ndim == 2:
            if pop_avg:
                # average over last dimension
                data = data.mean(dim=-1)  
            else:
                # pick a random unit
                random_unit_idx = randint(0, data.shape[-1] - 1)
                data = data[..., random_unit_idx]

        # 2) Slice the data [start_idx:end_idx]
        data = data[start_idx:end_idx]

        # 3) Convert to numpy if it's a PyTorch tensor
        if hasattr(data, 'cpu'):
            data_np = data.cpu().detach().numpy()
        else:
            data_np = np.asarray(data)

        # 4) Plot the data
        ax.plot(x_vals_rel, data_np, linewidth=3., color=resp_clrs[i])

        # 5) Shading of intervals
        for bf_on, bf_off in zip(before_on, before_off):
            ax.axvspan(bf_on - change_time, bf_off - change_time, color=PRE_CLR, alpha=0.25)
        for af_on, af_off in zip(after_on, after_off):
            ax.axvspan(af_on - change_time, af_off - change_time, color=CHANGE_CLR, alpha=0.2)

        # Configure ticks and formatting
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if i < len(include_responses) - 1:
            ax.tick_params(axis='x', which='both', labelbottom=False)
        else:
            ax.set_xlabel("Time (relative to image change)")

    # --- Plot licks on the *first axis* (axes[0]) ---
    if plot_licks:
        ax0 = axes[0]
        ymin, ymax = ax0.get_ylim()
        vertical_center = (ymin + ymax) / 2
        ax0.plot(
            licks_xvals,
            [vertical_center] * len(licks_xvals),
            'bo',
            markersize=8,
            alpha=0.6,
            label='Licks'
        )

    plt.tight_layout()

    return fig, axes

def plot_dprimes(ax, epoch_arr, dprime_fam, dprime_nov, xlabel=None, title=None):
        
        fam_mean = dprime_fam.mean(0)
        fam_sem = dprime_fam.std(0) / np.sqrt(len(dprime_fam))

        nov_mean = dprime_nov.mean(0)
        nov_sem = dprime_nov.std(0) / np.sqrt(len(dprime_nov))

        # Plot familiar d'
        color1 = 'darkorange'
        ax.errorbar(epoch_arr, fam_mean, fam_sem, fmt='-o', color=color1, ecolor=color1, linewidth=2.5,
                    elinewidth=1.5, markersize=5, label='Familiar')
        ax.set_ylabel(r"$d'$", fontsize=16)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)

        # Plot novel d'
        color2 = 'darkblue'
        ax.errorbar(epoch_arr, nov_mean, nov_sem, fmt='-o', color=color2, ecolor=color2, linewidth=2.5,
                    elinewidth=1.5, markersize=5, label='Novel')

        # Plot chance level performance
        ax.plot(epoch_arr, np.zeros_like(fam_mean), '--', linewidth=2, color='tab:gray', label='Chance')

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=16)
        
        if title is not None:
            ax.set_title(title, fontsize=20)
        
        ax.legend(loc='upper left', frameon=False)
    
def plot_training_progress(args, training_prog, save_fig=False):
    """
    
    """

    def plot_prog_and_schedule(ax, prog_array, sched_array,
                               xlabel=None, title=None,
                               prog_label='Loss', sched_label=r"$\lambda$"):

        # Plot the first dataset
        color1 = 'tab:blue'
        ax.plot(np.arange(len(prog_array)), prog_array, color=color1, linewidth=2, label=prog_label)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(prog_label, color=color1, fontsize=16)
        ax.tick_params(axis='y', labelcolor=color1, labelsize=14)
        ax.tick_params(axis='x', labelsize=14)

        # Create a second y-axis
        twin_ax = ax.twinx()

        # Plot the second dataset
        color2 = 'tab:red'
        twin_ax.plot(np.arange(len(prog_array)), sched_array, color=color2, linewidth=2, linestyle='--', label=sched_label)
        twin_ax.set_ylabel(sched_label, color=color2, fontsize=16)
        twin_ax.tick_params(axis='y', labelcolor=color2, labelsize=14)

        # Add a legend (optional)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = twin_ax.get_legend_handles_labels()
        #ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)

        if title is not None:
            ax.set_title(title, fontsize=20) 
    
    # create lambda schedules
    lambda_spatial_sched = args.lambda_spatial * np.ones(args.num_epochs)
    
    # create lambda schedules
    lambda_temporal_sched = ramp_schedule(args.num_epochs, args.temporal_start, stop=args.lambda_temporal, stop_epoch=args.num_epochs)
    lambda_energy_sched = ramp_schedule(args.num_epochs, args.energy_start, stop=args.lambda_energy, stop_epoch=args.num_epochs)
    lambda_rew_sched = ramp_schedule(args.num_epochs, args.value_start, start=args.lambda_reward/10., stop=args.lambda_reward, stop_epoch=args.num_epochs)

    # epsilon schedule for greedy exploration
    epsilon_sched = ramp_schedule(args.num_epochs, args.value_start, start=0.5, stop=0.01, stop_epoch=args.num_epochs-10) #decreasing_ramp_schedule(args.num_epochs, args.value_start, 0.5, 0.01, stop_epoch=args.num_epochs-10)
    
    # learning rate schedule for total loss
    lr_sched = args.lr * np.ones((args.num_epochs,)) # stepLR_schedule(args.lr, args.num_epochs, step_size=100, gamma=0.5)
    
    # plot each schedule
    fig, axs = plt.subplots(2, 3, figsize=(12, 5), sharex='col')
    #with plt.style.context(['nature']):

    plot_prog_and_schedule(axs[0, 0], training_prog['total'].mean(0), lr_sched,
                            title="Total Loss", prog_label=r"$L_{total}$", sched_label=r"$\eta$")
        
    plot_prog_and_schedule(axs[0, 1], training_prog['spatial_error'].mean(0), lambda_spatial_sched,
                            title="Layer 1 Error", prog_label=r"$L_{pred}^{(1)}$", sched_label=r"$\lambda_1$")
    
    plot_prog_and_schedule(axs[0, 2], training_prog['temporal_error'].mean(0), lambda_temporal_sched,
                            title='Layer 2 Error', prog_label=r"$L_{pred}^{(2)}$", sched_label=r"$\lambda_2$")
    
    plot_prog_and_schedule(axs[1, 0], training_prog['energy'].mean(0), lambda_energy_sched,
                            title="Energy Loss", prog_label=r"$L_{energy}$", sched_label=r"$\lambda_{energy}$")
    
    plot_prog_and_schedule(axs[1, 1], training_prog['value_loss'].mean(0), lambda_rew_sched,
                            title="Reward Prediction Error", prog_label=r"$L_{action}$", sched_label=r"$\lambda_{action}$", xlabel='Epoch')
    
    plot_prog_and_schedule(axs[1, 2], training_prog['episode_rewards'].mean(0), epsilon_sched,
                            title="Reward", prog_label="Mean Reward", sched_label=r"$\epsilon$", xlabel='Epoch')
    
    #plot_dprimes(axs[1, 2], np.arange(args.num_epochs, step=5.), training_prog['dprime'].mean(0), training_prog['dprime_novel'].mean(0),
    #             title="Behavioral Performance", xlabel='Epoch')
    
    fig.tight_layout()

    if save_fig:
        fig.savefig("./figures/training_progress.pdf", dpi=800)
    
    return fig

def plot_example_reward_sequence(ax, args, title=None):

    # construct timestamps for an example sequence
    before_onsets = [args.blank_ts // 2, args.blank_ts//2 + args.img_ts + args.blank_ts]
    before_offsets = [i+args.img_ts for i in before_onsets]
    after_onsets = [before_offsets[-1] + args.blank_ts, before_offsets[-1] + 2 * args.blank_ts + args.img_ts]
    after_offsets = [i+args.img_ts for i in after_onsets]
    example_ts = [{
        "before": (before_onsets, before_offsets),
        "after": (after_onsets, after_offsets)
    }]
    seq_len = after_offsets[-1] + args.blank_ts//2

    # construct a reward sequence
    example_r = get_reward_sequence(1, seq_len, example_ts, reward_window=args.img_ts+2, reward_amount=10.0, action_cost=2.0)
    example_r = example_r[0].numpy()

    # plot repeat and change images
    for bf_on, bf_off in zip(*example_ts[0]['before']):
        ax.axvspan(bf_on, bf_off, color=PRE_CLR, alpha=0.25)        

    for af_on, af_off in zip(*example_ts[0]['after']):
        ax.axvspan(af_on, af_off, color=CHANGE_CLR, alpha=0.2)
    
    # plot reward sequence
    lick_color = "tab:blue"
    nolick_color = "tab:red"
    ax.plot(example_r, color=lick_color, linewidth=3.0, label="Lick Action")
    ax.plot(np.zeros_like(example_r), color=nolick_color, linewidth=3.0, label="No Lick Action")

    ax.set_ylabel("Reward Value", fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

    ax.set_xlabel("Time", fontsize=16)
        
    if title is not None:
        ax.set_title(title, fontsize=20)
    
    ax.legend(loc='upper left', frameon=False)

    return ax

def plot_confidence_intervals(ax, familiar_responses, novel_responses, alpha=0.05, xlabels=None):

    if xlabels is None:
        xlabels = ['Familiar', 'Novel']

    fam_mean, fam_err = compute_population_stats(familiar_responses, alpha=alpha)
    nov_mean, nov_err = compute_population_stats(novel_responses, alpha=alpha)
    
    # Create a list of colors for the boxplots based on the number of features you have
    colors = ['darkorange', 'darkblue']

    # error bar plot
    ax.errorbar([1], [fam_mean], yerr=fam_err, fmt='o', markersize=8, color=colors[0],
                 ecolor=colors[0], capsize=0, elinewidth=2)
    
    ax.errorbar([2], [nov_mean], yerr=nov_err, fmt='o', markersize=8, color=colors[1],
                 ecolor=colors[1], capsize=0, elinewidth=2)

    ax.set_xticks(np.arange(1,3,1), xlabels)  # Set text labels.
    ax.set_ylabel('Average response')
    ax.set_xlim([0.5, 2.5])

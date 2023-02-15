import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, SpanSelector

class DisplayRangeSelector:
    def __init__(self, kymo, px_min=0, px_max=255):
        self._px_min = px_min
        self._px_max = px_max
        self._min_slider = None # Sliders have to be retained to keep them active
        self._max_slider = None
        self._plot = None
        
        self._create_plot(kymo)
                        
    def _create_plot(self, kymo):
        def update_min_intensity(val):
            self._px_min = val
            self._update_plot()
            
        def update_max_intensity(val):
            self._px_max = val
            self._update_plot()
        
        fig, ax1 = plt.subplots(1, figsize=(8,5))
        fig.subplots_adjust(left=0.25, bottom=0.35)
        
        kymo_line_length = kymo.size_um[0]
        raw_kymo = kymo.get_image(channel='red')
        t_stop = (kymo.stop-kymo.start)*1E-9
        self._plot = ax1.imshow(raw_kymo, aspect=3, cmap='viridis', extent=[0, t_stop, 0, kymo_line_length], norm=mpl.colors.Normalize(vmin=self._px_min, vmax=self._px_max, clip=False))
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Distance ($\mu$m)', fontsize=12)
        
        ax_min_int = fig.add_axes([0.25, 0.2, 0.65, 0.03])
        self._min_slider = Slider(
            ax=ax_min_int,
            label='Min intensity',
            valmin=0,
            valmax=255, # Should be bit depth of image?
            valinit=self._px_min,
            valstep=1
        )
        self._min_slider.on_changed(update_min_intensity)

        ax_max_int = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self._max_slider = Slider(
            ax=ax_max_int,
            label='Max intensity',
            valmin=0,
            valmax=255, # Should be bit depth of image?
            valinit=self._px_max,
            valstep=1
        )
        self._max_slider.on_changed(update_max_intensity)    

    def _update_plot(self):
        self._plot.set_norm(mpl.colors.Normalize(vmin=self._px_min, vmax=self._px_max, clip=False))
        
    def get_range(self):
        return (self._px_min, self._px_max)

class TraceAnalyser:
    def __init__(self, kymo, time_ns, data, y_label = "Force (pN)", px_min=0, px_max=255):
        self._time_s = (time_ns - time_ns[0])/1000000000
        self._data = data
        self._y_label = y_label
        self._px_min = px_min
        self._px_max = px_max

        self._crop_plot_ax = None
        self._filt_crop_plot = None

        self._span_selector_kymo = None
        self._span_selector_trace = None
        self._range_min = 0
        self._range_max = 1
        
        self._create_plot(kymo)

    def _create_plot(self, kymo):
        def on_span_select(range_min, range_max):
            self._range_min = range_min
            self._range_max = range_max
            self._update_plot()
            
            # sos = butter(butter_order, butter_freq, 'lp', output='sos')
            # raw_data['Filtered '+channel] =  signal.sosfiltfilt(sos, raw_data[channel])
            # global data
            # data = raw_data.loc[(raw_data['Time (s)'] > xmin) & (raw_data['Time (s)'] < xmax)]
            
        # Initialising the plot
        fig, ((ax1, ax2), (ax3, ax4), (self._crop_plot_ax, ax6)) = plt.subplots(3, 2, figsize=(11, 11), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 1, 1.8]})
        fig.subplots_adjust(hspace=0.1, wspace=0.06)

        # Kymograph image
        kymo_line_length = kymo.size_um[0]
        raw_kymo = kymo.get_image(channel='red')
        t_stop = (kymo.stop-kymo.start)*1E-9
        ax1.imshow(raw_kymo, cmap='viridis', aspect='auto', extent=[0, t_stop, 0, kymo_line_length], norm=mpl.colors.Normalize(vmin=self._px_min, vmax=self._px_max, clip=False))
        ax1.set_ylabel('Distance ($\mu$m)')
        ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax1.set_xlabel("Time (s)")  
        ax1.xaxis.set_label_position('top') 
        self._span_selector_kymo = SpanSelector(
            ax1,
            on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )

        ax2.axis('off')

        # Full length trace
        ax3.plot(self._time_s, self._data, color='black')
        ax3.set_ylabel(self._y_label)
        ax3.get_shared_x_axes().join(ax1, ax3)
        ax3.set_xticklabels([])
        self._span_selector_trace = SpanSelector(
            ax3,
            on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )

        ax4.axis('off')

        # Cropped trace
        self._crop_plot_ax.set_xlabel("Time (s)")
        self._crop_plot_ax.set_ylabel(self._y_label)
        self._crop_plot_ax.plot(self._time_s, self._data, label='Raw data', color='black', alpha=.3)
        self._filt_crop_plot = self._crop_plot_ax.plot([], [], label='Filtered data')

        # Histogram of values in cropped trace
        ax6.set_xlabel("Counts")
        ax6.set_title('Frequencies (raw force)', fontsize=8)
        ax6.get_shared_y_axes().join(self._crop_plot_ax, ax6)
        ax6.set_yticklabels([])
        ax6.minorticks_on()
        ax6.yaxis.set_tick_params(which='minor', bottom=False)

    def _update_plot(self):
        # region_y2 = data['Filtered '+channel]
        
        # counts, bins = np.histogram(data[channel], bins=150)
        # bin_mids = (bins[1:] + bins[:-1])/2
        
        # region_min = data[channel].min()
        # region_max = data[channel].max()
        # region_range = data[channel].max() - data[channel].min()

        if len(self._time_s) >= 2:    
            self._crop_plot_ax.set_xlim(self._range_min, self._range_max)
            # ax5.set_ylim(region_min- (region_range*0.05), region_max + (region_range*0.05))
            # ax5.legend(fontsize=9)
            # plot3.set_data(region_x, region_y2)
            
            # ax6.set_xlim(0, counts.max())
            # plot4.set_data(counts, bin_mids)
            # fig.canvas.draw_idle()
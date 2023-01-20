import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

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
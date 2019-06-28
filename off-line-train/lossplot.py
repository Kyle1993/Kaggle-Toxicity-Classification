import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

from matplotlib.figure import Figure
import numpy as np
import pickle

class LossPlot:
    '''
    record should be [(step0,value0),(step1,value1),...,(stepn,valuen)]
    '''
    def __init__(self,alpha=0.3,show_legend=True,window_size=(16, 8),single_select=False,title='Loss Polt'):
        self.records = []
        self.alpha = alpha
        self.single_select = single_select
        self.ID = 0
        self.curves_select = []
        self.init_smooth = 0
        self.init_x_range = [None,None]
        self.init_y_range = [None,None]
        self.smooth = self.init_smooth
        self.x_range = self.init_x_range
        self.y_range = self.init_y_range
        self.select_ID = None
        self.show_legend = show_legend

        self.title = title

        self.fig = Figure(figsize=window_size)
        self.ax, self.ax_text = self.fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},)
        self.ax.set_title(self.title)
        self.ax_text.set_title('Summary')
        self.ax_text.axis('off')
        self.print_()


        # set Tk
        self.root = Tk.Tk()
        self.root.wm_title("LossPloter")

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        reset_button = Tk.Button(master=self.root, text='Reset', command=self.reset_clicked,width=5)
        reset_button.place(relx=.63, rely=.95, anchor="c")

        set_button = Tk.Button(master=self.root, text='Set', command=self.set_click,width=5)
        set_button.place(relx=.53, rely=.95, anchor="c")

        exit_button = Tk.Button(master=self.root, text='Exit', command=self._quit,width=5)
        exit_button.place(relx=.83, rely=.95, anchor="c")

        selectAll_button = Tk.Button(master=self.root, text='Select All',command=self.select_all, width=8)
        selectAll_button.place(relx=.62, rely=.15, anchor="c")

        selectNone_button = Tk.Button(master=self.root, text='Select None',command=self.select_none, width=8)
        selectNone_button.place(relx=.62, rely=.2, anchor="c")

        showlegend_button = Tk.Button(master=self.root, text='Show Legend',command=self.show_legend_command, width=8)
        showlegend_button.place(relx=.62, rely=.25, anchor="c")

        smooth_label = Tk.Label(self.root,text='smooth:')
        smooth_label.place(relx=.15, rely=.95, anchor="c")
        self.smooth_text = Tk.Entry(master=self.root,width=5,)
        self.smooth_text.config(state=Tk.NORMAL)
        self.smooth_text.insert(Tk.END,self.init_smooth)
        self.smooth_text.place(relx=0.18, rely=.95, anchor="c")

        xrange_label = Tk.Label(self.root, text='x range:')
        xrange_label.place(relx=.25, rely=.95, anchor="c")
        self.xrange_text = Tk.Entry(master=self.root, width=10, )
        self.xrange_text.config(state=Tk.NORMAL)
        self.xrange_text.insert(Tk.END, '{},{}'.format(self.init_x_range[0],self.init_x_range[1]))
        self.xrange_text.place(relx=0.29, rely=.95, anchor="c")

        yrange_label = Tk.Label(self.root, text='y range:')
        yrange_label.place(relx=.35, rely=.95, anchor="c")
        self.yrange_text = Tk.Entry(master=self.root, width=10, )
        self.xrange_text.config(state=Tk.NORMAL)
        self.yrange_text.insert(Tk.END, '{},{}'.format(self.init_y_range[0],self.init_y_range[1]))
        self.yrange_text.place(relx=0.39, rely=.95, anchor="c")


    def print_(self):
        print('Note:')
        print('1.select cruves by click curve or legend(deep color means selected)')
        print("2.single select mode can be set by 'single_select=True', defalut False")
        print('3.compare only in selected curves')
        print('4.Config & Curve Summary will print last click curve no matter whether it be selected')

    def record_summary(self,record):
        min_data = min(record,key=lambda x:x[1])
        max_data = max(record,key=lambda x:x[1])
        return {'min':min_data,'max':max_data}


    def append(self,record,name=None,color=None,note=None):
        record = np.asarray(record)
        summary = self.record_summary(record)
        self.records.append({'record':record,'color':color,'name':name,'note':note,'summary':summary})
        self.ID += 1
        self.curves_select.append(True)

    def append_rcd(self,rcd_file,color=None):
        with open(rcd_file,'rb') as f:
            rcd = pickle.load(f)
        record = rcd['record']
        name = rcd['name']
        note = rcd['note']
        self.append(record,name,color,note)

    def window_smooth(self,record,smooth):
        assert smooth >= 0
        res = np.copy(record)
        for i in range(record.shape[0]):
            window_min = max(0,i-smooth)
            window_max = min(record.shape[0]-1,i+smooth)+1
            res[i,1] = np.mean(res[window_min:window_max,1])
        return res

    def clip_record(self,record,x_range,y_range):
        res = np.copy(record)

        x_min = res[:,0].min() if x_range[0] is None else max(x_range[0],res[:,0].min())
        x_max = res[:,0].max() if x_range[1] is None else min(x_range[1],res[:,0].max())
        assert x_min < x_max

        res = res[res[:,0]>=x_min]
        res = res[res[:,0]<=x_max]

        y_min = res[:,1].min() if y_range[0] is None else y_range[0]
        y_max = res[:,1].max() if y_range[1] is None else y_range[1]
        assert y_min < y_max

        res[:,1] = np.clip(res[:,1],y_min,y_max)

        return res

    def get_curve_summary_string(self,ID):
        note = self.records[ID]['note']
        summary = self.records[ID]['summary']

        string = ''
        if isinstance(note,str):
            string += note
        elif isinstance(note,dict):
            for i,v in sorted(note.items(),key=lambda x:x[0]):
                string += '{:<10}:{:<10}\n'.format(i,str(v))
        else:
            string += 'None\n'

        min_data = summary['min']
        max_data = summary['max']
        string += '\nCurve Summary:\n'
        string += 'Min Value:{:<.5f} at step {:<5d}\n'.format(min_data[1],int(min_data[0]))
        string += 'Max Value:{:<.5f} at step {:<5d}\n'.format(max_data[1],int(max_data[0]))

        return string

    def get_total_summary_string(self, ):
        min_value = np.inf
        min_cruve = None
        max_value = -np.inf
        max_cruve = None
        for r, s in zip(self.records, self.curves_select):
            if s:
                if r['summary']['max'][1] > max_value:
                    max_cruve = r['name']
                if r['summary']['min'][1] < min_value:
                    min_cruve = r['name']
        string = ''
        string += 'Max Value in Curve:{}\n'.format(max_cruve)
        string += 'Min Value in Curve:{}\n'.format(min_cruve)
        string += '\nNote: compare only in selected\n      curves'

        return string

    def refresh_summary(self,):
        self.ax_text.cla()
        self.ax_text.set_title('Summary')
        self.ax_text.axis('off')

        if self.select_ID is None:
            self.ax_text.text(0.03, 0.98, 'No Select', horizontalalignment='left', verticalalignment='top',fontdict={'family': 'monospace', 'weight': 'bold'})
            self.ax_text.text(0.03, 0.93, '', horizontalalignment='left', verticalalignment='top',fontdict={'family': 'monospace','weight':'medium'})
        else:
            name = self.records[self.select_ID]['name']
            config_str = self.get_curve_summary_string(self.select_ID)
            self.ax_text.text(0.03, 0.98, name,horizontalalignment='left',verticalalignment='top',fontdict={'family': 'monospace','weight':'bold'})
            self.ax_text.text(0.03, 0.93, 'Config:',horizontalalignment='left',verticalalignment='top',fontdict={'family': 'monospace','weight':'bold'})
            self.ax_text.text(0.03, 0.90, config_str,horizontalalignment='left',verticalalignment='top',fontdict={'family': 'monospace','weight':'medium'})

        summary = self.get_total_summary_string()
        self.ax_text.text(0.03, 0.3, 'Summary:', horizontalalignment='left', verticalalignment='top',fontdict={'family': 'monospace', 'weight': 'bold'})
        self.ax_text.text(0.03, 0.25, summary, horizontalalignment='left', verticalalignment='top',fontdict={'family': 'monospace', 'weight': 'medium'})

    def refresh_fig(self,):
        self.ax.cla()
        self.ax.set_title(self.title)

        self.curves = []
        for curve_data in self.records:
            r = self.window_smooth(curve_data['record'],self.smooth)
            r = self.clip_record(r,self.x_range,self.y_range)
            c, = self.ax.plot(r[:,0],r[:,1],lw=1,color=curve_data['color'],label=str(curve_data['name'])) # 注意!要个','
            self.curves.append(c)

        if self.show_legend:
            leg = self.ax.legend(loc='upper left', fancybox=True, shadow=True)
            leg.get_frame().set_alpha(0.8)

            self.leglines = leg.get_lines()

            self.legline2ID = {}
            for i, legline in enumerate(self.leglines):
                legline.set_picker(5)
                self.legline2ID[legline] = i

        self.line2ID = {}
        for i,line in enumerate(self.curves):
            line.set_picker(3)
            self.line2ID[line] = i

        # re-plot
        for i in range(self.ID):
            if self.curves_select[i]:
                self.curves[i].set_alpha(1)
            else:
                self.curves[i].set_alpha(self.alpha)

        if self.show_legend:
            for i in range(self.ID):
                if self.curves_select[i]:
                    self.leglines[i].set_alpha(1)
                else:
                    self.leglines[i].set_alpha(self.alpha)

    def reset_clicked(self,):
        for i,_ in enumerate(self.curves_select):
            self.curves_select[i] = True
        self.select_ID = None
        self.smooth = self.init_smooth
        self.x_range = self.init_x_range
        self.y_range = self.init_y_range

        self.smooth_text.delete(0, Tk.END)
        self.xrange_text.delete(0, Tk.END)
        self.yrange_text.delete(0, Tk.END)
        self.smooth_text.insert(Tk.END,self.init_smooth)
        self.xrange_text.insert(Tk.END, '{},{}'.format(self.init_x_range[0],self.init_x_range[1]))
        self.yrange_text.insert(Tk.END, '{},{}'.format(self.init_y_range[0],self.init_y_range[1]))


        self.refresh_fig()
        self.refresh_summary()

        self.fig.canvas.draw()


    def set_click(self):
        if self.smooth_text.get() != '':
            self.smooth = int(self.smooth_text.get())
        if len(self.xrange_text.get().split(','))==2:
            xrange = self.xrange_text.get()
            self.x_range = [int(x) if x != 'None' else None for x in xrange.split(',')]
        if len(self.yrange_text.get().split(','))==2:
            yrange = self.yrange_text.get()
            self.y_range = [float(y) if y!='None' else None for y in yrange.split(',')]

        self.refresh_fig()

        self.fig.canvas.draw()

    def select_all(self):
        self.curves_select = [True for i in range(self.ID)]

        self.refresh_fig()
        self.refresh_summary()

        self.fig.canvas.draw()

    def select_none(self):
        self.curves_select = [False for i in range(self.ID)]

        self.refresh_fig()
        self.refresh_summary()

        self.fig.canvas.draw()

    def show_legend_command(self):
        self.show_legend = not self.show_legend

        self.refresh_fig()
        self.refresh_summary()

        self.fig.canvas.draw()


    def onpick(self,event, ):
        click = event.artist

        # reset
        if self.single_select:
            self.curves_select = [False for i in range(self.ID)]

        if click in self.legline2ID:
            self.select_ID = self.legline2ID[click]
            self.curves_select[self.select_ID] = not self.curves_select[self.select_ID]
        elif click in self.line2ID:
            self.select_ID = self.line2ID[click]
            self.curves_select[self.select_ID] = not self.curves_select[self.select_ID]

        self.refresh_fig()
        self.refresh_summary()

        self.fig.canvas.draw()

    def _quit(self,):
        self.root.quit()
        self.root.destroy()

    def show(self):
        self.refresh_fig()
        self.refresh_summary()

        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.root.mainloop()




if __name__ == '__main__':
    import pickle


    ploter = LossPlot(alpha=0.3,single_select=False,)

    # ploter.append_rcd('train_record.rcd')
    # ploter.append_rcd('validate_record.rcd')
    # ploter.append_rcd('train_record1.rcd')
    # ploter.append_rcd('validate_record1.rcd')
    with open('bert_2epoch_lower/train_record_fold1.pkl','rb') as f:
        rec = pickle.load(f)
    ploter.append(rec,'2epoch')


    ploter.show()

"""
Created on %(20-December-2016)s

@author: %(Mustafa)s
"""

import sys
import os
import timeit
import time
import traceback
import faulthandler
import multiprocessing

from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as etree

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QPushButton,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QMdiArea,
    QTextEdit,
    QMdiSubWindow)

from normcopinfill import NormCopulaInfill
NormCopulaInfill.verbose = True

faulthandler.enable()


class OutLog:

    '''
    A class to print console messages to the messages window in the GUI
    '''

    def __init__(self, edit, out=None):

        """(edit, out=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        """

        self.edit = edit
        self.out = out
        return

    def write(self, m):

        self.edit.insertPlainText(m)
        return


class MainWindow(QTabWidget):

    '''
    Control panel of the plotting GUI
    '''

    def __init__(self, parent=None):

        super(MainWindow, self).__init__(parent)
        self.conf_input_tab = QWidget()
        self.flags_tab = QWidget()
        self.addTab(self.conf_input_tab, 'Configure Input')
        self.currentChanged.connect(self.back_tab)

        self.addTab(self.flags_tab, 'Flags')
        self.conf_input_tab_UI()
        self.flags_tab_UI()
        self.flags_tab.setDisabled(True)
        self.gui_debug = True
        self.read_succ = True
        print('#' * 50)
        print('Ready....')
        print('#' * 50, '\n')

        if not self.plot_ecops_check_box.isChecked() == True:
            self.ecop_bins_line.setDisabled(True)
        if not self.n_rand_infill_check_box.isChecked() == True:
            self.n_rand_infill_line.setDisabled(True)
        if not self.min_corr_check_box.isChecked() == True:
            self.min_corr_line.setDisabled(True)
        if not self.max_time_check_box.isChecked() == True:
            self.max_time_lag_corr_line.setDisabled(True)
        if not self.cut_cdf_thresh_check_box.isChecked() == True:
            self.cut_cdf_thresh_line.setDisabled(True)

        self.plot_ecops_check_box.stateChanged.connect(self.ecops_state)
        self.n_rand_infill_check_box.stateChanged.connect(self.n_rand_infill_state)
        self.min_corr_check_box.stateChanged.connect(self.min_corr_state)
        self.max_time_check_box.stateChanged.connect(self.max_time_state)
        self.cut_cdf_thresh_check_box.stateChanged.connect(self.cut_cdf_thresh_state)
        return

    def conf_input_tab_UI(self):
        '''
        The Configure Input tab
        '''
        # # Create Grids ##
        self.main_layout = QVBoxLayout()
        self.grid = QGridLayout()

        i = 0

        # # Input File Box ##
        self.in_q_orig_file_lab = QLabel('Input File:')
        self.in_q_orig_file_line = QLineEdit()
        self.in_q_orig_file_line.setText(r'P:/Synchronize/IWS/Discharge_data_longer_series/final_q_data/neckar_q_data.csv')
        self.in_q_orig_file_btn = QPushButton('Browse')
        self.in_q_orig_file_btn.clicked.connect(lambda: self.get_input_dir(_input=True))
        self.grid.addWidget(self.in_q_orig_file_lab, i, 0)
        self.grid.addWidget(self.in_q_orig_file_line, i, 1, 1, 20)
        self.grid.addWidget(self.in_q_orig_file_btn, i, 21)
        self.in_q_orig_file_lab.setToolTip(str('Full path to the input csv file.'))
        self.in_q_orig_file_line.setToolTip(str('Full path to the input csv file.'))
        self.in_q_orig_file_btn.setToolTip(str('Browse to the input csv file.'))
        self.in_q_orig_file = self.in_q_orig_file_line.text()

        i += 1

        # # Coordinates file box ##
        self.in_coords_file_lab = QLabel('Coordinates File:')
        self.in_coords_file_line = QLineEdit()
        self.in_coords_file_btn = QPushButton('Browse')
        self.in_coords_file_btn.clicked.connect(lambda: self.get_coords_dir(_input=True))
        self.grid.addWidget(self.in_coords_file_lab, i, 0)
        self.grid.addWidget(self.in_coords_file_line, i, 1, 1, 20)
        self.grid.addWidget(self.in_coords_file_btn, i, 21)
        self.in_coords_file_lab.setToolTip(str('Full path to the input coordinates file.'))
        self.in_coords_file_line.setToolTip(str('Full path to the input coordinates file.'))
        self.in_coords_file_btn.setToolTip(str('Browse to the input coordinates file.'))
        self.in_coords_file = self.in_coords_file_line.text()
        i += 1

        # # Output directory box ##
        self.out_dir_lab = QLabel('Output Directory:')
        self.out_dir_line = QLineEdit()
        self.out_dir_btn = QPushButton('Browse')
        self.out_dir_btn.clicked.connect(lambda: self.get_out_dir(_input=True))
        self.grid.addWidget(self.out_dir_lab, i, 0)
        self.grid.addWidget(self.out_dir_line, i, 1, 1, 20)
        self.grid.addWidget(self.out_dir_btn, i, 21)
        self.out_dir_lab.setToolTip(str('Directory to the output folder (where the results will be saved).'))
        self.out_dir_line.setToolTip(str('Directory to the output folder (where the results will be saved).'))
        self.out_dir_btn.setToolTip(str('Browse to output directory'))
        self.out_dir = self.out_dir_line.text()
        i += 1

        # # Date Format box ##
        self.date_fmt_lab = QLabel('Date Format:')
        self.date_fmt_box = QComboBox()
        self.date_fmt_box.addItems(['1999-03-15', '1999-Mar-15', '1999-March-15',
                                    '99-03-15', '99-Mar-15', '99-March-15',
                                    '15-03-1999', '15-Mar-1999', '15-March-1999',
                                    '15-03-99', '15-Mar-99', '15-March-99',
                                    '03-15-1999', 'Mar-15-1999', 'March-15-1999',
                                    '03-15-99', 'Mar-15-99', 'March-15-99',
                                    '1999.03.15', '1999.Mar.15', '1999.March.15',
                                    '99.03.15', '99.Mar.15', '99.March.15',
                                    '15.03.1999', '15.Mar.1999', '15.March.1999',
                                    '15.03.99', '15.Mar.99', '15.March.99',
                                    '03.15.1999', 'Mar.15.1999', 'March.15.1999',
                                    '03.15.99', 'Mar.15.99', 'March.15.99',
                                    '1999/03/15', '1999/Mar/15', '1999/March/15',
                                    '99/03/15', '99/Mar/15', '99/March/15',
                                    '15/03/1999', '15/Mar/1999', '15/March/1999',
                                    '15/03/99', '15/Mar/99', '15/March/99',
                                    '03/15/1999', 'Mar/15/1999', 'March/15/1999',
                                    '03/15/99', 'Mar/15/99', 'March/15/99'])

        self.grid.addWidget(self.date_fmt_lab, i, 0)
        self.grid.addWidget(self.date_fmt_box, i, 1, 1, 20)
        self.date_fmt_lab.setToolTip(str('Select Date Format.'))
        self.date_fmt_box.setToolTip(str('Select Date Format.'))
        i += 1

        # # Infill Stations box ##
        self.infill_stns_lab = QLabel('Infill Stations:')
        self.infill_stns_line = QLineEdit(str(''))
        self.grid.addWidget(self.infill_stns_lab, i, 0)
        self.grid.addWidget(self.infill_stns_line, i, 1, 1, 20)
        self.infill_stns_lab.setToolTip(str('list of infill stations'))
        self.infill_stns_line.setToolTip(str('list of infill stations.'))
        i += 1

        # # Drop stations box ##
        self.drop_stns_lab = QLabel('Drop Stations:')
        self.drop_stns_line = QLineEdit('')
        self.grid.addWidget(self.drop_stns_lab, i, 0)
        self.grid.addWidget(self.drop_stns_line, i, 1, 1, 20)
        self.drop_stns_lab.setToolTip(str('list of stations to drop.'))
        self.drop_stns_line.setToolTip(str('list of stations to drop.'))
        i += 1

        # # Censor period box ##
        self.censor_period_lab = QLabel('Censor Period:')
        self.censor_period_line = QLineEdit('')
        self.grid.addWidget(self.censor_period_lab, i, 0)
        self.grid.addWidget(self.censor_period_line, i, 1, 1, 20)
        self.drop_stns_lab.setToolTip(str('time period of simulation'))
        self.drop_stns_line.setToolTip(str('time period of simulation.'))
        i += 1

        # # Interval type box##
        self.infill_interval_type_lab = QLabel('Infill interval type:')
        self.infill_interval_type_box = QComboBox()
        self.infill_interval_type_box.addItems(['slice', 'individual', 'all'])
        self.grid.addWidget(self.infill_interval_type_lab, i, 0)
        self.grid.addWidget(self.infill_interval_type_box, i, 1, 1, 20)
        self.infill_interval_type_lab.setToolTip(str('Type of infill.'))
        self.infill_interval_type_box.setToolTip(str('Type of infill.'))
        i += 1

        # # Infill type box ##
        self.infill_type_lab = QLabel('Infill type:')
        self.infill_type_box = QComboBox()
        self.infill_type_box.addItems(['discharge', 'precipitation', 'discharge-censored'])
        self.grid.addWidget(self.infill_type_lab, i, 0)
        self.grid.addWidget(self.infill_type_box, i, 1, 1, 20)
        self.infill_type_lab.setToolTip(str('Infill Type.'))
        self.infill_type_box.setToolTip(str('Infill Type.'))
        i += 1

        # # minimum valid values box ##
        self.min_valid_vals_lab = QLabel('Minimum Valid Values:')
        self.min_valid_vals_line = QLineEdit()
        self.grid.addWidget(self.min_valid_vals_lab, i, 0)
        self.grid.addWidget(self.min_valid_vals_line, i, 1, 1, 20)
        self.min_valid_vals_lab.setToolTip(str('Minimum Valid Values.'))
        self.min_valid_vals_line.setToolTip(str('Minimum Valid Values.'))
        i += 1

        # # Minimum nearest stations box ##
        self.n_nrn_min_lab = QLabel('Minimum Nearest stations:')
        self.n_nrn_min_line = QLineEdit()
        self.grid.addWidget(self.n_nrn_min_lab, i, 0)
        self.grid.addWidget(self.n_nrn_min_line, i, 1, 1, 20)
        self.n_nrn_min_lab.setToolTip(str('Minimum Nearest stations'))
        self.n_nrn_min_line.setToolTip(str('Minimum Nearest stations'))
        i += 1

        # # Maximum nearest stations box ##
        self.n_nrn_max_lab = QLabel('Maximum Nearest Stations:')
        self.n_nrn_max_line = QLineEdit()
        self.grid.addWidget(self.n_nrn_max_lab, i, 0)
        self.grid.addWidget(self.n_nrn_max_line, i, 1, 1, 20)
        self.n_nrn_max_lab.setToolTip(str('Maximum Nearest Stations.'))
        self.n_nrn_max_line.setToolTip(str('Maximum Nearest Stations.'))
        i += 1

        # # number of cpus box ##
        self.ncpus_lab = QLabel('Number of CPUs:')
        self.ncpus_box = QComboBox()
        self.xcpus = multiprocessing.cpu_count()
        self.rncpus = list(range(1, self.xcpus + 1))
        self.strncpus = str(",".join(str(x) for x in self.rncpus))
        self.ncpus_box.addItems(self.strncpus.split(","))
        self.grid.addWidget(self.ncpus_lab, i, 0)
        self.grid.addWidget(self.ncpus_box, i, 1, 1, 20)
        self.ncpus_lab.setToolTip(str('Number of CPUs.'))
        self.ncpus_box.setToolTip(str('Number of CPUs.'))
        i += 1

        # # Separator box ##
        self.sep_lab = QLabel('Seperator:')
        self.sep_box = QComboBox()
        self.sep_box.addItems([';', ',', ':', '.'])
        self.grid.addWidget(self.sep_lab, i, 0)
        self.grid.addWidget(self.sep_box, i, 1, 1, 20)
        self.sep_lab.setToolTip(str('seperator.'))
        self.sep_box.setToolTip(str('seperator.'))
        i += 1

        # # Frequency box ##
        self.freq_lab = QLabel('Freq:')
        self.freq_box = QComboBox()
        self.freq_box.addItems(['D', 's', 'min', 'H', 'w', 'm'])
        self.grid.addWidget(self.freq_lab, i, 0)
        self.grid.addWidget(self.freq_box, i, 1, 1, 20)
        self.freq_lab.setToolTip(str('frequency.'))
        self.freq_box.setToolTip(str('frequency.'))
        i += 1

        # # Save button ##
        self.save_lab = 'Save Configuration file'
        self.save_btn = QPushButton()
        self.save_btn.setText(self.save_lab)
        self.save_btn.clicked.connect(lambda: self.save_xml_dir())
        self.grid.addWidget(self.save_btn, i, 1, 1, 9)
        self.save_btn.setToolTip(str('Save configuration file'))

        # # Load button ##
        self.load_lab = 'Load Configuration file'
        self.load_btn = QPushButton()
        self.load_btn.setText(self.load_lab)
        self.load_btn.clicked.connect(lambda: self.load_file())
        self.grid.addWidget(self.load_btn, i, 12, 1, 9)
        self.load_btn.setToolTip(str('Load configuration file'))
        i += 1

        # # Read data button ##
        self.read_data_lab = 'Proceed to next tab!'
        self.read_data_btn = QPushButton()
        self.read_data_btn.setText(self.read_data_lab)
        self.read_data_btn.clicked.connect(lambda: self.read_data())
        self.grid.addWidget(self.read_data_btn, i, 1, 1, 20)
        self.read_data_btn.setToolTip(str('Read data and move to the next tab!'))
        i += 1

        self.main_layout.addLayout(self.grid)
        self.conf_input_tab.setLayout(self.main_layout)

    def flags_tab_UI(self):
        '''
        The Render tab
        '''
        self.flags_layout = QVBoxLayout()
        self.flags_grid = QGridLayout()
        i = 0

        # # Debug mode flag ##
        self.debug_check_box = QCheckBox('Debug Mode', self)
        self.debug_check_box.setText("Debug Mode")
        self.flags_grid.addWidget(self.debug_check_box, i, 1, 1, 10)

        # # Compare infill flag ##
        self.compare_infill_check_box = QCheckBox('Compare Infill', self)
        self.compare_infill_check_box.setText("Compare Infill")
        self.flags_grid.addWidget(self.compare_infill_check_box, i, 10, 1, 10)
        i += 1

        # # Flag susp flag ##
        self.flag_susp_check_box = QCheckBox('Flag Susp', self)
        self.flag_susp_check_box.setText("Flag Susp")
        self.flags_grid.addWidget(self.flag_susp_check_box, i, 1, 1, 10)

        # # Force infill flag ##
        self.force_infill_check_box = QCheckBox('Force infill', self)
        self.force_infill_check_box.setText("Force infill")
        self.flags_grid.addWidget(self.force_infill_check_box, i, 10, 1, 10)
        i += 1

        # # Take minimum stations flag ##
        self.take_min_stns_check_box = QCheckBox('Take min stations', self)
        self.take_min_stns_check_box.setText("Take Min stations")
        self.flags_grid.addWidget(self.take_min_stns_check_box, i, 1, 1, 10)
        self.take_min_stns_check_box.setChecked(True)

        # # read pickles flag ##
        self.read_pickles_check_box = QCheckBox('Read pickles', self)
        self.read_pickles_check_box.setText("Read pickles")
        self.flags_grid.addWidget(self.read_pickles_check_box, i, 10, 1, 10)
        i += 1

        # # overwrite flag ##
        self.overwrite_check_box = QCheckBox('Overwrite', self)
        self.overwrite_check_box.setText("Overwrite")
        self.flags_grid.addWidget(self.overwrite_check_box, i, 1, 1, 10)

        # # plot diag flag ##
        self.plot_diag_check_box = QCheckBox('Plot Diag', self)
        self.plot_diag_check_box.setText("Plot Diag")
        self.flags_grid.addWidget(self.plot_diag_check_box, i, 10, 1, 10)
        i += 1

        # # plot step flag ##
        self.plot_step_cdf_pdf_check_box = QCheckBox('plot step cdf pdf', self)
        self.plot_step_cdf_pdf_check_box.setText("plot step cdf pdf")
        self.flags_grid.addWidget(self.plot_step_cdf_pdf_check_box, i, 1, 1, 10)

        # # plot neighbours flag ##
        self.plot_neighbors_flag_check_box = QCheckBox('plot neighbours', self)
        self.plot_neighbors_flag_check_box.setText("plot neighbours")
        self.flags_grid.addWidget(self.plot_neighbors_flag_check_box, i, 10, 1, 10)
        i += 1

        # # Ignore bad stations ##
        self.ignore_bad_stns_flag_check_box = QCheckBox('Ignore bad stations', self)
        self.ignore_bad_stns_flag_check_box.setText("Ignore bad stations")
        self.flags_grid.addWidget(self.ignore_bad_stns_flag_check_box, i, 1, 1, 10)
        self.ignore_bad_stns_flag_check_box.setChecked(True)

        # # use best stations flag ##
        self.use_best_stns_flag_check_box = QCheckBox('Use best Stations', self)
        self.use_best_stns_flag_check_box.setText("Use best stations")
        self.flags_grid.addWidget(self.use_best_stns_flag_check_box, i, 10, 1, 10)
        self.use_best_stns_flag_check_box.setChecked(True)
        i += 1

        # # dont stop flag ##
        self.dont_stop_flag_check_box = QCheckBox('Dont stop', self)
        self.dont_stop_flag_check_box.setText("Dont stop")
        self.flags_grid.addWidget(self.dont_stop_flag_check_box, i, 1, 1, 10)

        # # plot long term corrs flag ##
        self.plot_long_term_corrs_flag_check_box = QCheckBox('Plot long term corrs', self)
        self.plot_long_term_corrs_flag_check_box.setText("Plot long term corrs")
        self.flags_grid.addWidget(self.plot_long_term_corrs_flag_check_box, i, 10, 1, 10)
        i += 1

        # # plot nearest stations flag ##
        self.plot_nrst_stns_flag_check_box = QCheckBox("Plot nearest stations", self)
        self.plot_nrst_stns_flag_check_box.setText("Plot nearest stations")
        self.flags_grid.addWidget(self.plot_nrst_stns_flag_check_box, i, 1, 1, 10)

        # # nearest stations box ##
        self.nrst_stns_type_lab = QLabel('nrst stns type:')
        self.nrst_stns_type_box = QComboBox()
        self.nrst_stns_type_box.addItems(['dist', 'rank', 'symm'])
        self.flags_grid.addWidget(self.nrst_stns_type_lab, i, 10, 1, 1)
        self.flags_grid.addWidget(self.nrst_stns_type_box, i, 11, 1, 2)

        i += 1

        # # plot ecops flag ##
        self.plot_ecops_check_box = QCheckBox("Plot ecops", self)
        self.flags_grid.addWidget(self.plot_ecops_check_box, i, 1, 1, 10)
        self.ecop_bins_lab = QLabel('ecops:')
        self.ecop_bins_line = QLineEdit()
        self.ecop_bins_line.setText("15")
        self.flags_grid.addWidget(self.ecop_bins_lab, i, 10, 1, 10)
        self.flags_grid.addWidget(self.ecop_bins_line, i, 11, 1, 2)
        self.ecop_bins_lab.setToolTip(str('Directory to the output folder (where the results will be saved).'))
        self.ecop_bins_line.setToolTip(str('Directory to the output folder (where the results will be saved).'))
        i += 1

        # # Save step vars flag ##
        self.save_step_vars_flag_check_box = QCheckBox('Save step vars', self)
        self.flags_grid.addWidget(self.save_step_vars_flag_check_box, i, 1, 1, 10)

        # # cmpt plot stats flag ##
        self.cmpt_plot_stats_check_box = QCheckBox("Cmpt plot stats", self)
        self.flags_grid.addWidget(self.cmpt_plot_stats_check_box, i, 10, 1, 10)
        i += 1

        # # cmpt plot avail stns flag ##
        self.cmpt_plot_avail_stns_check_box = QCheckBox("Cmpt plot avail stns", self)
        self.flags_grid.addWidget(self.cmpt_plot_avail_stns_check_box, i, 1, 1, 10)

        self.plot_rand_check_box = QCheckBox("Plot Rand", self)
        self.flags_grid.addWidget(self.plot_rand_check_box, i, 10, 1, 10)
        i += 1

        self.stn_based_mp_infill_check_box = QCheckBox("station based mp infill")
        self.flags_grid.addWidget(self.stn_based_mp_infill_check_box, i, 1, 1, 10)
        i += 1

        self.n_rand_infill_check_box = QCheckBox('n rand infill', self)
        self.flags_grid.addWidget(self.n_rand_infill_check_box, i, 1, 1, 10)
        self.n_rand_infill_lab = QLabel("n rand infill: ")
        self.n_rand_infill_line = QLineEdit()
        self.n_rand_infill_line.setText("10")
        self.flags_grid.addWidget(self.n_rand_infill_lab, i, 10)
        self.flags_grid.addWidget(self.n_rand_infill_line, i, 11, 1, 2)
        self.n_rand_infill_lab.setToolTip(str(''))
        self.n_rand_infill_line.setToolTip(str(''))
        i += 1

        self.min_corr_check_box = QCheckBox('min corr', self)
        self.flags_grid.addWidget(self.min_corr_check_box, i, 1, 1, 10)
        self.min_corr_lab = QLabel("min corr: ")
        self.min_corr_line = QLineEdit()
        self.min_corr_line.setText("0.5")
        self.flags_grid.addWidget(self.min_corr_lab, i, 10)
        self.flags_grid.addWidget(self.min_corr_line, i, 11, 1, 2)
        self.min_corr_lab.setToolTip(str(''))
        self.min_corr_line.setToolTip(str(''))
        i += 1

        self.max_time_check_box = QCheckBox('max time lag corr', self)
        self.flags_grid.addWidget(self.max_time_check_box, i, 1, 1, 10)
        self.max_time_lag_corr_lab = QLabel("Max time lag corr: ")
        self.max_time_lag_corr_line = QLineEdit("6")
        self.flags_grid.addWidget(self.max_time_lag_corr_lab, i, 10)
        self.flags_grid.addWidget(self.max_time_lag_corr_line, i, 11, 1, 2)
        i += 1

        self.cut_cdf_thresh_check_box = QCheckBox('cut cdf thresh', self)
        self.flags_grid.addWidget(self.cut_cdf_thresh_check_box, i, 1, 1, 10)
        self.cut_cdf_thresh_lab = QLabel("cut cdf thresh: ")
        self.cut_cdf_thresh_line = QLineEdit("0.5")
        self.flags_grid.addWidget(self.cut_cdf_thresh_lab, i, 10)
        self.flags_grid.addWidget(self.cut_cdf_thresh_line, i, 11, 1, 2)
        i += 1

        # # default button ##
        self.default_btn = QPushButton('Default')
        self.flags_grid.addWidget(self.default_btn, i, 10, 1, 3)
        self.default_btn.clicked.connect(lambda: self.default_ticks())
        i += 1

        # # add space ##
        self.space_lab = QLabel(' ')
        self.space_lab.setFixedHeight(50)
        self.flags_grid.addWidget(self.space_lab, i, 0)

        i += 1

        # # runn button ##
        self.run_lab = 'Run!'
        self.run_btn = QPushButton()
        self.run_btn.setText(self.run_lab)
        self.run_btn.clicked.connect(lambda: self.run())
        self.flags_grid.addWidget(self.run_btn, i, 1, 1, 15)
        self.run_btn.setToolTip(str('Run!'))
        i += 1

        # # back button ##
        self.back_lab = 'Back'
        self.back_btn = QPushButton()
        self.back_btn.setText(self.back_lab)
        self.back_btn.clicked.connect(lambda: self.back())
        self.flags_grid.addWidget(self.back_btn, i, 1, 1, 15)
        self.back_btn.setToolTip(str('Go back to previous Tab!'))
        i += 1

        self.flags_layout.addLayout(self.flags_grid)
        self.flags_tab.setLayout(self.flags_layout)

    def ecops_state(self):
        if self.plot_ecops_check_box.isChecked() == True:
            self.ecop_bins_line.setDisabled(False)
        else:
            self.ecop_bins_line.setDisabled(True)

    def n_rand_infill_state(self):
        if self.n_rand_infill_check_box.isChecked() == True:
            self.n_rand_infill_line.setDisabled(False)
        else:
            self.n_rand_infill_line.setDisabled(True)

    def min_corr_state(self):
        if self.min_corr_check_box.isChecked() == True:
            self.min_corr_line.setDisabled(False)
        else:
            self.min_corr_line.setDisabled(True)

    def max_time_state(self):
        if self.max_time_check_box.isChecked() == True:
            self.max_time_lag_corr_line.setDisabled(False)
        else:
            self.max_time_lag_corr_line.setDisabled(True)

    def cut_cdf_thresh_state(self):
        if self.cut_cdf_thresh_check_box.isChecked() == True:
            self.cut_cdf_thresh_line.setDisabled(False)
        else:
            self.cut_cdf_thresh_line.setDisabled(True)

    def read_data(self):
        self.read_succ = True

        try:
            self.in_q_orig_file = self.in_q_orig_file_line.text()
            assert self.in_q_orig_file
            print('\u2714', "Input file is:", self.in_q_orig_file)
        except Exception as msg:
            self.show_error('Please Select a valid Input file', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading input file")
            self.read_succ = False

        try:
            self.in_coords_file = self.in_coords_file_line.text()
            assert self.in_coords_file
            print('\u2714', "Coordinates file is:", self.in_coords_file)
        except Exception as msg:
            self.show_error('Please Select a valid Coordinates file', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Main Coordinates file")
            self.read_succ = False

        try:
            self.out_dir = self.out_dir_line.text()
            assert self.out_dir
            print('\u2714', "Output directory is:", self.out_dir)
        except Exception as msg:
            self.show_error('Please Select a valid Output Directory', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Output directory")
            self.read_succ = False

        try:
            self.date_fmt_choice = ['%Y-%m-%d', '%Y-%b-%d', '%Y-%B-%d',
                                    '%y-%m-%d', '%y-%b-%d', '%y-%B-%d',
                                    '%d-%m-%Y', '%d-%b-%Y', '%d-%B-%Y',
                                    '%d-%m-%y', '%d-%b-%y', '%d-%B-%y',
                                    '%m-%d-%Y', '%b-%d-%Y', '%B-%d-%Y',
                                    '%m-%d-%y', '%b-%d-%y', '%B-%d-%y',
                                    '%Y.%m.%d', '%Y.%b.%d', '%Y.%B.%d',
                                    '%y.%m.%d', '%y.%b.%d', '%y.%B.%d',
                                    '%d.%m.%Y', '%d.%b.%Y', '%d.%B.%Y',
                                    '%d.%m.%y', '%d.%b.%y', '%d.%B.%y',
                                    '%m.%d.%Y', '%b.%d.%Y', '%B.%d.%Y',
                                    '%m.%d.%y', '%b.%d.%y', '%B.%d.%y',
                                    '%Y/%m/%d', '%Y/%b/%d', '%Y/%B/%d',
                                    '%y/%m/%d', '%y/%b/%d', '%y/%B/%d',
                                    '%d/%m/%Y', '%d/%b/%Y', '%d/%B/%Y',
                                    '%d/%m/%y', '%d/%b/%', '%d/%B/%y',
                                    '%m/%d/%Y', '%b/%d/%Y', '%B/%d/%Y',
                                    '%m/%d/%y', '%b/%d/%y', '%B/%d/%y']
            self.date_fmt = self.date_fmt_choice[self.date_fmt_box.currentIndex()]

            assert self.date_fmt
            print('\u2714', "Date Format is:", self.date_fmt_box.currentText())
        except Exception as msg:
            self.show_error('Please Select a valid Date format', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Date format")
            print(self.date_fmt)
            self.read_succ = False

        try:
            self.infill_stns = str(self.infill_stns_line.text())
            self.infill_stns_list = self.infill_stns.split(";")
            assert self.infill_stns_list
            print('\u2714', "Infill Stations:", self.infill_stns_list)
        except Exception as msg:
            self.show_error('Please Specify infill stations (values must be separated by a ";")', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading infill stations")
            self.read_succ = False

        try:
            self.drop_stns = str(self.drop_stns_line.text())
            self.drop_stns_list = self.drop_stns.split(";")
            assert self.drop_stns_list
            print('\u2714', "Stations to drop:", self.drop_stns_list)
        except:
            print('\u2714', "No stations to drop")

        try:
            self.censor_period = str(self.censor_period_line.text())
            self.censor_period_list = self.censor_period.split(";")
            assert self.censor_period_list
            print('\u2714', "Censor period:", self.censor_period_list)
        except Exception as msg:
            self.show_error('Please Specify a valid Censor Period, (values must be separated by a ";")', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Censor Period")
            self.read_succ = False

        try:
            self.infill_interval_type = self.infill_interval_type_box.currentText()
            assert self.infill_interval_type
            print('\u2714', "Infill Interval Type is:", self.infill_interval_type)
        except Exception as msg:
            self.show_error('Please Select a valid infill interval type', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Interval type")
            self.read_succ = False

        try:
            self.infill_type = self.infill_type_box.currentText()
            assert self.infill_type
            print('\u2714', "Infill Type is:", self.infill_type)
        except Exception as msg:
            self.show_error('Please Select a valid infill type', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Infill Type")
            self.read_succ = False

        try:
            self.min_valid_vals = int(self.min_valid_vals_line.text())
            assert self.min_valid_vals
            print('\u2714', "Minimum Valid Values:", self.min_valid_vals)

        except Exception as msg:
            self.show_error('Please specify an integer in the Minimum valid values box', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Minimum Valid Values")
            self.read_succ = False

        try:
            self.n_nrn_min = int(self.n_nrn_min_line.text())
            assert self.n_nrn_min
            print('\u2714', "Minimum nearest stations:", self.n_nrn_min)
        except Exception as msg:
            self.show_error('Please specify an integer in Minimum nearest stations box', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Minimum nearest station")
            self.read_succ = False

        try:
            self.n_nrn_max = int(self.n_nrn_max_line.text())
            assert self.n_nrn_max
            print('\u2714', "Maximum nearest stations:", self.n_nrn_max)
        except Exception as msg:
            self.show_error('Please specify an integer in Maximum nearest stations box', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Maximum nearest stations")
            self.read_succ = False

        try:
            self.ncpus = int(self.ncpus_box.currentText())
            assert self.ncpus
            assert isinstance(self.ncpus, int)
            print('\u2714', "Number of CPUs:", self.ncpus)
        except Exception as msg:
            self.show_error('Please Select a valid Number of CPUs', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Number of CPUs")
            self.read_succ = False

        try:
            self.sep = self.sep_box.currentText()
            assert self.sep
            print('\u2714', "Separator is:", self.sep)
        except Exception as msg:
            self.show_error('Please Select a valid Separator', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Separator")
            self.read_succ = False

        try:
            self.freq = self.freq_box.currentText()
            assert self.freq
            print('\u2714', "Freq is:", self.freq)
        except Exception as msg:
            self.show_error('Please Select a valid Freq', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error reading Freq")
            self.read_succ = False

        try:
            if self.read_succ == True:
                self.flags_tab.setDisabled(False)
                print('\n')
                print('\u2714' * 37)
                print('Data read successfully....')
                print('\u2714' * 37, '\n')
                self.conf_input_tab.setDisabled(True)
                self.setCurrentIndex(self.currentIndex() + 1)
            else:
                self.flags_tab.setDisabled(True)
                print('\n')
                print('\u2716' * 37)
                print("Error reading data")
                print('\u2716' * 37, '\n')

        except Exception as msg:
            self.show_error('Please check your data', QMessageBox.Critical, details=repr(msg))
            self.flags_tab.setDisabled(True)

        # define NormalCopulaInfill (imported script) #
        self.infill_cop = NormCopulaInfill(in_var_file=self.in_q_orig_file,
                                           out_dir=self.out_dir,
                                           infill_stns=self.infill_stns_list,
                                           min_valid_vals=self.min_valid_vals,
                                           infill_interval_type=self.infill_interval_type,
                                           infill_type=self.infill_type,
                                           infill_dates_list=self.censor_period_list,
                                           in_coords_file=self.in_coords_file,
                                           n_min_nebs=self.n_nrn_min,
                                           n_max_nebs=self.n_nrn_max,
                                           ncpus=self.ncpus,
                                           skip_stns=self.drop_stns_list,
                                           sep=self.sep,
                                           time_fmt=self.date_fmt,
                                           freq=self.freq,
                                           verbose=True,
                                           )

    def run(self):
        try:
            self.conf_input_tab.setDisabled(True)
            self.flags_tab.setDisabled(True)
            print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
            start = timeit.default_timer()

            if self.debug_check_box.isChecked() == True:
                self.infill_cop.debug_mode_flag = True
            else:
                self.infill_cop.debug_mode_flag = False

            if self.plot_diag_check_box.isChecked() == True:
                self.infill_cop.plot_diag_flag = True
            else:
                self.infill_cop.plot_diag_flag = False

            if self.plot_step_cdf_pdf_check_box.isChecked() == True:
                self.infill_cop.plot_step_cdf_pdf_flag = True
            else:
                self.infill_cop.plot_step_cdf_pdf_flag = False

            if self.compare_infill_check_box.isChecked() == True:
                self.infill_cop.compare_infill_flag = True
            else:
                self.infill_cop.compare_infill_flag = False

            if self.flag_susp_check_box.isChecked() == True:
                self.infill_cop.flag_susp_flag = True
            else:
                self.infill_cop.flag_susp_flag = False

            if self.force_infill_check_box.isChecked() == True:
                self.infill_cop.force_infill_flag = True
            else:
                self.infill_cop.force_infill_flag = False

            if self.plot_neighbors_flag_check_box.isChecked() == True:
                self.infill_cop.plot_neighbors_flag = True
            else:
                self.infill_cop.plot_neighbors_flag = False

            if self.take_min_stns_check_box.isChecked() == True:
                self.infill_cop.take_min_stns_flag = True
            else:
                self.infill_cop.take_min_stns_flag = False

            if self.overwrite_check_box.isChecked() == True:
                self.infill_cop.overwrite_flag = True
            else:
                self.infill_cop.overwrite_flag = False

            if self.read_pickles_check_box.isChecked() == True:
                self.infill_cop.read_pickles_flag = True
            else:
                self.infill_cop.read_pickles_flag = False

            if self.use_best_stns_flag_check_box.isChecked() == True:
                self.infill_cop.use_best_stns_flag = True
            else:
                self.infill_cop.use_best_stns_flag = False

            if self.dont_stop_flag_check_box.isChecked() == True:
                self.infill_cop.dont_stop_flag = True
            else:
                self.infill_cop.dont_stop_flag = False

            if self.plot_long_term_corrs_flag_check_box.isChecked() == True:
                self.infill_cop.plot_long_term_corrs_flag = True
            else:
                self.infill_cop.plot_long_term_corrs_flag = False

            if self.save_step_vars_flag_check_box.isChecked() == True:
                self.infill_cop.save_step_vars_flag = True
            else:
                self.infill_cop.save_step_vars_flag = False

            if self.plot_rand_check_box.isChecked() == True:
                self.infill_cop.plot_rand_flag = True
            else:
                self.infill_cop.plot_rand_flag = False

            if self.stn_based_mp_infill_check_box.isChecked() == True:
                self.infill_cop.stn_based_mp_infill = True
            else:
                self.infill_cop.stn_based_mp_infill = False

            if self.save_step_vars_flag_check_box.isChecked() == True:
                self.infill_cop.save_step_vars_flag = True
            else:
                self.infill_cop.save_step_vars_flag = False

            if self.infill_cop.nrst_stns_type == 'dist':
                self.infill_cop.cmpt_plot_nrst_stns()
            elif self.infill_cop.nrst_stns_type == 'rank':
                self.infill_cop.cmpt_plot_rank_corr_stns()
            # elif self.infill_cop.nrst_stns_type == 'symm':
            #     self.infill_cop.cmpt_plot_symm_stns()
            else:
                pass

            if self.n_rand_infill_check_box.isChecked() == True:
                self.infill_cop.n_rand_infill_values = int(self.n_rand_infill_line.text())
            else:
                pass

            if self.min_corr_check_box.isChecked() == True:
                self.infill_cop.min_corr = float(self.n_rand_infill_line.text())
            else:
                pass

            if self.max_time_check_box.isChecked() == True:
                self.infill_cop.max_time_lag_corr = int(self.max_time_lag_corr_line.text())
            else:
                pass

            if not self.cut_cdf_thresh_check_box.isChecked() == True:
                self.infill_cop.cut_cdf_thresh = self.cut_cdf_thresh_line.text()
            else:
                pass

            if self.save_step_vars_flag_check_box.isChecked() == True:
                self.infill_cop.nrst_stns_type = self.nrst_stns_type_box.currentText()
            else:
                pass
            if self.plot_nrst_stns_flag_check_box.isChecked() == True:
                self.infill_cop.cmpt_plot_nrst_stns()
            else:
                pass

            if self.plot_ecops_check_box.isChecked() == True:
                self.infill_cop.cop_bins = int(self.ecop_bins_line.text())
                self.infill_cop.plot_ecops()
            else:
                pass

            if self.ignore_bad_stns_flag_check_box.isChecked() == True:
                self.infill_cop.ignore_bad_stns_flag = True
            else:
                self.infill_cop.ignore_bad_stns_flag = False

            if self.cmpt_plot_stats_check_box.isChecked() == True:
                self.infill_cop.plot_stats()
            else:
                pass

            self.infill_cop.infill()

            if self.cmpt_plot_avail_stns_check_box.isChecked() == True:
                self.infill_cop.cmpt_plot_avail_stns()
            else:
                pass

            self.infill_cop.plot_summary()

            stop = timeit.default_timer()  # Ending time
            print(('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a') % (time.asctime(), stop - start))

            self.conf_input_tab.setDisabled(False)
            self.setCurrentIndex(self.currentIndex() - 1)
        except Exception as msg:
            self.show_error('Simulation Error', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error Running simulation")
        finally:
            self.conf_input_tab.setDisabled(False)
            self.setCurrentIndex(self.currentIndex() - 1)

    def back(self):
        self.setCurrentIndex(self.currentIndex() - 1)
        self.flags_tab.setDisabled(True)
        self.conf_input_tab.setDisabled(False)

    def back_tab(self):
        if self.currentIndex() == 0:
            self.flags_tab.setDisabled(True)
            self.conf_input_tab.setDisabled(False)

    def plot_step_cdf_pdf(self, state):
        if self.plot_step_cdf_pdf_check_box.isChecked() == True:
            print("plot step cdf pdf is set to: ON")
        else:
            print("plot step cdf pdf is set to: OFF")

    def plot_diag(self, state):
        if self.plot_diag_check_box.isChecked() == True:
            print("Plot diag is set to: ON")
        else:
            print("Plot diag is set to: OFF")

    def overwrite(self, state):
        if self.overwrite_check_box.isChecked() == True:
            print("Overwrite is set to: ON")
        else:
            self.infill_cop.read_pickles_flag = False
            print("Overwrite is set to: OFF")

    def read_pickles(self, state):
        if self.read_pickles_check_box.isChecked() == True:
            print("Read pickles is set to: ON")
        else:
            print("Read pickles is set to: OFF")

    def take_min_stns(self, state):
        if self.take_min_stns_check_box.isChecked() == True:
            print("Take min stations is set to: ON")
        else:
            print("Take min stations is set to: OFF")

    def force_infill(self, state):
        if self.force_infill_check_box.isChecked() == True:
            print("Force infill is set to: ON")
        else:
            print("Force infill is set to: OFF")

    def flag_susp(self, state):
        if self.flag_susp_check_box.isChecked() == True:
            print("Flag Susp is set to: ON")
        else:
            print("Flag Susp is set to: OFF")

    def debug_mode(self, state):
        if self.debug_check_box.isChecked() == True:
            print("Debug mode is set to: ON")
        else:
            print("Debug mode is set to: OFF")

    def compare_infill(self, state):
        if self.compare_infill_check_box.isChecked() == True:
            print("Compare infill is set to: ON")
        else:
            print("Compare infill is set to: OFF")

    def get_input_dir(self, _input=True):
        if _input:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.FileMode())
            if dlg.exec_():
                self.in_q_orig_file = dlg.selectedFiles()[0]
                self.in_q_orig_file_line.setText(self.in_q_orig_file)

    def get_out_dir(self, _input=True):
        if _input:
            self.out_dir = str(QFileDialog.getExistingDirectory(self))
            self.out_dir_line.setText(self.out_dir)

    def get_coords_dir(self, _input=True):
        if _input:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.FileMode())
            if dlg.exec_():
                self.in_coords_file = dlg.selectedFiles()[0]
                self.in_coords_file_line.setText(self.in_coords_file)

    def default_ticks(self):
        self.debug_check_box.setChecked(False)
        self.compare_infill_check_box.setChecked(False)
        self.flag_susp_check_box.setChecked(False)
        self.force_infill_check_box.setChecked(False)
        self.take_min_stns_check_box.setChecked(True)
        self.read_pickles_check_box.setChecked(False)
        self.overwrite_check_box.setChecked(False)
        self.plot_diag_check_box.setChecked(False)
        self.plot_step_cdf_pdf_check_box.setChecked(False)
        self.plot_neighbors_flag_check_box.setChecked(False)
        self.ignore_bad_stns_flag_check_box.setChecked(True)
        self.use_best_stns_flag_check_box.setChecked(True)
        self.dont_stop_flag_check_box.setChecked(False)
        self.plot_long_term_corrs_flag_check_box.setChecked(False)
        self.save_step_vars_flag_check_box.setChecked(False)

    def show_error(self, msg, icon, details='Whoops, No details.'):
        '''
        Show an error message box
        '''
        widget = QWidget()
        err_box = QMessageBox(widget)
        err_box.setIcon(icon)
        err_box.setText(msg)
        err_box.setWindowTitle('Error')
        err_box.setDetailedText('The details are as follow:\n' + details)
        err_box.setStandardButtons(QMessageBox.Ok)
        err_box.exec_()

    def load_file(self):
        dlg = QFileDialog()
        load_xml_file = dlg.getOpenFileName (self, 'Load config file', self.in_q_orig_file_line.text(), '*.xml')
        my_xml_file = open (str(load_xml_file[0]).replace('\\', '/'), 'r')
        self.dom = etree.parse(my_xml_file)
        self.xml_settings()

    def xml_settings(self):
        try:
            self.user_csv_path = self.dom.find('csv_path').text
            self.in_q_orig_file_line.setText(self.user_csv_path)
            if self.gui_debug == True:
                print('DEBUGGER: path loaded successfully')

            self.user_coords_path = self.dom.find('coords_path').text
            self.in_coords_file_line.setText(self.user_coords_path)
            if self.gui_debug == True:
                print('DEBUGGER: coordinates loaded successfully')

            self.user_output_path = self.dom.find('output_path').text
            self.out_dir_line.setText(self.user_output_path)
            if self.gui_debug == True:
                print('DEBUGGER: output loaded successfully')

            self.user_date_format = self.dom.find('date_format').text
            self.index = self.date_fmt_box.findText(self.user_date_format, Qt.MatchFixedString)
            self.date_fmt_box.setCurrentIndex(self.index)
            if self.gui_debug == True:
                print('DEBUGGER: date format loaded successfully')

            self.user_infill_stations = self.dom.find('infill_stations').text
            self.infill_stns_line.setText(self.user_infill_stations)
            if self.gui_debug == True:
                print('DEBUGGER: infill stations loaded successfully')

            self.user_drop_stns = self.dom.find('drop_stations').text
            self.drop_stns_line.setText(self.user_drop_stns)
            if self.gui_debug == True:
                print('DEBUGGER: drop stations loaded successfully')

            self.user_censor_period = self.dom.find('censor_period').text
            self.censor_period_line.setText(self.user_censor_period)
            if self.gui_debug == True:
                print('DEBUGGER: censor period loaded successfully')

            self.user_infill_interval_type = self.dom.find('infill_interval_type').text
            self.index1 = self.infill_interval_type_box.findText(self.user_infill_interval_type, Qt.MatchFixedString)
            self.infill_interval_type_box.setCurrentIndex(self.index1)
            if self.gui_debug == True:
                print('DEBUGGER: interval type loaded successfully')

            self.user_infill_type = self.dom.find('infill_type').text
            self.index2 = self.infill_type_box.findText(self.user_infill_type, Qt.MatchFixedString)
            self.infill_type_box.setCurrentIndex(self.index2)
            if self.gui_debug == True:
                print('DEBUGGER: infill type loaded successfully')

            self.user_min_valid_vals = self.dom.find('minimum_valid_values').text
            self.min_valid_vals_line.setText(self.user_min_valid_vals)
            if self.gui_debug == True:
                print('DEBUGGER: min valid vals loaded successfully')

            self.user_n_nrn_min = self.dom.find('minimum_nearest_stations').text
            self.n_nrn_min_line.setText(self.user_n_nrn_min)
            if self.gui_debug == True:
                print('DEBUGGER: minimum nearest stations loaded successfully')

            self.user_n_nrn_max = self.dom.find('maximum_nearest_stations').text
            self.n_nrn_max_line.setText(self.user_n_nrn_max)
            if self.gui_debug == True:
                print('DEBUGGER: max nearest loaded successfully')

            self.user_ncpus = self.dom.find('number_of_cpus').text
            self.index3 = self.ncpus_box.findText(self.user_ncpus, Qt.MatchFixedString)
            self.ncpus_box.setCurrentIndex(self.index3)
            if self.gui_debug == True:
                print('DEBUGGER: number of cpus loaded successfully')

            self.user_sep = self.dom.find('separator').text
            self.index4 = self.sep_box.findText(self.user_sep, Qt.MatchFixedString)
            self.sep_box.setCurrentIndex(self.index4)
            if self.gui_debug == True:
                print('DEBUGGER: separator loaded successfully')

            self.user_freq = self.dom.find('frequency').text
            self.index5 = self.freq_box.findText(self.user_freq, Qt.MatchFixedString)
            self.freq_box.setCurrentIndex(self.index5)
            if self.gui_debug == True:
                print('DEBUGGER: frequency loaded successfully')

            self.user_plot_step_cdf_pdf = self.dom.find('plot_step_cdf_pdf_bool').text
            if self.user_plot_step_cdf_pdf == 'checked':
                self.plot_step_cdf_pdf_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: Plot step cdf pdf check box loaded successfully")
                else:
                    pass
            else:
                self.plot_step_cdf_pdf_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: Plot step cdf pdf check box failed")
                else:
                    pass

            self.user_compare_infill = self.dom.find('compare_infill_bool').text
            if self.user_compare_infill == 'checked':
                self.compare_infill_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: compare infill check box loaded successfully")
                else:
                    pass
            else:
                self.compare_infill_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: compare infill check box failed")
                else:
                    pass

            self.user_plot_diag = self.dom.find('plot_diag_bool').text
            if self.user_plot_diag == 'checked':
                self.plot_diag_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: plot diag check box loaded successfully")
                else:
                    pass
            else:
                self.plot_diag_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: plot diag check box failed")
                else:
                    pass

            self.user_overwrite = self.dom.find('overwrite_bool').text
            if self.user_overwrite == 'checked':
                self.overwrite_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: overwrite check box loaded successfully")
                else:
                    pass
            else:
                self.overwrite_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: overwrite check box failed")
                else:
                    pass

            self.user_read_pickles = self.dom.find('read_pickles_bool').text
            if self.user_read_pickles == 'checked':
                self.read_pickles_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: read pickles check box loaded successfully")
                else:
                    pass
            else:
                self.read_pickles_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: read pickles check box failed")
                else:
                    pass

            self.user_take_min_stns = self.dom.find('take_min_stns_bool').text
            if self.user_take_min_stns == 'checked':
                self.take_min_stns_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: take min stns check box loaded successfully")
                else:
                    pass
            else:
                self.take_min_stns_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: take min stns check box failed")
                else:
                    pass

            self.user_force_infill = self.dom.find('force_infill_bool').text
            if self.user_force_infill == 'checked':
                self.force_infill_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: force infill check box loaded successfully")
                else:
                    pass
            else:
                self.force_infill_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: force infill check box failed")
                else:
                    pass

            self.user_flag_susp = self.dom.find('flag_susp_bool').text
            if self.user_flag_susp == 'checked':
                self.flag_susp_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: flag susp check box loaded successfully")
                else:
                    pass
            else:
                self.flag_susp_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: flag susp check box failed")
                else:
                    pass

            self.user_debug = self.dom.find('debug_bool').text
            if self.user_debug == 'checked':
                self.debug_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: debug check box loaded successfully")
                else:
                    pass
            else:
                self.debug_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: debug check box failed")
                else:
                    pass

            self.user_plot_neighbors_flag = self.dom.find('plot_neighbors_flag_bool').text
            if self.user_plot_neighbors_flag == 'checked':
                self.plot_neighbors_flag_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: plot neighbours check box loaded successfully")
                else:
                    pass
            else:
                self.plot_neighbors_flag_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: plot neighbours check box failed")
                else:
                    pass

            self.user_ignore_bad_stns_flag = self.dom.find('ignore_bad_stns_flag_bool').text
            if self.user_ignore_bad_stns_flag == 'checked':
                self.ignore_bad_stns_flag_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: ignore bad stns check box loaded successfully")
                else:
                    pass
            else:
                self.ignore_bad_stns_flag_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: ignore bad stns check box failed")
                else:
                    pass

            self.user_use_best_stns_flag = self.dom.find('use_best_stns_flag_bool').text
            if self.user_use_best_stns_flag == 'checked':
                self.use_best_stns_flag_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: use best stns check box loaded successfully")
                else:
                    pass
            else:
                self.use_best_stns_flag_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: use best stns check box failed")
                else:
                    pass

            self.user_dont_stop_flag = self.dom.find('dont_stop_flag_bool').text
            if self.user_dont_stop_flag == 'checked':
                self.dont_stop_flag_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: dont stop flag check box loaded successfully")
                else:
                    pass
            else:
                self.dont_stop_flag_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: dont stop flag check box failed")
                else:
                    pass

            self.user_plot_long_term_corrs_flag = self.dom.find('plot_long_term_corrs_flag_bool').text
            if self.user_plot_long_term_corrs_flag == 'checked':
                self.plot_long_term_corrs_flag_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: plot long term corrs flag check box loaded successfully")
                else:
                    pass
            else:
                self.plot_long_term_corrs_flag_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: plot long term corrs flag check box failed")
                else:
                    pass

            self.user_save_step_vars_flag = self.dom.find('save_step_vars_flag_bool').text
            if self.user_save_step_vars_flag == 'checked':
                self.save_step_vars_flag_check_box.setChecked(True)
                if self.gui_debug == True:
                    print("DEBUGGER: save step vars flag check box loaded successfully")
                else:
                    pass
            else:
                self.save_step_vars_flag_check_box.setChecked(False)
                if self.gui_debug == False:
                    print("DEBUGGER: save step vars flag check box failed")
                else:
                    pass

            print('\u2714', 'User settings applied successfully')
        except Exception as msg:
            self.show_error('Error applying user settings', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error applying user settings")

    def save_xml_dir(self, _input=True):
         if _input:
            dlg = QFileDialog()
            xml_file = dlg.getSaveFileName(self, 'Save Config file', self.in_q_orig_file_line.text(), '*.xml')
            try:
                self.save_to_xml()
                print (xml_file[0])
                self.tree.write(str(xml_file[0]).replace('\\', '/'))
                print('\u2714', 'File saved successfully')
            except Exception as msg:
                self.show_error('Error Saving XML file', QMessageBox.Critical, details=repr(msg))
                print('\u2716', "Error Saving XML file")
                traceback.print_exc()

    def save_to_xml(self):
        try:
            self.root = Element('config')
            self.tree = ElementTree(self.root)

            self.file_xml = Element('csv_path')
            self.root.append(self.file_xml)
            self.file_xml.text = str(self.in_q_orig_file_line.text())
            if self.gui_debug == True:
                print ("DEBUGGER: in_q_orig_file Saved successfully ")
            else:
                pass

            self.coords_xml = Element('coords_path')
            self.root.append(self.coords_xml)
            self.coords_xml.text = str(self.in_coords_file_line.text())
            if self.gui_debug == True:
                print ("DEBUGGER: in_coords_file Saved successfully ")
            else:
                pass

            self.output_xml = Element('output_path')
            self.root.append(self.output_xml)
            self.output_xml.text = self.out_dir_line.text()
            if self.gui_debug == True:
                print ("DEBUGGER: out_dir Saved successfully ")
            else:
                pass

            self.date_fmt_xml = Element('date_format')
            self.root.append(self.date_fmt_xml)
            self.date_fmt_xml.text = self.date_fmt_box.currentText()
            if self.gui_debug == True:
                print ("DEBUGGER: date_fmt_box Saved successfully ")
            else:
                pass

            self.infill_stns_xml = Element('infill_stations')
            self.root.append(self.infill_stns_xml)
            self.infill_stns_xml.text = self.infill_stns_line.text()
            if self.gui_debug == True:
                print ("DEBUGGER: infill_stns Saved successfully ")
            else:
                pass

            self.drop_stns_xml = Element('drop_stations')
            self.root.append(self.drop_stns_xml)
            self.drop_stns_xml.text = self.drop_stns_line.text()

            self.censor_period_xml = Element('censor_period')
            self.root.append(self.censor_period_xml)
            self.censor_period_xml.text = self.censor_period_line.text()

            self.infill_interval_type_xml = Element('infill_interval_type')
            self.root.append(self.infill_interval_type_xml)
            self.infill_interval_type_xml.text = self.infill_interval_type_box.currentText()

            self.infill_type_xml = Element('infill_type')
            self.root.append(self.infill_type_xml)
            self.infill_type_xml.text = self.infill_type_box.currentText()

            self.min_valid_vals_xml = Element('minimum_valid_values')
            self.root.append(self.min_valid_vals_xml)
            self.min_valid_vals_xml.text = self.min_valid_vals_line.text()

            self.n_nrn_min_xml = Element('minimum_nearest_stations')
            self.root.append(self.n_nrn_min_xml)
            self.n_nrn_min_xml.text = self.n_nrn_min_line.text()

            self.n_nrn_max_xml = Element('maximum_nearest_stations')
            self.root.append(self.n_nrn_max_xml)
            self.n_nrn_max_xml.text = self.n_nrn_max_line.text()

            self.ncpus_xml = Element('number_of_cpus')
            self.root.append(self.ncpus_xml)
            self.ncpus_xml.text = self.ncpus_box.currentText()

            self.sep_xml = Element('separator')
            self.root.append(self.sep_xml)
            self.sep_xml.text = self.sep_box.currentText()

            self.freq_xml = Element('frequency')
            self.root.append(self.freq_xml)
            self.freq_xml.text = self.freq_box.currentText()

            self.plot_step_cdf_pdf_xml = Element('plot_step_cdf_pdf_bool')
            self.root.append(self.plot_step_cdf_pdf_xml)
            if self.plot_step_cdf_pdf_check_box.isChecked() == True:
                self.plot_step_cdf_pdf_xml.text = 'checked'
            else:
                self.plot_step_cdf_pdf_xml.text = 'unchecked'

            self.compare_infill_xml = Element('compare_infill_bool')
            self.root.append(self.compare_infill_xml)
            if self.compare_infill_check_box.isChecked() == True:
                self.compare_infill_xml.text = 'checked'
            else:
                self.compare_infill_xml.text = 'unchecked'

            self.plot_diag_xml = Element('plot_diag_bool')
            self.root.append(self.plot_diag_xml)
            if self.plot_diag_check_box.isChecked() == True:
                self.plot_diag_xml.text = 'checked'
            else:
                self.plot_diag_xml.text = 'unchecked'

            self.overwrite_xml = Element('overwrite_bool')
            self.root.append(self.overwrite_xml)
            if self.overwrite_check_box.isChecked() == True:
                self.overwrite_xml.text = 'checked'
            else:
                self.overwrite_xml.text = 'unchecked'

            self.read_pickles_xml = Element('read_pickles_bool')
            self.root.append(self.read_pickles_xml)
            if self.read_pickles_check_box.isChecked() == True:
                self.read_pickles_xml.text = 'checked'
            else:
                self.read_pickles_xml.text = 'unchecked'

            self.take_min_stns_xml = Element('take_min_stns_bool')
            self.root.append(self.take_min_stns_xml)
            if self.take_min_stns_check_box.isChecked() == True:
                self.take_min_stns_xml.text = 'checked'
            else:
                self.take_min_stns_xml.text = 'unchecked'

            self.force_infill_xml = Element('force_infill_bool')
            self.root.append(self.force_infill_xml)
            if self.force_infill_check_box.isChecked() == True:
                self.force_infill_xml.text = 'checked'
            else:
                self.force_infill_xml.text = 'unchecked'

            self.flag_susp_xml = Element('flag_susp_bool')
            self.root.append(self.flag_susp_xml)
            if self.flag_susp_check_box.isChecked() == True:
                self.flag_susp_xml.text = 'checked'
            else:
                self.flag_susp_xml.text = 'unchecked'

            self.debug_xml = Element('debug_bool')
            self.root.append(self.debug_xml)
            if self.debug_check_box.isChecked() == True:
                self.debug_xml.text = 'checked'
            else:
                self.debug_xml.text = 'unchecked'

            self.plot_neighbors_flag_xml = Element('plot_neighbors_flag_bool')
            self.root.append(self.plot_neighbors_flag_xml)
            if self.plot_neighbors_flag_check_box.isChecked() == True:
                self.plot_neighbors_flag_xml.text = 'checked'
            else:
                self.plot_neighbors_flag_xml.text = 'unchecked'

            self.ignore_bad_stns_flag_xml = Element('ignore_bad_stns_flag_bool')
            self.root.append(self.ignore_bad_stns_flag_xml)
            if self.ignore_bad_stns_flag_check_box.isChecked() == True:
                self.ignore_bad_stns_flag_xml.text = 'checked'
            else:
                self.ignore_bad_stns_flag_xml.text = 'unchecked'

            self.use_best_stns_flag_xml = Element('use_best_stns_flag_bool')
            self.root.append(self.use_best_stns_flag_xml)
            if self.use_best_stns_flag_check_box.isChecked() == True:
                self.use_best_stns_flag_xml.text = 'checked'
            else:
                self.use_best_stns_flag_xml.text = 'unchecked'

            self.dont_stop_flag_xml = Element('dont_stop_flag_bool')
            self.root.append(self.dont_stop_flag_xml)
            if self.dont_stop_flag_check_box.isChecked() == True:
                self.dont_stop_flag_xml.text = 'checked'
            else:
                self.dont_stop_flag_xml.text = 'unchecked'

            self.plot_long_term_corrs_flag_xml = Element('plot_long_term_corrs_flag_bool')
            self.root.append(self.plot_long_term_corrs_flag_xml)
            if self.plot_long_term_corrs_flag_check_box.isChecked() == True:
                self.plot_long_term_corrs_flag_xml.text = 'checked'
            else:
                self.plot_long_term_corrs_flag_xml.text = 'unchecked'

            self.save_step_vars_flag_xml = Element('save_step_vars_flag_bool')
            self.root.append(self.save_step_vars_flag_xml)
            if self.save_step_vars_flag_check_box.isChecked() == True:
                self.save_step_vars_flag_xml.text = 'checked'
            else:
                self.save_step_vars_flag_xml.text = 'unchecked'

            print('\u2714', 'xml file compiled successfully')
        except Exception as msg:
            self.show_error('Error Compiling XML file', QMessageBox.Critical, details=repr(msg))
            print('\u2716', "Error Compiling XML file")
        return


class NormCopulaGUI(QMainWindow):
    '''
    The whole window of the GUI
    This holds the tabs and all the other stuff
    '''

    def __init__(self, parent=None):
        super(NormCopulaGUI, self).__init__(parent)

        self.window_area = QMdiArea()
        self.setCentralWidget(self.window_area)
        self.setWindowTitle('NormaCopula Plotter v0.1')

        self.msgs_wid = QTextEdit()
        self.msgs_wid.setReadOnly(True)
        sys.stdout = OutLog(self.msgs_wid, sys.stdout)
        sys.stderr = OutLog(self.msgs_wid, sys.stderr)

        self.msgs_window = QMdiSubWindow()
        self.msgs_wid.setWindowTitle('Messages')
        self.msgs_window.setWidget(self.msgs_wid)
        self.window_area.addSubWindow(self.msgs_window)
        self.msgs_window.setWindowFlags(Qt.FramelessWindowHint)
        self.msgs_window.setWindowFlags(Qt.WindowTitleHint)

        self.input_window = QMdiSubWindow()
        self.input_window.setWindowTitle('Control Panel')
        self.input_window.setWidget(MainWindow())
        self.window_area.addSubWindow(self.input_window)
        self.input_window.setWindowFlags(Qt.FramelessWindowHint)
        self.input_window.setWindowFlags(Qt.WindowTitleHint)
        self.showMaximized()
        self.input_window.show()
        self.window_area.tileSubWindows()


def main():
    app = QApplication(sys.argv)
    ex = NormCopulaGUI()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(r'P:\\',
                                    r'Synchronize',
                                    r'python_script_logs',
                                    ('%s_log_%s.log' % (
                                    os.path.basename(__file__),
                                    datetime.now().strftime('%Y%m%d%H%M%S'))))
        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea,
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
                           QSpinBox, QDoubleSpinBox, QFileDialog, QTextEdit, QComboBox,
                           QGroupBox, QRadioButton, QButtonGroup, QFrame)
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

from settings import SettingsFactory
from train import TrainingWorker
from correction_ui import UserGuidedVisualization
from engine import RegistrationWorker
from analysis import AnalysisWorker

class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.training_in_progress = False
        self.worker = None
        self.initUI()

###########

    def initUI(self):
        
        # Set window
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Logo-64x64.png')
        self.setWindowIcon(QtGui.QIcon(logo_path))
        self.setWindowTitle('Align')

        dark_stylesheet = dark_stylesheet = """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        QLabel {
            color: #bbbbbb;
        }

        QPushButton {
            background-color: #444444;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 4px;
        }

        QPushButton:hover {
            background-color: #555555;
        }

        QPushButton:pressed {
            background-color: #666666;
        }

        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
            background-color: #3b3b3b;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 4px;
        }

        QGroupBox {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 4px;
            margin-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }

        QTextEdit {
            background-color: #3b3b3b;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 4px;
        }

        QScrollBar:vertical {
            border: none;
            background: #2b2b2b;
            width: 10px;
            margin: 0px 0px 0px 0px;
        }

        QScrollBar::handle:vertical {
            background: #555555;
            border-radius: 5px;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
            height: 0;
        }

        QScrollBar:horizontal {
            border: none;
            background: #2b2b2b;
            height: 10px;
            margin: 0px 0px 0px 0px;
        }

        QScrollBar::handle:horizontal {
            background: #555555;
            border-radius: 5px;
        }

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            background: none;
            width: 0;
        }

        QFileDialog {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        QFileDialog QLabel {
            color: #bbbbbb;
        }
        """

        self.setStyleSheet(dark_stylesheet)
        self.resize(420, 800)
        self.label = QLabel("Align", self)
        self.label.move(100,100)
        
        self.show()

        # Create main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add mode selector at the top
        layout.addWidget(self.create_mode_selector())
        
        # Create all widget groups
        self.create_widget_groups()
        
        # Add shared widgets first (always visible)
        layout.addWidget(self.widgets['shared']['paths'])
        layout.addWidget(self.widgets['shared']['data_structure'])

        for widget in self.widgets['apply'].values():
            layout.addWidget(widget)

        # Add unsupervised training info
        for widget in self.widgets['autotraining'].values():
            layout.addWidget(widget)
            widget.setVisible(False)    

        # Add training settings
        layout.addWidget(self.widgets['shared']['training_settings'])
        
        for widget in self.widgets['analysis'].values():
            layout.addWidget(widget)
            widget.setVisible(False)

        # Add progress group last (always visible)
        layout.addWidget(self.widgets['shared']['progress'])
        
        # Create unified action button
        button_layout = QHBoxLayout()
        self.action_button = QPushButton("Start Training")
        self.action_button.clicked.connect(self.handle_action)
        button_layout.addWidget(self.action_button)
        layout.addLayout(button_layout)

        # Initialize channel config visibility
        self.toggle_channel_config()        
        # Initialize checkbox visibility
        self.update_timepoint_checkboxes_visibility()

        # Initialize visibility
        self.update_mode_visibility()
        self.update_processing_options_visibility()

    def create_mode_selector(self):
        container = QWidget()
        main_layout = QHBoxLayout(container)
        
        # Create a single button group for all radio buttons
        self.mode_button_group = QButtonGroup(self)
        
        # Create Training Mode group
        training_group = QGroupBox("Training Mode")
        training_layout = QVBoxLayout()
        self.training_radio = QRadioButton("Supervised")
        self.autotraining_radio = QRadioButton("Unsupervised")
        self.mode_button_group.addButton(self.training_radio)
        self.mode_button_group.addButton(self.autotraining_radio)
        self.training_radio.setToolTip("Train new registration model with user correction")
        self.training_radio.setToolTip("Train new registration model with pre-aligned image pairs")
        training_layout.addWidget(self.training_radio)
        training_layout.addWidget(self.autotraining_radio)
        training_group.setLayout(training_layout)
        # Create Analysis Mode group
        analysis_group = QGroupBox("Analysis Mode")
        analysis_layout = QVBoxLayout()
        self.training_dynamics_radio = QRadioButton("Training Dynamics")
        self.evaluate_radio = QRadioButton("Evaluate")
        self.mode_button_group.addButton(self.training_dynamics_radio)
        self.mode_button_group.addButton(self.evaluate_radio)
        self.training_dynamics_radio.setToolTip("Analyze training metadata and generate report")
        self.evaluate_radio.setToolTip("Test model performance on image dataset")
        analysis_layout.addWidget(self.training_dynamics_radio)
        analysis_layout.addWidget(self.evaluate_radio)
        analysis_group.setLayout(analysis_layout)
        # Create Alignment Mode group
        alignment_group = QGroupBox("Alignment Mode")
        alignment_layout = QVBoxLayout()
        self.manual_radio = QRadioButton("Supervised")
        self.apply_radio = QRadioButton("Unsupervised")
        self.mode_button_group.addButton(self.manual_radio)
        self.mode_button_group.addButton(self.apply_radio)
        self.manual_radio.setToolTip("Manually correct alignment without model")
        self.apply_radio.setToolTip("Apply existing model to new data")
        alignment_layout.addWidget(self.manual_radio)
        alignment_layout.addWidget(self.apply_radio)
        alignment_group.setLayout(alignment_layout)
        



        # Add groups to main layout
        main_layout.addWidget(training_group)
        main_layout.addWidget(analysis_group)
        main_layout.addWidget(alignment_group)

        # Set default selection
        self.manual_radio.setChecked(True)
        
        # Connect mode change handlers
        self.manual_radio.toggled.connect(self.update_mode_visibility)
        self.training_radio.toggled.connect(self.update_mode_visibility)
        self.autotraining_radio.toggled.connect(self.update_mode_visibility)
        self.autotraining_radio.toggled.connect(self.update_augmentation_for_unsupervised)
        self.apply_radio.toggled.connect(self.update_mode_visibility)
        self.training_dynamics_radio.toggled.connect(self.update_mode_visibility)
        self.evaluate_radio.toggled.connect(self.update_mode_visibility)

        return container

    def create_widget_groups(self):
        self.widgets = {
            'training': {},
            'autotraining': {},
            'apply': {},
            'analysis': {},
            'shared': {}}
        
        # Shared widgets
        self.widgets['shared']['paths'] = self.create_paths_group()
        self.widgets['shared']['progress'] = self.create_progress_group()
        self.widgets['shared']['data_structure'] = self.create_data_structure_group()
        self.widgets['shared']['training_settings'] = self.create_training_settings_group()

        # Apply-specific widgets
        self.widgets['apply']['model'] = self.create_model_group()
        self.widgets['apply']['processing'] = self.create_processing_group()

        # Unsupervised training specific widget
        self.widgets['autotraining']['info'] = self.create_unsupervised_info_panel()

        # Analysis-specific widgets
        self.widgets['analysis']['info'] = self.create_analysis_info_panel()

    def create_paths_group(self):
        """Create the paths selection group"""
        group = QGroupBox("Paths")
        layout = QVBoxLayout()
        
        # Input path
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        input_button = QPushButton("Browse")
        input_button.clicked.connect(lambda: self.browse_folder(self.input_path))
        input_layout.addWidget(QLabel("Input Path:"))
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(input_button)
        layout.addLayout(input_layout)
        
        # Output path
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        output_button = QPushButton("Browse")
        output_button.clicked.connect(lambda: self.browse_folder(self.output_path))
        output_layout.addWidget(QLabel("Output Path:"))
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)
        
        group.setLayout(layout)
        return group

    def create_training_settings_group(self):
        group = QGroupBox("Training Settings")
        layout = QVBoxLayout()

        # Add training mode selection
        mode_layout = QHBoxLayout()
        self.train_from_scratch_radio = QRadioButton("Train from scratch")
        self.retrain_radio = QRadioButton("Retrain existing model")
        self.train_from_scratch_radio.setChecked(True)  # Default to training from scratch
        mode_layout.addWidget(self.train_from_scratch_radio)
        mode_layout.addWidget(self.retrain_radio)
        layout.addLayout(mode_layout)
        
        # Add model path selection (initially hidden)
        model_frame = QFrame()
        model_layout = QHBoxLayout(model_frame)
        self.model_path_training = QLineEdit()
        model_button = QPushButton("Browse")
        model_button.clicked.connect(lambda: self.browse_file(self.model_path_training, file_type="model"))
        model_layout.addWidget(QLabel("Model Path:"))
        model_layout.addWidget(self.model_path_training)
        model_layout.addWidget(model_button)
        layout.addWidget(model_frame)
        
        # Connect radio buttons to show/hide model path
        self.train_from_scratch_radio.toggled.connect(lambda: model_frame.setVisible(not self.train_from_scratch_radio.isChecked()))
        model_frame.setVisible(False)  # Initially hidden since "from scratch" is default
        
        # Learning rate
        lr_layout = QHBoxLayout()
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 1)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setToolTip("Starting at 0.001 is recommended, reduce with batch size")
        lr_layout.addWidget(QLabel("Learning Rate:"))
        lr_layout.addWidget(self.learning_rate)
        layout.addLayout(lr_layout)

        # Batch size
        batch_layout = QHBoxLayout()
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1000)
        self.batch_size.setValue(32)
        batch_layout.addWidget(QLabel("Training Dataset Size:"))
        batch_layout.addWidget(self.batch_size)
        layout.addLayout(batch_layout)

        # Backward batch size
        backward_batch_layout = QHBoxLayout()
        self.backward_batch_size = QSpinBox()
        self.backward_batch_size.setRange(1, 1000)
        self.backward_batch_size.setValue(1)
        self.backward_batch_size.setToolTip("Number of samples to process before accumulating gradients (reduces memory usage)")
        backward_batch_layout.addWidget(QLabel("Backward Batch size:"))
        backward_batch_layout.addWidget(self.backward_batch_size)
        layout.addLayout(backward_batch_layout)

        # Target loss
        loss_layout = QHBoxLayout()
        self.target_loss = QDoubleSpinBox()
        self.target_loss.setRange(0.00001, 100)
        self.target_loss.setDecimals(6)
        self.target_loss.setValue(0.01)
        self.target_loss.setToolTip("Target loss value for training convergence, increase with batch size")
        loss_layout.addWidget(QLabel("Target Loss:"))
        loss_layout.addWidget(self.target_loss)
        layout.addLayout(loss_layout)

        # Data augmentation settings
        augmentation_group = QGroupBox("Data Augmentation")
        augmentation_layout = QVBoxLayout()

        # Enable/disable checkbox
        self.augment_data_checkbox = QCheckBox("Enable Data Augmentation")
        self.augment_data_checkbox.setToolTip("Generate additional training pairs through random transformations. Required for unsupervised training.")
        augmentation_layout.addWidget(self.augment_data_checkbox)

        # Create a container for augmentation settings
        self.augmentation_controls_container = QWidget()
        self.augmentation_controls_container.setObjectName("augmentation_controls_container")
        controls_layout = QVBoxLayout(self.augmentation_controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Augmentation factor
        factor_layout = QHBoxLayout()
        self.augmentation_factor = QSpinBox()
        self.augmentation_factor.setRange(1, 50)
        self.augmentation_factor.setValue(5)
        self.augmentation_factor.setToolTip("Number of augmented samples per original sample")
        factor_layout.addWidget(QLabel("Augmentation Factor:"))
        factor_layout.addWidget(self.augmentation_factor)
        controls_layout.addLayout(factor_layout)

        # Max rotation angle
        rotation_layout = QHBoxLayout()
        self.max_rotation_angle = QSpinBox()
        self.max_rotation_angle.setRange(1, 45)
        self.max_rotation_angle.setValue(8)
        self.max_rotation_angle.setToolTip("Maximum rotation angle in degrees (1-45)")
        rotation_layout.addWidget(QLabel("Max Rotation (degrees):"))
        rotation_layout.addWidget(self.max_rotation_angle)
        controls_layout.addLayout(rotation_layout)

        # Max translation
        translation_layout = QHBoxLayout()
        self.max_translation_factor = QDoubleSpinBox()
        self.max_translation_factor.setRange(0.001, 0.15)
        self.max_translation_factor.setValue(0.04)
        self.max_translation_factor.setDecimals(3)
        self.max_translation_factor.setSingleStep(0.005)
        self.max_translation_factor.setToolTip("Maximum translation as fraction of dimension size (0-0.15)")
        translation_layout.addWidget(QLabel("Max Translation Factor:"))
        translation_layout.addWidget(self.max_translation_factor)
        controls_layout.addLayout(translation_layout)

        # Add the augmentation controls container to the augmentation group
        augmentation_layout.addWidget(self.augmentation_controls_container)
        augmentation_group.setLayout(augmentation_layout)
        layout.addWidget(augmentation_group)

        # Connect checkbox and initialize visibility
        self.augment_data_checkbox.toggled.connect(self.toggle_augmentation_settings)
        self.augmentation_controls_container.setVisible(self.augment_data_checkbox.isChecked())

        group.setLayout(layout)
        return group

    def create_progress_group(self):
        """Create the progress display group"""
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        font = self.progress_text.font()
        font.setFamily("Courier")
        self.progress_text.setFont(font)
        layout.addWidget(self.progress_text)
        
        group.setLayout(layout)
        return group

    def create_model_group(self):
        """Create the model selection group"""
        group = QGroupBox("Model Path")
        layout = QHBoxLayout()

        self.model_path = QLineEdit()
        model_button = QPushButton("Browse")
        model_button.clicked.connect(lambda: self.browse_file(self.model_path, file_type="model"))
        
        layout.addWidget(QLabel("Model Path:"))
        layout.addWidget(self.model_path)
        layout.addWidget(model_button)
        
        group.setLayout(layout)
        return group

    def create_data_structure_group(self):
        """Create the data structure configuration group with a horizontal pattern layout"""
        group = QGroupBox("Data Structure")
        layout = QVBoxLayout()

        # File Pattern Section with horizontal layout
        pattern_group = QGroupBox("File Naming Pattern")
        pattern_layout = QVBoxLayout()

        help_label = QLabel("Enter prefixes used in filenames. Leave empty if not applicable.")
        help_label.setWordWrap(True)
        pattern_layout.addWidget(help_label)

        # Horizontal layout for pattern fields
        pattern_fields = QHBoxLayout()

        # Create each prefix field with its label in a vertical arrangement
        for prefix_name, placeholder in [
            ("Sample Prefix", "e.g., SPM"),
            ("Time Prefix", "e.g., TM"),
            ("Channel Prefix", "e.g., CHN")
        ]:
            field_container = QVBoxLayout()
            field_container.addWidget(QLabel(prefix_name))
            
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            if prefix_name == "Sample Prefix":
                self.sample_prefix = line_edit
            elif prefix_name == "Time Prefix":
                self.time_prefix = line_edit
            else:
                self.channel_prefix = line_edit
                
            field_container.addWidget(line_edit)
            pattern_fields.addLayout(field_container)

        pattern_layout.addLayout(pattern_fields)
        self.channel_prefix.textChanged.connect(self.toggle_channel_config)
        self.time_prefix.textChanged.connect(self.update_timepoint_checkboxes_visibility)
        example_label = QLabel("Example: For 'SPM01_TM001_CHN02.tif', enter SPM, TM, CHN")
        example_label.setWordWrap(True)
        pattern_layout.addWidget(example_label)

        pattern_group.setLayout(pattern_layout)
        layout.addWidget(pattern_group)


        # Channel Configuration Section
        self.channel_group = QGroupBox("Channel Configuration")
        channel_layout = QVBoxLayout()

        ref_channel_layout = QHBoxLayout()
        self.ref_channel_spin = QSpinBox()
        self.ref_channel_spin.setRange(0, 10)
        self.ref_channel_spin.setValue(1) 
        self.ref_channel_spin.setToolTip("Select the channel to use as reference for registration")
        ref_channel_layout.addWidget(QLabel("Reference Channel:"))
        ref_channel_layout.addWidget(self.ref_channel_spin)
        channel_layout.addLayout(ref_channel_layout)

        self.channel_group.setLayout(channel_layout)
        layout.addWidget(self.channel_group)

        group.setLayout(layout)
        return group

    def create_processing_group(self):
        group = QGroupBox("Processing Mode")
        main_layout = QVBoxLayout(group)

        # Create a separate button group for processing modes
        self.processing_button_group = QButtonGroup(self)

        # Create horizontal layout for categories
        categories_layout = QHBoxLayout()

        # Sample Processing Category
        sample_category = QGroupBox("Sample Processing")
        sample_layout = QVBoxLayout()

        # Sample Reference Alignment radio button
        self.batch_radio = QRadioButton("Sample Reference Alignment")
        self.batch_radio.setToolTip("Align all samples to a reference using a single-channel TIFF z-stack")
        sample_layout.addWidget(self.batch_radio)

        # Channel Alignment radio button
        self.channel_align_radio = QRadioButton("Channel Alignment")
        self.channel_align_radio.setToolTip("Align channels within each sample to reference channel")
        sample_layout.addWidget(self.channel_align_radio)

        # Add single "Apply to all timepoints" checkbox below radio buttons
        self.apply_all_timepoints = QCheckBox("Apply to all timepoints")
        self.apply_all_timepoints.setToolTip("Apply transformations to all timepoints")
        self.apply_all_timepoints.setVisible(False)  # Initially hidden, will be shown when time prefix exists
        self.apply_all_timepoints.toggled.connect(self.update_ref_timepoint_visibility)
        sample_layout.addWidget(self.apply_all_timepoints)

        # Set initial visibility for checkboxes
        self.apply_all_timepoints.setVisible(False)

        # Add to processing button group
        self.processing_button_group.addButton(self.batch_radio)
        self.processing_button_group.addButton(self.channel_align_radio)

        sample_category.setLayout(sample_layout)

        # Timelapse Processing Category
        timelapse_category = QGroupBox("Timelapse Processing")
        timelapse_layout = QVBoxLayout()

        self.drift_radio = QRadioButton("Drift Correction")
        self.drift_radio.setToolTip("Each timepoint will be registered to the previous timepoint")

        self.jitter_radio = QRadioButton("Jitter Correction")
        self.jitter_radio.setToolTip("Register to averaged reference window around a selected timepoint")

        # Add to processing button group
        self.processing_button_group.addButton(self.drift_radio)
        self.processing_button_group.addButton(self.jitter_radio)

        timelapse_layout.addWidget(self.drift_radio)
        timelapse_layout.addWidget(self.jitter_radio)
        timelapse_category.setLayout(timelapse_layout)

        # Add categories side by side
        categories_layout.addWidget(sample_category)
        categories_layout.addWidget(timelapse_category)
        main_layout.addLayout(categories_layout)

        # Create option widgets
        self.create_mode_options()
        
        # Create containers for options
        options_container = QWidget()
        options_layout = QVBoxLayout(options_container)
        options_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        options_layout.setSpacing(0)

        # Add all option widgets to container
        options_layout.addWidget(self.batch_options)
        options_layout.addWidget(self.channel_align_options)
        options_layout.addWidget(self.drift_options)
        options_layout.addWidget(self.jitter_options)
        options_layout.addWidget(self.global_ref_options)
        options_layout.addWidget(self.timelapse_channel_options)
        
        # Add options container to main layout
        main_layout.addWidget(options_container)

        # Connect mode change handler
        self.processing_button_group.buttonClicked.connect(self.on_processing_mode_changed)
        
        # Set initial visibility
        self.update_processing_options_visibility()
        
        group.setLayout(main_layout)
        return group
    
    def create_mode_options(self):
        """Create option widgets for each processing mode"""
        # Create shared reference selection
        self.reference_selection = self.create_reference_selection()
        
        # Batch options
        self.batch_options = QWidget()
        batch_layout = QVBoxLayout(self.batch_options)
        batch_layout.addWidget(self.reference_selection)

        # Channel alignment options
        self.channel_align_options = QWidget()
        channel_layout = QVBoxLayout(self.channel_align_options)
        channel_layout.setContentsMargins(0, 0, 0, 0)

        # Drift correction options
        self.drift_options = QWidget()
        drift_layout = QVBoxLayout(self.drift_options)
        drift_layout.setContentsMargins(0, 0, 0, 0)

        # Jitter correction options
        self.jitter_options = QWidget()
        jitter_layout = QVBoxLayout(self.jitter_options)

        window_layout = QHBoxLayout()
        self.window_size = QSpinBox()
        self.window_size.setRange(3, 21)
        self.window_size.setValue(5)
        self.window_size.setSingleStep(2)
        window_layout.addWidget(QLabel("Window Size:"))
        window_layout.addWidget(self.window_size)
        jitter_layout.addLayout(window_layout)

        ref_layout = QHBoxLayout()
        self.ref_timepoint = QSpinBox()
        self.ref_timepoint.setRange(0, 9999)
        ref_layout.addWidget(QLabel("Reference Timepoint:"))
        ref_layout.addWidget(self.ref_timepoint)
        jitter_layout.addLayout(ref_layout)

        # Reference timepoint selection (to be used when applying to all timepoints)
        self.ref_timepoint_selection = QComboBox()
        self.ref_timepoint_selection.addItems(["First Timepoint", "Last Timepoint"])
        
        # Create empty containers for compatibility but we won't use them
        self.global_ref_options = QWidget()
        self.timelapse_channel_options = QWidget()

    def create_reference_selection(self):
        """Create a reusable reference file selection widget"""
        reference_widget = QWidget()
        layout = QVBoxLayout(reference_widget)
        
        # Create file selection
        file_layout = QHBoxLayout()
        self.reference_path = QLineEdit()
        self.reference_path.setPlaceholderText("Select reference TIFF file")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: self.browse_file(self.reference_path, file_type="tif"))

        file_layout.addWidget(QLabel("Reference File:"))
        file_layout.addWidget(self.reference_path)
        file_layout.addWidget(browse_button)
        layout.addLayout(file_layout)
        
        return reference_widget

    def create_unsupervised_info_panel(self):
        """Create an information panel explaining the data structure for unsupervised training"""
        group = QGroupBox("Unsupervised Training Data Structure")
        layout = QVBoxLayout()
        
        info_label = QLabel()
        info_label.setWordWrap(True)  # Enable text wrapping
        
        # Set the information text about expected data structure
        info_message = """
        <p>The input directory should contain pre-aligned image pairs organized as follows:</p>
            <li><b>Sample IDs:</b> Each sample ID represents a distinct image pair</li>
            <li><b>Channel 0:</b> Reference images</li>
            <li><b>Channel 1:</b> Pre-aligned moving images</li>
        """
        
        info_label.setText(info_message)
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group

    def create_analysis_info_panel(self):
        """Create an information panel for analysis mode"""
        group = QGroupBox("Analysis Information")
        layout = QVBoxLayout()
        
        # Training dynamics info
        self.training_dynamics_info = QLabel(
            "Generate a PDF report showing training dynamics and feature visualization.\n"
            "Required inputs:\n"
            " - Model path: Select trained model file (.pth)\n"
            " - Output path: Where to save the PDF report\n"
            " - Input path: Directory with sample images (for feature visualization)\n"
            " - Processing settings: Same settings used during model training"
        )
        self.training_dynamics_info.setWordWrap(True)
        layout.addWidget(self.training_dynamics_info)
        
        # Evaluation info
        self.evaluate_info = QLabel(
            "Evaluate model performance on a test dataset.\n"
            "Required inputs:\n"
            " - Model path: Select trained model file (.pth)\n"
            " - Output path: Where to save the PDF report\n"
            " - Input path: Directory containing test images\n"
            " - Processing settings: Select how to organize test files"
        )
        self.evaluate_info.setWordWrap(True)
        layout.addWidget(self.evaluate_info)
        
        group.setLayout(layout)
        return group

###########

    def collect_ui_settings(self) -> dict:
        settings = {
            # Base settings
            'input_path': self.input_path.text(),
            'output_path': self.output_path.text(),
            'reference_channel': self.ref_channel_spin.value(),
            
            # Worker type settings
            'training_radio': self.training_radio.isChecked(),
            'autotraining_radio': self.autotraining_radio.isChecked(),
            'apply_radio': self.apply_radio.isChecked(),
            
            # Training specific settings
            'learning_rate': self.learning_rate.value(),
            'target_loss': self.target_loss.value(),
            'batch_size': self.batch_size.value(),
            'backward_batch_size': self.backward_batch_size.value(),
            'train_from_scratch': self.train_from_scratch_radio.isChecked(),
            'model_path': self.model_path_training.text() if not self.train_from_scratch_radio.isChecked() else None,

            # Analysis mode settings
            'training_dynamics_radio': self.training_dynamics_radio.isChecked(),
            'evaluate_radio': self.evaluate_radio.isChecked(),

            # Augmentation settings
            'augment_data': self.augment_data_checkbox.isChecked() if hasattr(self, 'augment_data_checkbox') else False,
            'augmentation_factor': self.augmentation_factor.value() if hasattr(self, 'augmentation_factor') else 5,
            'max_rotation_angle': self.max_rotation_angle.value() if hasattr(self, 'max_rotation_angle') else 15.0,
            'max_translation_factor': self.max_translation_factor.value() if hasattr(self, 'max_translation_factor') else 0.1,

            # File structure settings
            'sample_prefix': self.sample_prefix.text().strip(),
            'time_prefix': self.time_prefix.text().strip(),
            'channel_prefix': self.channel_prefix.text().strip(),

            # Processing mode settings
            'batch_radio': self.batch_radio.isChecked(),
            'channel_align_radio': self.channel_align_radio.isChecked(),
            'drift_radio': self.drift_radio.isChecked(),
            'jitter_radio': self.jitter_radio.isChecked(),
            'global_ref_radio': self.batch_radio.isChecked() and self.apply_all_timepoints.isChecked(),
            't_channel_align_radio': self.channel_align_radio.isChecked() and self.apply_all_timepoints.isChecked()
        }
        
        # Add model path for apply and analysis modes
        if self.apply_radio.isChecked() or self.training_dynamics_radio.isChecked() or self.evaluate_radio.isChecked():
            settings['model_path'] = self.model_path.text()

        # Add mode-specific settings
        if self.batch_radio.isChecked():
            settings['reference_file'] = self.reference_path.text()
            
        if self.batch_radio.isChecked() and self.apply_all_timepoints.isChecked():
            settings['reference_timepoint'] = 'First' if self.ref_timepoint_selection.currentText() == "First Timepoint" else 'Last'
            
        if self.jitter_radio.isChecked():
            settings['window_size'] = self.window_size.value()
            settings['ref_timepoint'] = self.ref_timepoint.value()
        
        settings['is_timelapse'] = (
            self.drift_radio.isChecked() or 
            self.jitter_radio.isChecked() or 
            (self.apply_all_timepoints.isChecked() and (self.batch_radio.isChecked() or self.channel_align_radio.isChecked()))
        )

        settings['is_unsupervised'] = self.autotraining_radio.isChecked()
        
        return settings

    def handle_worker_finished(self):
        """Handle worker completion"""
        self.action_button.setEnabled(True)
        self.worker = None
        self.progress_text.append("Processing completed")

############

    def toggle_channel_config(self):
        """Show or hide channel configuration based on whether channel prefix is entered"""
        has_channel_prefix = bool(self.channel_prefix.text().strip())
        self.channel_group.setVisible(has_channel_prefix)
        
        # If no channel prefix, set reference channel to 1
        if not has_channel_prefix:
            self.ref_channel_spin.setValue(1)

    def toggle_augmentation_settings(self, enabled):
        """Show or hide augmentation settings based on checkbox state"""
        if hasattr(self, 'augmentation_controls_container'):
            self.augmentation_controls_container.setVisible(enabled)

    def on_processing_mode_changed(self, button):
        """Handle processing mode selection changes"""
        self.update_processing_options_visibility()

    def update_mode_visibility(self):
        """Update widget visibility based on selected mode"""
        is_manual_training = self.training_radio.isChecked()
        is_auto_training = self.autotraining_radio.isChecked()
        is_apply = self.apply_radio.isChecked()
        is_manual = self.manual_radio.isChecked()
        is_training_dynamics = self.training_dynamics_radio.isChecked()
        is_evaluate = self.evaluate_radio.isChecked()
        
        # Ensure a processing mode is selected
        if not any([self.batch_radio.isChecked(), 
                    self.channel_align_radio.isChecked(),
                    self.drift_radio.isChecked(), 
                    self.jitter_radio.isChecked()]):
            self.batch_radio.setChecked(True)

        # Show/hide shared widgets
        self.widgets['shared']['paths'].setVisible(True)  # Always show paths widgets
        self.widgets['shared']['training_settings'].setVisible(is_manual_training or is_auto_training)
        self.widgets['shared']['data_structure'].setVisible(True)  # Always show data structure
        
        # Control model selection widget visibility
        self.widgets['apply']['model'].setVisible(is_apply or is_training_dynamics or is_evaluate)
        
        # Processing group is now visible for analysis modes as well
        self.widgets['apply']['processing'].setVisible(is_apply or is_manual or is_manual_training or is_training_dynamics or is_evaluate)
        
        # Show/hide analysis widgets
        for name, widget in self.widgets['analysis'].items():
            # Don't show the reference container
            if name == 'reference':
                widget.setVisible(False)
            else:
                widget.setVisible(is_training_dynamics or is_evaluate)

        # Update autotraining widgets visibility
        for name, widget in self.widgets['autotraining'].items():
            widget.setVisible(is_auto_training)

        # Control analysis widgets
        if 'info' in self.widgets['analysis']:
            self.widgets['analysis']['info'].setVisible(is_training_dynamics or is_evaluate)
            # Show appropriate info panel
            if hasattr(self, 'training_dynamics_info'):
                self.training_dynamics_info.setVisible(is_training_dynamics)
            if hasattr(self, 'evaluate_info'):
                self.evaluate_info.setVisible(is_evaluate)

        # Update action button text
        button_text = {
            'training': "Start Supervised Training",
            'autotraining': "Start Unsupervised Training",
            'apply': "Apply Model",
            'manual': "Start Supervised Alignment",
            'training_dynamics': "Generate Training Report",
            'evaluate': "Evaluate Model"
        }
        mode = next(k for k, v in {
            'training': is_manual_training,
            'autotraining': is_auto_training,
            'apply': is_apply,
            'manual': is_manual,
            'training_dynamics': is_training_dynamics,
            'evaluate': is_evaluate
        }.items() if v)
        self.action_button.setText(button_text[mode])

    def update_timepoint_checkboxes_visibility(self):
        """Show or hide 'Apply to all timepoints' checkbox based on whether time prefix is entered"""
        has_time_prefix = bool(self.time_prefix.text().strip())
        
        # Show/hide checkbox based on time prefix presence
        if hasattr(self, 'apply_all_timepoints'):
            self.apply_all_timepoints.setVisible(has_time_prefix)
        
        # Update reference timepoint container visibility as well
        if hasattr(self, 'ref_timepoint_container'):
            self.ref_timepoint_container.setVisible(has_time_prefix and 
                                                self.apply_all_timepoints.isChecked() and 
                                                self.batch_radio.isChecked())

    def update_processing_options_visibility(self):
        """Update visibility of processing mode option widgets and handle reference selection parenting."""
        # Store the reference selection's current parent
        current_parent = self.reference_selection.parent()
        
        # Hide all option widgets initially
        self.batch_options.setVisible(False)
        self.channel_align_options.setVisible(False)
        self.drift_options.setVisible(False)
        self.jitter_options.setVisible(False)

        # Determine the target parent for the reference file selection widget.
        if self.batch_radio.isChecked():
            ref_target = self.batch_options
            self.batch_options.setVisible(True)
        else:
            ref_target = None

        # Update visibility for the other processing mode option containers.
        self.channel_align_options.setVisible(self.channel_align_radio.isChecked())
        self.drift_options.setVisible(self.drift_radio.isChecked())
        self.jitter_options.setVisible(self.jitter_radio.isChecked())

        # Handle reparenting of the reference selection widget.
        current_parent = self.reference_selection.parent()
        if ref_target is not None:
            if current_parent != ref_target:
                if current_parent is not None:
                    current_parent.layout().removeWidget(self.reference_selection)
                self.reference_selection.setParent(ref_target)
                if ref_target == self.global_ref_options:
                    # In Global Reference mode, add below timepoint selection.
                    ref_target.layout().addWidget(self.reference_selection)
                else:
                    ref_target.layout().insertWidget(1, self.reference_selection)
            self.reference_selection.setVisible(True)
        else:
            self.reference_selection.setVisible(False)

    def update_ref_timepoint_visibility(self, checked):
        """Show/hide reference timepoint selection based on 'Apply to all timepoints' checkbox state"""
        if hasattr(self, 'ref_timepoint_selection'):
            # Create a container for the selection if it doesn't exist
            if not hasattr(self, 'ref_timepoint_container'):
                self.ref_timepoint_container = QWidget(self.batch_options)
                timepoint_layout = QHBoxLayout(self.ref_timepoint_container)
                timepoint_layout.addWidget(QLabel("Reference Timepoint:"))
                timepoint_layout.addWidget(self.ref_timepoint_selection)
                self.batch_options.layout().addWidget(self.ref_timepoint_container)
            
            # Show or hide based on checkbox state and batch radio selection
            self.ref_timepoint_container.setVisible(checked and self.batch_radio.isChecked())

    def update_augmentation_for_unsupervised(self, checked):
        """Automatically enable and lock data augmentation when unsupervised training is selected"""
        if hasattr(self, 'augment_data_checkbox'):
            if checked:
                # When unsupervised training is selected, enable augmentation and disable the checkbox
                self.augment_data_checkbox.setChecked(True)
                self.augment_data_checkbox.setEnabled(False)
                # Make sure augmentation controls are visible
                self.augmentation_controls_container.setVisible(True)
            else:
                # When not in unsupervised mode, re-enable the checkbox
                self.augment_data_checkbox.setEnabled(True)

    def handle_action(self):
        """Handle button click based on current mode"""
        try:
            # Collect settings from UI
            ui_settings = self.collect_ui_settings()
            
            # Create appropriate settings instance
            settings = SettingsFactory.create_settings(ui_settings)
            
            # Create and start appropriate worker
            if self.training_radio.isChecked() or self.autotraining_radio.isChecked():
                self.worker = TrainingWorker(settings, QApplication.instance())
                self.worker.visualization_needed_signal.connect(self.show_visualization)
            elif self.training_dynamics_radio.isChecked() or self.evaluate_radio.isChecked():
                self.worker = AnalysisWorker(settings)
            else: 
                self.worker = RegistrationWorker(settings)
                self.worker.visualization_needed_signal.connect(self.show_visualization)
        
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.finished_signal.connect(self.handle_worker_finished)
            self.action_button.setEnabled(False)
            self.worker.start()
            
        except ValueError as e:
            self.progress_text.append(f"Error: {str(e)}")

    def browse_file(self, line_edit, file_type="model"):
        if file_type == "model":
            file_filter = "Model Files (*.pth *.pt)"
        elif file_type == "tiff":
            file_filter = "TIFF Stack Files (*.tif *.tiff)"
        else:
            file_filter = "All Files (*)"

        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            file_filter)
        if file:
            line_edit.setText(file)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select Folder",
            "")
        if folder:
            line_edit.setText(folder)

    def update_progress(self, message):
        # Append message and scroll to bottom
        self.progress_text.append(message.rstrip()) 
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum())

    def show_visualization(self, viz_data):
        """Handle visualization request from worker thread"""
        # Check what worker is currently active
        if isinstance(self.worker, RegistrationWorker):
            mode = 'correction'
        else:
            mode = viz_data.mode

        visualizer = UserGuidedVisualization(
            moving_img=viz_data.moving,
            reference_img=viz_data.reference,
            reference_file_path=str(viz_data.file_pair[1]),
            predicted_transform=viz_data.predicted_transform,
            app=QApplication.instance(),
            mode=mode
        )
        
        returned_value = visualizer.show()
        
        if mode == 'validate':
            was_validated = returned_value  # boolean
            self.worker.visualization_result = was_validated

        else:
            transform = returned_value
            self.worker.visualization_result = transform
        
        # Finally, let the worker proceed
        self.worker.visualization_event.set()
    
    def training_finished(self):
        self.action_button.setEnabled(True)
        self.training_in_progress = False
        if self.worker:
            self.worker.stop()
            self.worker = None
            
    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
import torch
import os
import sys
import traceback
import multiprocessing as mp
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit,
    QPushButton, QLabel, QComboBox, QApplication, QSlider,QGridLayout, QFrame, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
from engine import get_metadata
from registration_net import RegistrationNet

class UserGuidedVisualization(QMainWindow):
    def __init__(self, moving_img, reference_img, reference_file_path, predicted_transform, app=None, mode='correction'):

        if predicted_transform is None:
            raise ValueError("predicted_transform is required")

        # Initialize QApplication if needed
        self.app = app if app else QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        super().__init__()

        self.mode = mode  # Save the mode, which can be 'validate' or 'correction'
        
        self.correction_ready = False
        self.was_validated = None
        self.manual_mode = True if mode == 'correction' else False

        # Get reference image spacing once
        pixel_size_x, pixel_size_y, pixel_size_z, *_ = get_metadata(reference_file_path)
        self.spacing = (pixel_size_x, pixel_size_y, pixel_size_z)

        # Store original raw images - handle both numpy arrays and tensors
        if torch.is_tensor(moving_img):
            self.raw_moving_img = moving_img.clone()  # For tensors use clone()
            self.moving_img = moving_img.clone()
            if moving_img.dim() == 4 and moving_img.size(0) == 1:
                self.raw_moving_img = self.raw_moving_img[0]
                self.moving_img = self.moving_img[0]
            self.raw_moving_img = self.raw_moving_img.cpu().numpy()
            self.moving_img = self.moving_img.cpu().numpy()
        else:
            self.raw_moving_img = moving_img.copy()  # For numpy arrays use copy()
            self.moving_img = moving_img.copy()
            
        if torch.is_tensor(reference_img):
            self.reference_img = reference_img.clone()
            if reference_img.dim() == 4 and reference_img.size(0) == 1:
                self.reference_img = self.reference_img[0]
            self.reference_img = self.reference_img.cpu().numpy()
        else:
            self.reference_img = reference_img.copy()

        # Get image dimensions
        self.depth, self.height, self.width = self.reference_img.shape

        # Initialize state variables
        self.current_slice = {'xy': self.depth // 2, 'yz': self.width // 2, 'xz': self.height // 2}
        self.current_plane = 'xy'

        # Initialize tracking variables
        self.was_validated = False
        self.corrected_transform = None

        # VTK components
        self.renderer = None
        self.moving_actor = None
        self.reference_actor = None

        # Initialize current_transform
        self.current_transform = None

        # Apply initial transform
        self.apply_transform(predicted_transform)

        # Add new attributes for manual alignment
        self.manual_mode = False  # False = prediction-based, True = raw alignment
        self.manual_params = {
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, 0.0, 0.0]
        }  # Track cumulative adjustments
        self.step_size = {'rotation': 0.1, 'translation': 1.0}  # Adjustment step sizes
        
        self.model = None
        self.model_path = None

        # Initialize UI
        self.setup_ui()

        if self.mode == 'correction':
            self.manual_controls.setVisible(True)
            self.initialize_alignment_mode()
        else:
            self.manual_controls.setVisible(False)

        # Initial display update
        self.update_display()

####### Setup UI

    def setup_ui(self):
        """Setup the PyQt UI"""
        # Set dark theme stylesheet
        dark_stylesheet = """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #444444;
            }
            QStatusBar {
                color: #ffffff;
            }
            
            QStatusBar::item {
                border: none;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #3b3b3b;
                border: 1px solid #555555;
                color: #ffffff;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #484848;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QComboBox {
                background-color: #3b3b3b;
                border: 1px solid #555555;
                color: #ffffff;
                padding: 5px;
            }
            QComboBox:hover {
                background-color: #484848;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                background: #3b3b3b;
                height: 8px;
            }
            QSlider::handle:horizontal {
                background: #5c5c5c;
                border: 1px solid #777777;
                width: 18px;
                margin: -2px 0;
            }
            QSlider::groove:vertical {
                border: 1px solid #555555;
                background: #3b3b3b;
                width: 8px;
            }
            QSlider::handle:vertical {
                background: #5c5c5c;
                border: 1px solid #777777;
                height: 18px;
                margin: 0 -2px;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444444;
            }
            QTreeWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444444;
            }
            QHeaderView::section {
                background-color: #3b3b3b;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555555;
            }
            
            QTreeWidget::item {
                color: #ffffff;
            }
            
            QTreeWidget QHeaderView {
                background-color: #3b3b3b;
                color: #ffffff;
            }
            QTreeWidget::item:selected {
                background-color: #404040;
            }
        """
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Logo-64x64.png')
        self.setWindowIcon(QtGui.QIcon(logo_path))
        self.setWindowTitle('Align Correction UI')
        self.setGeometry(250, 150, 1200, 800)
        self.label = QLabel("Align", self)
        self.label.move(100,100)
        self.setStyleSheet(dark_stylesheet)  # Apply the dark theme

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create left panel for image display and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Create image display
        self.setup_vtk_widget(left_layout)

        # Create control panel
        control_panel = self.create_control_panel()
        left_layout.addWidget(control_panel)

        # Add left panel to main layout
        main_layout.addWidget(left_panel, stretch=7)

        # Create right panel for manual input and transform info
        right_panel = self.create_info_panel()
        main_layout.addWidget(right_panel, stretch=3)

        # Add status bar
        self.statusBar().showMessage('Ready for registration correction')

    def create_control_panel(self):
        """Create the control panel with improved layout"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Panel | QFrame.Raised)
        panel.setMaximumHeight(120)
        layout = QGridLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Create left section for slice navigation
        nav_group = QFrame()
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add plane selection with better labeling
        nav_layout.addWidget(QLabel('View Plane:'))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(['XY Plane (Dorsal)', 'YZ Plane (Transverse)', 'XZ Plane (Sagittal)'])
        self.plane_combo.setToolTip("Select viewing plane\nUse arrow keys to navigate slices")
        self.plane_combo.currentIndexChanged.connect(self.update_plane_and_slider)
        nav_layout.addWidget(self.plane_combo)
        
        # Add slice navigation with better layout
        slice_frame = QFrame()
        slice_layout = QHBoxLayout(slice_frame)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        
        self.slice_label = QLabel('Slice: 0')
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.on_slice_change)
        self.slice_slider.setRange(0, self.reference_img.shape[0]-1)
        
        slice_layout.addWidget(QLabel('Slice:'))
        slice_layout.addWidget(self.slice_slider, stretch=1)
        slice_layout.addWidget(self.slice_label)
        
        nav_layout.addWidget(slice_frame)

        # Create center section for brightness/contrast controls
        brightness_group = self.create_brightness_contrast_controls()

        # Create right section for action buttons
        button_group = QFrame()
        button_layout = QVBoxLayout(button_group)

        if self.mode == 'validate':
            # ============== VALIDATE MODE ==============
            validate_btn = QPushButton('Validate (V)')
            validate_btn.setToolTip("Check alignment and close.\nNo correction transform is returned.")
            validate_btn.clicked.connect(self.validate_registration)
            
            reset_btn = QPushButton('Reset View (R)')
            reset_btn.clicked.connect(self.reset_view)

            reset_lut_btn = QPushButton('Reset LUT')
            reset_lut_btn.setToolTip("Reset brightness/contrast to defaults")
            reset_lut_btn.clicked.connect(self.reset_lut)
            
            for btn in [validate_btn, reset_btn, reset_lut_btn]:
                btn.setMinimumWidth(150)
                button_layout.addWidget(btn)

        else:
            # ============= CORRECTION MODE =============
         
            apply_correction_btn = QPushButton('Confirm Correction (A)')
            apply_correction_btn.clicked.connect(self.on_generate_correction_clicked)
            
            reset_btn = QPushButton('Reset View (R)')
            reset_btn.clicked.connect(self.reset_view)

            reset_lut_btn = QPushButton('Reset LUT')
            reset_lut_btn.setToolTip("Reset brightness/contrast to defaults")
            reset_lut_btn.clicked.connect(self.reset_lut)
            
            for btn in [apply_correction_btn, reset_btn, reset_lut_btn]:
                btn.setMinimumWidth(150)
                button_layout.addWidget(btn)

        # Add all sections to main layout
        layout.addWidget(nav_group, 0, 0, 1, 1)
        layout.addWidget(brightness_group, 0, 1, 1, 1)
        layout.addWidget(button_group, 0, 2, 1, 1)
        
        return panel

    def create_brightness_contrast_controls(self):
        """Create improved brightness/contrast controls without a reset button"""
        group = QFrame()
        group.setMaximumHeight(100)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Create controls for both images
        for img_type, label in [('moving', 'Moving'), ('reference', 'Fixed')]:
            frame = QFrame()
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.setSpacing(2)
            
            # Add label at top
            frame_layout.addWidget(QLabel(label), alignment=Qt.AlignCenter)
            
            # Create horizontal layout for B/C controls
            controls_layout = QHBoxLayout()
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(4)
            
            # Create brightness and contrast controls
            for ctrl_type, label in [('brightness', 'B'), ('contrast', 'C')]:
                ctrl_frame = QFrame()
                ctrl_layout = QVBoxLayout(ctrl_frame)
                ctrl_layout.setContentsMargins(0, 0, 0, 0)
                ctrl_layout.setSpacing(1)
                
                # Label
                ctrl_layout.addWidget(QLabel(label), alignment=Qt.AlignCenter)
                
                # Slider
                slider = QSlider(Qt.Vertical)
                slider.setRange(1, 200)
                slider.setValue(100)
                slider.setTickPosition(QSlider.TicksRight)
                slider.setTickInterval(50)
                
                # Store slider reference
                setattr(self, f"{img_type}_{ctrl_type}_slider", slider)
                
                # Connect after setting the attribute to avoid lambda capture issues
                # This explicit pattern prevents the common lambda closure bug in loops
                if ctrl_type == 'brightness' and img_type == 'moving':
                    slider.valueChanged.connect(lambda v: self.update_image_display('moving'))
                elif ctrl_type == 'contrast' and img_type == 'moving':
                    slider.valueChanged.connect(lambda v: self.update_image_display('moving'))
                elif ctrl_type == 'brightness' and img_type == 'reference':
                    slider.valueChanged.connect(lambda v: self.update_image_display('reference'))
                elif ctrl_type == 'contrast' and img_type == 'reference':
                    slider.valueChanged.connect(lambda v: self.update_image_display('reference'))
                    
                ctrl_layout.addWidget(slider)
                
                controls_layout.addWidget(ctrl_frame)
            
            frame_layout.addLayout(controls_layout)
            layout.addWidget(frame)
        
        return group

    def update_image_display(self, image_type):
        """Update display with amplified brightness scaling"""
        if image_type == 'moving' and hasattr(self, 'moving_lut'):
            brightness = self.moving_brightness_slider.value() / 100.0
            contrast = self.moving_contrast_slider.value() / 100.0
            lut = self.moving_lut
            color = (1.0, 0.0, 0.0)
        elif image_type == 'reference' and hasattr(self, 'reference_lut'):
            brightness = self.reference_brightness_slider.value() / 100.0
            contrast = self.reference_contrast_slider.value() / 100.0
            lut = self.reference_lut
            color = (0.0, 1.0, 0.0)
        else:
            return

        # Calculate brightness amplification factor
        brightness_factor = 1.0 + 3.0 * brightness  # Scales from 1.0 to 4.0

        # Update LUT with amplified scaling
        for i in range(256):
            val = i / 255.0
            
            # Skip background
            if val == 0:
                lut.SetTableValue(i, 0.0, 0.0, 0.0, 0.0)
                continue
                    
            # Apply contrast
            adjusted = 0.5 + (val - 0.5) * contrast
            
            # Amplify by brightness (multiplicative scaling UP)
            intensity = adjusted * brightness_factor
            
            # Clamp to valid range
            intensity = min(1.0, max(0.0, intensity))
            
            lut.SetTableValue(i, color[0], color[1], color[2], intensity)
        
        lut.Modified()
        self.vtk_widget.GetRenderWindow().Render()

    def reset_lut(self):
        """Reset LUTs to default linear intensity mapping"""
        # Reset slider values
        self.moving_brightness_slider.setValue(100)
        self.moving_contrast_slider.setValue(100)
        self.reference_brightness_slider.setValue(100)
        self.reference_contrast_slider.setValue(100)
        
        # Recreate linear mapping for moving image LUT
        if hasattr(self, 'moving_lut'):
            for i in range(256):
                val = i / 255.0  # Linear scaling
                self.moving_lut.SetTableValue(i, 1.0, 0.0, 0.0, val)
            self.moving_lut.Modified()
        
        # Recreate linear mapping for reference image LUT
        if hasattr(self, 'reference_lut'):
            for i in range(256):
                val = i / 255.0  # Linear scaling
                self.reference_lut.SetTableValue(i, 0.0, 1.0, 0.0, val)
            self.reference_lut.Modified()
        
        # Trigger render
        self.vtk_widget.GetRenderWindow().Render()

    def create_lookup_table(self, color, num_entries):
        """Create VTK lookup table with specified color and number of entries"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(num_entries)
        lut.SetRange(0, 1)  # Keep range 0-1 for consistent interface
        lut.Build()
        
        # Initialize with full range
        for i in range(num_entries):
            alpha = i / (num_entries - 1)  # Normalize to [0,1]
            lut.SetTableValue(i, color[0], color[1], color[2], alpha)
        
        return lut

    def create_info_panel(self):
        """Create right panel with improved layout"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Panel | QFrame.Raised)
        panel.setMinimumWidth(300)  # Set minimum width for right panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        # Add model selection group at top
        model_group = QFrame()
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(5)

        # Model selection header
        model_layout.addWidget(QLabel('<b>Model Selection</b>'))
        
        # Model file selection row
        file_frame = QFrame()
        file_layout = QHBoxLayout(file_frame)
        file_layout.setContentsMargins(0, 0, 0, 0)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select model file...")
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                color: #ffffff;
                padding: 5px;
                selection-background-color: #3d3d3d;}
            QLineEdit:disabled {
                background-color: #222222;
                color: #666666;
            } """)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_model)
        browse_btn.setMaximumWidth(70)
        
        file_layout.addWidget(self.model_path_edit)
        file_layout.addWidget(browse_btn)
        model_layout.addWidget(file_frame)
        
        # Prediction control buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.predict_btn = QPushButton("Predict Transform")
        self.predict_btn.clicked.connect(self.predict_transform)
        self.predict_btn.setEnabled(False)
        
        reset_transform_btn = QPushButton("Reset Transform")
        reset_transform_btn.clicked.connect(self.reset_transform)
        
        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(reset_transform_btn)
        model_layout.addWidget(button_frame)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        model_layout.addWidget(separator)
        
        layout.addWidget(model_group)
        layout.addSpacing(10)

        # Manual alignment controls (initially hidden)
        self.manual_controls = QFrame()
        manual_layout = QVBoxLayout(self.manual_controls)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(10)

        # Step size controls in a grid
        step_frame = QFrame()
        step_layout = QGridLayout(step_frame)
        step_layout.setContentsMargins(0, 0, 0, 0)
        
        # Rotation step
        step_layout.addWidget(QLabel('Rotation Step:'), 0, 0)
        self.rotation_step = QComboBox()
        self.rotation_step.addItems(['0.1°', '1.0°', '5.0°'])
        self.rotation_step.currentTextChanged.connect(self.update_rotation_step)
        step_layout.addWidget(self.rotation_step, 0, 1)
        
        # Translation step
        step_layout.addWidget(QLabel('Translation Step:'), 1, 0)
        self.translation_step = QComboBox()
        self.translation_step.addItems(['0.1 vox', '1.0 vox', '5.0 vox'])
        self.translation_step.currentTextChanged.connect(self.update_translation_step)
        step_layout.addWidget(self.translation_step, 1, 1)
        
        manual_layout.addWidget(step_frame)

        # Rotation controls
        rotation_group = QFrame()
        rotation_layout = QVBoxLayout(rotation_group)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        
        rotation_layout.addWidget(QLabel('<b>Rotation Controls</b>'))
        for axis in ['X', 'Y', 'Z']:
            axis_frame = QFrame()
            axis_layout = QHBoxLayout(axis_frame)
            axis_layout.setContentsMargins(0, 0, 0, 0)
            
            axis_layout.addWidget(QLabel(f'{axis}-axis:'))
            minus_btn = QPushButton(f'R{axis}-')
            plus_btn = QPushButton(f'R{axis}+')
            
            minus_btn.clicked.connect(
                lambda c, a=axis.lower(): self.adjust_transform('rotation', a, -1))
            plus_btn.clicked.connect(
                lambda c, a=axis.lower(): self.adjust_transform('rotation', a, 1))
            
            axis_layout.addWidget(minus_btn)
            axis_layout.addWidget(plus_btn)
            rotation_layout.addWidget(axis_frame)
        
        manual_layout.addWidget(rotation_group)

        # Translation controls
        translation_group = QFrame()
        translation_layout = QVBoxLayout(translation_group)
        translation_layout.setContentsMargins(0, 0, 0, 0)
        
        translation_layout.addWidget(QLabel('<b>Translation Controls</b>'))
        for axis in ['X', 'Y', 'Z']:
            axis_frame = QFrame()
            axis_layout = QHBoxLayout(axis_frame)
            axis_layout.setContentsMargins(0, 0, 0, 0)
            
            axis_layout.addWidget(QLabel(f'{axis}-axis:'))
            minus_btn = QPushButton(f'T{axis}-')
            plus_btn = QPushButton(f'T{axis}+')
            
            minus_btn.clicked.connect(
                lambda c, a=axis.lower(): self.adjust_transform('translation', a, -1))
            plus_btn.clicked.connect(
                lambda c, a=axis.lower(): self.adjust_transform('translation', a, 1))
            
            axis_layout.addWidget(minus_btn)
            axis_layout.addWidget(plus_btn)
            translation_layout.addWidget(axis_frame)
        
        manual_layout.addWidget(translation_group)
        
        layout.addWidget(self.manual_controls)
        self.manual_controls.setVisible(self.mode == 'correction')

        # Transform info
        info_group = QFrame()
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        info_layout.addWidget(QLabel('<b>Transform Info</b>'))
        self.transform_info = QTextEdit()
        self.transform_info.setReadOnly(True)
        self.transform_info.setMinimumHeight(50)
        info_layout.addWidget(self.transform_info)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        return panel

    def update_transform_info(self):
        """Update transform info display """
        # Get transform parameters
        transform_info = (
            f"Transform Parameters:\n"
            f"Rotation (deg):\n"
            f"  X={self.manual_params['rotation'][0]:.1f}\n"
            f"  Y={self.manual_params['rotation'][1]:.1f}\n"
            f"  Z={self.manual_params['rotation'][2]:.1f}\n"
            f"\nTranslation (mm):\n"
            f"  X={self.manual_params['translation'][0]:.1f}\n"
            f"  Y={self.manual_params['translation'][1]:.1f}\n"
            f"  Z={self.manual_params['translation'][2]:.1f}"
        )
     
        # Update display
        self.transform_info.setText(transform_info)

####### VTK image display

    def setup_vtk_widget(self, layout):
        """Setup VTK visualization widget"""
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor()
        layout.addWidget(self.vtk_widget)

        # Create renderer and set background
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        # Add renderer to render window
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.renderer)

        # Create and set interactor style
        style = vtk.vtkInteractorStyleImage()
        self.vtk_widget.SetInteractorStyle(style)

        # Setup image visualization pipeline with transform
        self.setup_image_pipeline()

        # Initialize the interactor and start event loop
        self.vtk_widget.Initialize()
        render_window.Render()

        # Setup keyboard callbacks only (no mouse needed for manual transform)
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.AddObserver("KeyPressEvent", self.on_key_press)

    def create_vtk_3d_image(self, np_array):
        """Convert numpy array to VTK 3D image data with proper bit depth handling"""
        vtk_image = vtk.vtkImageData()
        depth, height, width = np_array.shape
        
        # Handle 16-bit images by pre-scaling them to 8-bit range
        dtype = np_array.dtype
        if dtype == np.uint16:
            # Find max value (avoid division by zero)
            max_val = float(np.max(np_array))
            if max_val == 0:
                max_val = 1.0  # Avoid division by zero
            
            # Create a scaled copy that maps [0, max_val] to [0, 255]
            scale_factor = 255.0 / max_val
            scaled_array = (np_array.astype(np.float32) * scale_factor).astype(np.uint8)
            
            # Use the scaled array and set type to uint8
            np_array = scaled_array
            vtk_dtype = vtk.VTK_UNSIGNED_CHAR
        elif dtype == np.uint8:
            vtk_dtype = vtk.VTK_UNSIGNED_CHAR
        else:
            # Convert to float32 for other types
            vtk_dtype = vtk.VTK_FLOAT
            np_array = np_array.astype(np.float32)
        
        # Initialize with 3D dimensions
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.SetExtent(0, width-1, 0, height-1, 0, depth-1)
        vtk_image.SetSpacing(*self.spacing)
        vtk_image.AllocateScalars(vtk_dtype, 1)
        
        # Fill 3D data preserving original data type
        memory_view = vtk_image.GetPointData().GetScalars()
        numpy_array = numpy_support.vtk_to_numpy(memory_view)
        numpy_array.reshape(depth * height * width)[:] = np_array.ravel()
        
        return vtk_image

    def setup_image_pipeline(self):
        """Setup VTK pipeline with proper linear intensity mapping"""
        self.moving_data = self.create_vtk_3d_image(self.transformed_img)
        self.reference_data = self.create_vtk_3d_image(self.reference_img)
                
        # Initialize LUTs with linear mapping
        self.moving_lut = vtk.vtkLookupTable()
        self.moving_lut.SetNumberOfTableValues(256)
        self.moving_lut.SetRange(0, 1)
        
        self.reference_lut = vtk.vtkLookupTable()
        self.reference_lut.SetNumberOfTableValues(256)
        self.reference_lut.SetRange(0, 1)

        # linear intensity mapping
        for i in range(256):
            val = i / 255.0  # Linear scaling
            self.moving_lut.SetTableValue(i, 1.0, 0.0, 0.0, val)
            self.reference_lut.SetTableValue(i, 0.0, 1.0, 0.0, val)

        # Standard pipeline setup
        self.vtk_transform = vtk.vtkTransform()
        self.vtk_transform.PostMultiply()

        self.transform_filter = vtk.vtkImageReslice()
        self.transform_filter.SetInputData(self.moving_data)
        self.transform_filter.SetResliceTransform(self.vtk_transform)
        self.transform_filter.SetInterpolationModeToCubic()
        self.transform_filter.SetOutputDimensionality(3)
        self.transform_filter.AutoCropOutputOff()

        moving_mapper = vtk.vtkImageMapToColors()
        moving_mapper.SetInputConnection(self.transform_filter.GetOutputPort())
        moving_mapper.SetLookupTable(self.moving_lut)
        
        reference_mapper = vtk.vtkImageMapToColors()
        reference_mapper.SetInputData(self.reference_data)
        reference_mapper.SetLookupTable(self.reference_lut)

        self.moving_actor = vtk.vtkImageActor()
        self.moving_actor.GetMapper().SetInputConnection(moving_mapper.GetOutputPort())
        self.moving_actor.InterpolateOn()
        
        self.reference_actor = vtk.vtkImageActor()
        self.reference_actor.GetMapper().SetInputConnection(reference_mapper.GetOutputPort())
        self.reference_actor.InterpolateOn()

        self.renderer.AddActor(self.moving_actor)
        self.renderer.AddActor(self.reference_actor)
        self.renderer.ResetCamera()

    def update_display(self):
        """Update the displayed slice with proper extent handling"""
        if not self.renderer:
            return

        # Get image dimensions
        dims = self.reference_data.GetDimensions()
        width, height, depth = dims

        # Clamp current slice
        if self.current_plane == 'xy':
            max_slice = depth - 1
            current_slice = np.clip(self.current_slice[self.current_plane], 0, max_slice)
            self.moving_actor.SetDisplayExtent(0, width-1, 0, height-1, current_slice, current_slice)
            self.reference_actor.SetDisplayExtent(0, width-1, 0, height-1, current_slice, current_slice)
        
        elif self.current_plane == 'yz':
            max_slice = width - 1
            current_slice = np.clip(self.current_slice[self.current_plane], 0, max_slice)
            self.moving_actor.SetDisplayExtent(current_slice, current_slice, 0, height-1, 0, depth-1)
            self.reference_actor.SetDisplayExtent(current_slice, current_slice, 0, height-1, 0, depth-1)
        
        else:  # 'xz'
            max_slice = height - 1
            current_slice = np.clip(self.current_slice[self.current_plane], 0, max_slice)
            self.moving_actor.SetDisplayExtent(0, width-1, current_slice, current_slice, 0, depth-1)
            self.reference_actor.SetDisplayExtent(0, width-1, current_slice, current_slice, 0, depth-1)

        # Update slice label
        self.slice_label.setText(f'Slice: {current_slice}')
        
        # Ensure proper camera clipping
        self.renderer.ResetCameraClippingRange()
        
        # Force render
        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    def update_plane_and_slider(self, plane_index):
        """Update the viewing plane, synchronize the slider range, and set camera orientation."""
        try:
            # Map index to plane
            planes = ['xy', 'yz', 'xz']
            self.current_plane = planes[plane_index]

            # Determine maximum slice based on the current plane
            if self.current_plane == 'xy':  # Axial
                max_slice = self.reference_img.shape[0] - 1
                current_slice = min(max(0, self.current_slice['xy']), max_slice)
            elif self.current_plane == 'yz':  # Sagittal
                max_slice = self.reference_img.shape[2] - 1
                current_slice = min(max(0, self.current_slice['yz']), max_slice)
            else:  # Coronal (xz)
                max_slice = self.reference_img.shape[1] - 1
                current_slice = min(max(0, self.current_slice['xz']), max_slice)

            # Update current slice for the selected plane
            self.current_slice[self.current_plane] = current_slice
            self.slice_slider.setRange(0, max_slice)
            self.slice_slider.setValue(current_slice)

            # Set camera orientation once per plane change
            camera = self.renderer.GetActiveCamera()
            dims = self.reference_data.GetDimensions()  # (width, height, depth)
            width, height, depth = dims
            camera.SetFocalPoint(width/2.0, height/2.0, depth/2.0)

            # Adjust camera per plane
            if self.current_plane == 'xy':  
                # Axial: looking along Z
                camera.SetPosition(width/2.0, height/2.0, -500.0)
                camera.SetViewUp(0, -1, 0)

            elif self.current_plane == 'yz':  
                # Sagittal: looking along X
                camera.SetPosition(-500.0, height/2.0, depth/2.0)
                camera.SetViewUp(0, 0, -1)

            else:  
                # Coronal: looking along Y
                camera.SetPosition(width/2.0, -500.0, depth/2.0)
                camera.SetViewUp(0, 0, -1) 

            # Reset camera after setting orientation
            self.renderer.ResetCamera()
            self.renderer.ResetCameraClippingRange()

            # Now update display for the newly chosen plane
            self.update_display()

        except Exception as e:
            print(f"Error in update_plane_and_slider: {e}")
            traceback.print_exc()

####### Manual input Managment

    def apply_correction(self):
        """Generate SITK transform from current manual parameters"""
        # Decompose each rotation
        rx = np.radians(self.manual_params['rotation'][0])  # X rotation in VTK
        ry = np.radians(self.manual_params['rotation'][1])  # Y rotation
        rz = np.radians(self.manual_params['rotation'][2])  # Z rotation
        tx=self.manual_params['translation'][0]
        ty=self.manual_params['translation'][1]
        tz=self.manual_params['translation'][2]
        # Combine into final parameters
        params = [rx, ry, rz] + [tx, ty, tz] 
        # Convert to tensor
        target_transform = torch.tensor(params, dtype=torch.float32)
            
        self.correction_ready = False
        return target_transform

    def adjust_transform(self, transform_type, axis, direction):
        """Apply incremental transformation using VTK transform in anatomical space"""
        # Get step size and apply direction
        step = self.step_size[transform_type] * direction
        # Map to anatomical axes (x, y, z)
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        # Update tracking parameters
        self.manual_params[transform_type][axis_index] += step

        #  Get dimensions
        dims = self.reference_data.GetDimensions()
        spacing = self.reference_data.GetSpacing()

        # Calculate center in physical coordinates
        center = [
            (dims[0] - 1) *spacing[0]/ 2.0,
            (dims[1] - 1) *spacing[1]/ 2.0,
            (dims[2] - 1) *spacing[2]/ 2.0
        ]

        # Create transform and set to use post-multiplication
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Identity()
        
        # First translate geometry to origin
        transform.Translate(-center[0], -center[1], -center[2])
        # Apply the rotations in SITK order
        rotations = self.manual_params['rotation']
        transform.RotateZ(rotations[2])
        transform.RotateY(rotations[1])
        transform.RotateX(rotations[0])
        # Move geometry back
        transform.Translate(center[0], center[1], center[2])
        
        # Apply any translations
        translations = self.manual_params['translation']
        transform.Translate(translations[0]*spacing[0], 
                            translations[1]*spacing[1],
                            translations[2]*spacing[2])
        
        # Update the VTK transform
        self.vtk_transform.SetMatrix(transform.GetMatrix())
        
        # Update visualization
        self.transform_filter.Modified()
        self.vtk_widget.GetRenderWindow().Render()
        
        # Update transform info display
        self.update_transform_info()

###### Model prediction

    def load_model(self, model_path):
        """Load the registration model from the specified path"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RegistrationNet()
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            self.model = model
            self.model_path = model_path
            self.statusBar().showMessage('Model loaded successfully')
            return True
        except Exception as e:
            self.statusBar().showMessage(f'Error loading model: {str(e)}')
            return False

    def browse_model(self):
        """Open file dialog to select model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pth);;All Files (*.*)")
        if file_path:
            if self.load_model(file_path):
                self.model_path_edit.setText(file_path)
                self.predict_btn.setEnabled(True)

    def predict_transform(self):
        """Use loaded model to predict transform"""
        if not self.model:
            self.statusBar().showMessage('No model loaded')
            return
            
        try:
            # Prepare images for model
            if torch.is_tensor(self.raw_moving_img):
                moving_tensor = self.raw_moving_img.clone()
            else:
                moving_tensor = torch.from_numpy(self.raw_moving_img).float()
                
            if torch.is_tensor(self.reference_img):
                reference_tensor = self.reference_img.clone()
            else:
                reference_tensor = torch.from_numpy(self.reference_img).float()
                
            # Ensure proper dimensions
            if moving_tensor.dim() == 3:
                moving_tensor = moving_tensor.unsqueeze(0).unsqueeze(0)
            elif moving_tensor.dim() == 4:
                moving_tensor = moving_tensor.unsqueeze(0)
                
            if reference_tensor.dim() == 3:
                reference_tensor = reference_tensor.unsqueeze(0).unsqueeze(0)
            elif reference_tensor.dim() == 4:
                reference_tensor = reference_tensor.unsqueeze(0)
                
            # Make prediction
            with torch.no_grad():
                transform = self.model(reference_tensor, moving_tensor)
                
            # Convert to numpy if needed
            if torch.is_tensor(transform):
                transform = transform.detach().cpu().numpy()
                if transform.ndim > 1:
                    transform = transform[0]
                    
            # Update UI parameters
            self.manual_params['rotation'] = [
                float(np.degrees(transform[0])), 
                float(np.degrees(transform[1])), 
                float(np.degrees(transform[2]))
            ]
            self.manual_params['translation'] = [
                float(transform[3]), 
                float(transform[4]), 
                float(transform[5])
            ]
            
            # Apply transform
            self.adjust_transform('rotation', 'x', 0)  # This will trigger UI update
            self.statusBar().showMessage('Transform predicted and applied')
            
        except Exception as e:
            self.statusBar().showMessage(f'Error predicting transform: {str(e)}')

    def reset_transform(self):
        """Reset transform parameters to zero"""
        self.manual_params = {
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, 0.0, 0.0]
        }
        self.vtk_transform.Identity()
        self.update_transform_info()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage('Transform reset to identity')

####### SITK Tools

    def sitk_transform_to_tensor(self, transform):
        """Convert SITK transform to tensor parameters
        Returns: tensor [rx, ry, rz, tx, ty, tz] in SITK format
        """
        params = transform.GetParameters()
        # Already in correct format, just convert to tensor
        return torch.tensor(params, dtype=torch.float32)

    def apply_transform(self, transform_tensor):
        """Apply initial predicted transform using SITK"""

        # Set the number of threads for resampling
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(mp.cpu_count())

        # Fetch Raw images
        moving_sitk = sitk.GetImageFromArray(self.raw_moving_img)
        reference_sitk = sitk.GetImageFromArray(self.reference_img)
        # 1. Identity transform for spatial normalization
        null_transform = sitk.Euler3DTransform()
        identity_transformed = sitk.Resample(moving_sitk, reference_sitk,
                null_transform, sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())

        # 2. Convert prediction into SITK transform
        # Convert tensor to numpy if needed
        if torch.is_tensor(transform_tensor):
                transform_params = transform_tensor.detach().cpu().numpy()
                if transform_params.ndim > 1:
                    transform_params = transform_params[0]
        else:
                transform_params = transform_tensor
        current_transform = sitk.Euler3DTransform()
        # Parameters from tensor already in SITK order
        current_transform.SetParameters(transform_params.tolist())
        # Set transform center
        size = reference_sitk.GetSize()
        center = [(size[0] - 1) / 2.0, (size[1] - 1) / 2.0, (size[2] - 1) / 2.0]
        current_transform.SetCenter(center)

        # 3. Apply predicted transform to normalized image
        transformed = sitk.Resample(identity_transformed, reference_sitk,
                current_transform, sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())

        # Update transformed image and display
        self.transformed_img = sitk.GetArrayFromImage(transformed)

        # Update VTK visualization if pipeline exists
        if hasattr(self, 'moving_data'):
                self.moving_data = self.create_vtk_3d_image(self.transformed_img)
                self.transform_filter.SetInputData(self.moving_data)
                self.transform_filter.Update()
                self.vtk_transform.Identity()
                self.update_display()

####### Interactions

    def on_slice_change(self, value):
        """Handle slider changes and update the current slice."""
        # Update the current slice and display
        self.current_slice[self.current_plane] = value
        self.update_display()

    def update_rotation_step(self, value):
        """Update rotation step size"""
        self.step_size['rotation'] = float(value.rstrip('°'))

    def update_translation_step(self, value):
        """Update translation step size"""
        self.step_size['translation'] = float(value.rstrip(' vox')) 

    def update_step_size(self, transform_type, value):
        """Update step size for the specified transform type"""
        self.step_size[transform_type] = value

    def initialize_alignment_mode(self):
        moving_sitk = sitk.GetImageFromArray(self.raw_moving_img)
        reference_sitk = sitk.GetImageFromArray(self.reference_img)
        self.transformed_img = sitk.GetArrayFromImage(sitk.Resample(moving_sitk, reference_sitk, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()))
        self.moving_data = self.create_vtk_3d_image(self.transformed_img)
        self.transform_filter.SetInputData(self.moving_data)
        self.manual_params = {'rotation': [0.0, 0.0, 0.0], 'translation': [0.0, 0.0, 0.0]}
        self.vtk_transform.Identity()
        self.vtk_widget.GetRenderWindow().Render()
          
    def setup_callbacks(self):
        """Setup mouse and keyboard interaction callbacks"""
        # Get the interactor
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Add observers to the interactor
        interactor.AddObserver("KeyPressEvent", self.on_key_press)

    def on_generate_correction_clicked(self):
        """Handler for Generate Correction button"""
        self.correction_ready = True
        self.apply_correction()
        self.close()

    def on_key_press(self, obj, event):
        """Enhanced keyboard event handling"""
        key = obj.GetKeySym()
        if key in ["Up", "Down", "Left", "Right"]:
            current = self.slice_slider.value()
            if key in ["Up", "Right"]:
                self.slice_slider.setValue(current + 1)
            else:
                self.slice_slider.setValue(current - 1)
        elif key == "a":
            self.apply_correction()
        elif key == "v":
            self.validate_registration()
        elif key == "r":
            self.reset_view()

    def validate_registration(self):
        """Validate current registration"""
        self.was_validated = True
        self.close()

    def reset_view(self):
        """Reset camera view"""
        if self.renderer:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

    def show(self):
        """Show window and wait for user interaction to complete"""
        self.setWindowModality(Qt.ApplicationModal)
        super().show()

        # Initialize the VTK widget but don't start the event loop
        self.vtk_widget.Initialize()
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.Render()
        
        # Set up a continuous rendering loop
        timer = QTimer(self)
        timer.timeout.connect(self.render_update)
        timer.start(16)  # ~60 FPS
        
        # Keep the window visible and responsive
        while self.isVisible():
            QApplication.instance().processEvents()
            
        timer.stop()
        
        # Return the results
        if self.mode == 'validate':
            return True
        else:
            target_transform = self.apply_correction() if not self.was_validated else None
            return target_transform

    def render_update(self):
        """Update rendering in response to timer"""
        if self.isVisible():
            self.vtk_widget.GetRenderWindow().Render()

    def closeEvent(self, event):
            """Handle window close event"""
            self.vtk_widget.Finalize()
            event.accept()
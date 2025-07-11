"""Automatically run all figure generation scripts."""

import os
import importlib
import pytest
from unittest.mock import patch
import matplotlib.pyplot as plt

def get_all_figure_modules():
    """Dynamically discover all figure modules in the figures directory."""
    figure_modules = []
    figures_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(figures_dir), "figures")
    
    for filename in os.listdir(figures_dir):
        # Only look at Python files that start with "figure"
        if filename.startswith("figure") and filename.endswith(".py"):
            module_name = filename[:-3]  # Remove .py extension
            
            # Import the module dynamically
            module = importlib.import_module(f"..figures.{module_name}", package="cellcommunicationpf2.tests")
            
            # Check if the module has a makeFigure function
            if hasattr(module, "makeFigure"):
                figure_modules.append(module)
    
    return figure_modules

@pytest.mark.parametrize("figure_module", get_all_figure_modules())
def test_figure_generation(figure_module):
    """Test that figures can be generated without errors."""
    print(f"Testing {figure_module.__name__}")
    
    # Patch plt.show and plt.savefig to avoid displaying/saving figures during tests
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
        # Generate the figure
        fig = figure_module.makeFigure()
        
        # Make sure we got a figure back
        assert fig is not None
        
        # Close the figure to clean up
        plt.close(fig)
"""Build script for creating executable using PyInstaller

This version conditionally includes hidden imports, metadata, and collected
resources only for packages that are actually installed. This avoids
PackageNotFoundError when copying metadata (e.g., for GDAL) on systems where
those packages are not present.
"""
import subprocess
import sys
import os
import time
from pathlib import Path
import importlib.util

def is_package_installed(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def build():
    try:
        # Install PyInstaller
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
        # Get paths
        current_dir = Path(__file__).parent
        simulation_dir = current_dir / "simulation_sukabumi"
        dist_dir = current_dir / "dist"
        exe_path = dist_dir / "AmbulanceRouting.exe"
        
        # Try to delete existing exe if it exists
        if exe_path.exists():
            print(f"Removing existing executable: {exe_path}")
            try:
                exe_path.unlink()
                time.sleep(0.5)
            except PermissionError:
                print("\n" + "="*60)
                print("ERROR: Cannot delete existing executable!")
                print(f"File: {exe_path}")
                print("\nPossible causes:")
                print("  1. The application is currently running - close it")
                print("  2. File is open in another program")
                print("  3. Antivirus is scanning the file")
                print("  4. Windows Explorer has the file open")
                print("\nPlease close all instances and try again.")
                print("="*60)
                input("\nPress Enter after closing the application to continue...")
                try:
                    exe_path.unlink()
                except PermissionError as e:
                    print(f"Still cannot delete: {e}")
                    print("Please manually delete the file and run build again.")
                    sys.exit(1)
        
        # Build command
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name", "AmbulanceRouting",
            "--windowed",  # ✅ Changed to console for debugging
            "--clean",
            "--noconfirm",
        ]
        
        # Add simulation data folder if it exists
        if simulation_dir.exists():
            cmd.extend(["--add-data", f"{simulation_dir};simulation_sukabumi"])
        
        # Core dependencies (always needed)
        core_imports = [
            "nicegui",
            "starlette", 
            "uvicorn",
            "folium",
            "networkx",
            "matplotlib",
            "pandas",
            "numpy",
            "branca",
            "jinja2",
        ]
        
        for pkg in core_imports:
            if is_package_installed(pkg):
                cmd.extend(["--hidden-import", pkg])
        
        # Geospatial stack - add hidden imports
        geo_imports = [
            "osmnx",
            "geopandas", 
            "shapely",
            "pyogrio",
            "pyogrio._io",  # ✅ Add pyogrio C extensions explicitly
            "pyogrio._geometry",  # ✅
            "pyogrio.core",
            "pyogrio._env",
            "fiona",
            "pyproj",
            "osgeo",
            "osgeo.gdal",
            "osgeo.ogr",
            "osgeo.osr",
        ]
        
        for pkg in geo_imports:
            if is_package_installed(pkg.split('.')[0]):  # Check base package
                cmd.extend(["--hidden-import", pkg])
        
        # Copy metadata for packages that need it
        metadata_packages = [
            "osmnx",
            "nicegui",
            "folium", 
            "geopandas",
            "shapely",
            "networkx",
            "pyogrio",
            "fiona",
            "pyproj",
        ]
        
        for pkg in metadata_packages:
            if is_package_installed(pkg):
                cmd.extend(["--copy-metadata", pkg])
        
        # Collect all binaries and data - CRITICAL for GDAL/pyogrio
        collect_packages = [
            "osmnx",
            "nicegui",
            "pyogrio",  # ✅ Most important - includes GDAL DLLs and C extensions
            "fiona",
            "pyproj",
            "geopandas",
            "shapely",
            "osgeo",
        ]
        
        for pkg in collect_packages:
            if is_package_installed(pkg):
                cmd.extend(["--collect-all", pkg])
        
        cmd.append("main.py")
        
        print("\nBuilding application with PyInstaller...")
        print(f"Command: {' '.join(cmd)}\n")
        subprocess.check_call(cmd)
        
        print("\n" + "="*60)
        print("✅ Build completed successfully!")
        print(f"📁 Executable location: {exe_path}")
        print("\nNote: Built with --console for debugging.")
        print("If GDAL works, change to --windowed in build.py")
        print("="*60)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    build()
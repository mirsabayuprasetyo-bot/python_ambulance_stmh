import os, sys, pathlib


def _set_gdal_env():
    """Configure GDAL/PROJ environment for both dev and PyInstaller exe"""
    # In a PyInstaller exe, collected files reside in _MEIPASS
    if getattr(sys, 'frozen', False):
        base = pathlib.Path(sys._MEIPASS)
    else:
        base = pathlib.Path(__file__).parent
    
    print(f"[GDAL Setup] Base path: {base}")
    
    # Common locations after --collect-all
    osgeo_dir = base / 'osgeo'
    pyproj_dir = base / 'pyproj'
    pyogrio_dir = base / 'pyogrio'
    pyogrio_lib_dir = base / 'pyogrio' / '.libs'  # ✅ pyogrio stores GDAL DLLs here
    
    # Add ALL potential DLL directories FIRST (before any imports)
    dll_dirs = [
        base,
        osgeo_dir,
        pyproj_dir, 
        pyogrio_dir,
        pyogrio_lib_dir,
        base / 'gdal',
        base / 'fiona',
        base / 'shapely',
        base / 'shapely' / '.libs',
    ]
    
    if hasattr(os, 'add_dll_directory'):
        for dll_dir in dll_dirs:
            if dll_dir.exists():
                try:
                    os.add_dll_directory(str(dll_dir))
                    print(f"[GDAL Setup] Added DLL directory: {dll_dir}")
                except Exception as e:
                    print(f"[GDAL Setup] Could not add {dll_dir}: {e}")
    
    # Set GDAL_DATA
    gdal_data_candidates = [
        pyogrio_dir / 'gdal_data',  # ✅ pyogrio often has this
        osgeo_dir / 'data' / 'gdal',
        osgeo_dir / 'data',
        base / 'gdal' / 'data',
        base / 'Library' / 'share' / 'gdal',  # conda-forge location
    ]
    
    for p in gdal_data_candidates:
        if p.exists():
            os.environ.setdefault('GDAL_DATA', str(p))
            print(f"[GDAL Setup] Set GDAL_DATA: {p}")
            break
    
    # Set PROJ_DATA/PROJ_LIB - try pyproj API first
    try:
        from pyproj import datadir as _proj_datadir
        proj_dir = _proj_datadir.get_data_dir()
        if proj_dir and os.path.exists(proj_dir):
            os.environ.setdefault('PROJ_LIB', proj_dir)
            os.environ.setdefault('PROJ_DATA', proj_dir)
            print(f"[GDAL Setup] Set PROJ via pyproj API: {proj_dir}")
    except Exception as e:
        print(f"[GDAL Setup] pyproj API failed: {e}")
        # Fallback to manual search
        proj_data_candidates = [
            pyogrio_dir / 'proj_data',
            pyproj_dir / 'share' / 'proj',
            pyproj_dir / 'proj_data',
            osgeo_dir / 'share' / 'proj',
            base / 'Library' / 'share' / 'proj',  # conda-forge
        ]
        
        for p in proj_data_candidates:
            if p.exists():
                os.environ.setdefault('PROJ_LIB', str(p))
                os.environ.setdefault('PROJ_DATA', str(p))
                print(f"[GDAL Setup] Set PROJ_DATA: {p}")
                break
    
    # Print final environment for debugging
    print(f"[GDAL Setup] GDAL_DATA={os.environ.get('GDAL_DATA', 'NOT SET')}")
    print(f"[GDAL Setup] PROJ_DATA={os.environ.get('PROJ_DATA', 'NOT SET')}")
    print(f"[GDAL Setup] PROJ_LIB={os.environ.get('PROJ_LIB', 'NOT SET')}")

_set_gdal_env()

import asyncio
import map_downloader as map
import simulation as sim
from nicegui import ui, app
from pathlib import Path
import time


class main() : 
    def __init__(self) : 
        print("This is main class")
        self.simulation_instance = None

    def run_nicegui(self):
        
        @ui.page('/')
        def index_page():
            ui.page_title("Multi-Agent Hospital System")

            def on_window_close():
                print("Window closed - stopping simulation")
                if self.simulation_instance is not None:
                    self.simulation_instance.stop_simulation()
            
            app.on_disconnect(on_window_close)  

            with ui.left_drawer(top_corner=False, bottom_corner=False, bordered=True, elevated=True) as drawer:
                with ui.row().classes('items-center justify-between'):
                    ui.label("Simulation Settings")     
                ui.label("Number of Generation")
                num_generation = ui.slider(min=1, max=100, value=2, step=1).props('label-always markers snap ')

                ui.label("Number of Population")
                num_population = ui.slider(min=3, max=100, value=5, step=1).props('label-always markers snap ')

                ui.label("Simulation Time (minutes)")
                simulation_time = ui.slider(min=10, max=120, value=15, step=2).props('label-always markers snap ')
                

                # ui.label("Update Traffic Interval (minutes)")
                # update_interval = ui.slider(min=1, max=10, value=5, step=1).props('label-always markers snap ')
                ui.separator()
                
                simulation_text = ui.label("simulation not started")

                async def run_simulation():
                    start_button.enabled = False
                    start_button.text = "Running..."
                    
                    simulation_text.text = "Initializing simulation..."
                    await asyncio.sleep(0.1)
                    
                    self.simulation_instance = sim.simulation(
                        ga_generation=num_generation.value, 
                        ga_population=num_population.value, 
                        simulation_time_in_minute=simulation_time.value,
                        update_traffic_interval=5
                    )
                    
                    try:
                        simulation_text.text = "Simulation Running..."
                        await asyncio.sleep(0.1)
                        
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, 
                            self.simulation_instance.run_simulation, 
                            "sukabumi"
                        )
                        
                        if self.simulation_instance.is_stopped:
                            simulation_text.text = "Simulation Cancelled"
                            ui.notify("Simulation was cancelled", type='warning')
                        else:
                            simulation_text.text = "Simulation Ended Successfully"
                            ui.notify("Simulation completed successfully!", type='positive')
                            
                            await asyncio.sleep(0.5)
                            tab_panels.value = mean
                            show_simulation_results.refresh()
                            show_ga_result.refresh()
                            show_astar_result.refresh()
                            show_djikstra_result.refresh()
                            show_ox_result.refresh()
                        
                    except Exception as e:
                        simulation_text.text = f"Simulation Failed: {str(e)}"
                        ui.notify(f"Simulation failed: {str(e)}", type='negative')
                    finally:
                        start_button.enabled = True
                        start_button.text = "Start Simulation"
                
                def stop_simulation():
                    if self.simulation_instance:
                        self.simulation_instance.stop_simulation()
                        ui.notify("Stopping simulation...", type='warning')
                
                start_button = ui.button("Start Simulation", on_click=run_simulation, color="primary")
                ui.button("Stop Simulation", on_click=stop_simulation, color="red")

            with ui.header(wrap=False, value=True):
                ui.button('☰', on_click=drawer.toggle).classes('button--flat')
                ui.label('Ambulance Routing Simulation').style('font-weight: bold; font-size: 25pt; margin-left: 16pt;')

            location_name = "sukabumi"
            out_dir = Path.cwd() / f"simulation_{location_name}"
            map_ox_path = f"map_ox_{location_name}.html"
            map_djikstra_path = f"map_djikstra_{location_name}.html"
            map_astar_path = f"map_astar_{location_name}.html"
            map_ga_path = f"map_ga_{location_name}.html"
            plot_path = f"mean_response_time_{location_name}.png"

            @ui.refreshable
            def show_simulation_results():
                img_path = out_dir / plot_path
                if img_path.exists():
                    import time
                    cache_buster = int(time.time())
                    ui.image(f"/maps/{plot_path}?v={cache_buster}").style("max-width:100%; height:700px; width:700px;")
                else:
                    ui.label("No results yet. Please run simulation first.").classes('text-grey')

            @ui.refreshable
            def show_ga_result():
                cache_buster = int(time.time())
                app.add_static_files('/maps', str(out_dir))
                ui.html(f'''
                        <div style="padding:8px;">
                          <h3 style="margin:0 0 8px 0;">Genetics Algorithm Routing</h3>
                          <iframe src="/maps/{map_ga_path}" 
                                  style="width:100vh; height:60vh; border:0;">
                          </iframe>
                        </div>
                    ''', sanitize=False)
                
            @ui.refreshable
            def show_astar_result():
                cache_buster = int(time.time())
                app.add_static_files('/maps', str(out_dir))
                ui.html(f'''
                        <div style="padding:8px;">
                          <h3 style="margin:0 0 8px 0;">A* Algorithm Routing</h3>
                          <iframe src="/maps/{map_astar_path}" 
                                  style="width:100vh; height:60vh; border:0;">
                          </iframe>
                        </div>
                    ''', sanitize=False)
            
            @ui.refreshable
            def show_djikstra_result():
                cache_buster = int(time.time())
                app.add_static_files('/maps', str(out_dir))
                ui.html(f'''
                        <div style="padding:8px;">
                          <h3 style="margin:0 0 8px 0;">Djikstra Algorithm Routing</h3>
                          <iframe src="/maps/{map_djikstra_path}" 
                                  style="width:100vh; height:60vh; border:0;">
                          </iframe>
                        </div>
                    ''', sanitize=False)
            
            @ui.refreshable
            def show_ox_result():
                cache_buster = int(time.time())
                app.add_static_files('/maps', str(out_dir))
                ui.html(f'''
                        <div style="padding:8px;">
                          <h3 style="margin:0 0 8px 0;">BFS Algorithm Routing</h3>
                          <iframe src="/maps/{map_ox_path}" 
                                  style="width:100vh; height:60vh; border:0;">
                          </iframe>
                        </div>
                    ''', sanitize=False)

            with ui.tabs() as tabs:
                mean = ui.tab("Mean Response Time Comparison")
                ga = ui.tab("GA Map Visualization")
                astar = ui.tab("A-Star Map Visualization")
                djikstra = ui.tab("Djikstra Map Visualization")
                osm = ui.tab("OSM Map Visualization")
                
            with ui.tab_panels(tabs=tabs, value=mean) as tab_panels:
                with ui.tab_panel(mean):
                    ui.label("Simulation of Ambulance Routing").classes('text-h5')
                    show_simulation_results()
                    
                with ui.tab_panel(ga):
                    show_ga_result()
                    
                with ui.tab_panel(astar):
                    show_astar_result()
                    
                with ui.tab_panel(djikstra):
                    show_djikstra_result()
                    
                with ui.tab_panel(osm):
                    show_ox_result()

        ui.run(native=True, title="Multi-Agent Hospital System Simulation", 
               viewport="width=device-width, initial-scale=1.0", reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    obj = main()
    obj.run_nicegui()
    
    # obj.run()

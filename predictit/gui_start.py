#%%

from pathlib import Path
import eel
import sys
import warnings
import traceback

web_path = str(Path(__file__).resolve().parents[0] / "files_for_GUI")

eel.init(web_path)


# If some information from inside main(), define function here
def edit_gui_py(content, id):
    eel.edit_gui_js(content, id)


if __name__ == "__main__":

    this_path = Path(__file__).resolve().parents[1]
    this_path_string = str(this_path)
    save_plot_path = str(this_path / "files_for_GUI" / "plot.html")

    # If used not as a library but as standalone framework, add path to be able to import predictit if not opened in folder
    sys.path.insert(0, this_path_string)

    import predictit

    config = predictit.config.config

    predictit.misc._GUI = 1
    config.update({
        "show_plot": 0,
        "save_plot": 0,
        "return_type": 'detailed results dictionary',
        "data": None,
        'data_source': 'csv',
        "csv_test_data_relative_path": "",
    })

    @eel.expose
    def make_predictions(configured):

        for i, j in configured.items():
            if j != "" and i in config:
                config[i] = j
            else:
                warnings.warn(f"\n \t Inserted option with command line --{i} not found in config.py use --help for more information.\n")

        eel.edit_gui_js("Setup finished", "progress_phase")

        try:
            results = predictit.main.predict()

            div = results["plot"]

            #                            content      p_tag    id_parent    id_created    label    classes
            if config["print_result"]:
                eel.add_HTML_element(str(results["best"]), True, "content", "best_result", "Best result")

            eel.add_HTML_element(div, False, "content", "ploted_results", "Interactive plot", ["plot"])

            if config["print_table"]:
                eel.add_HTML_element(results["models_table"], False, "content", "models_table", "Models results", "table")


            eel.execute("ploted_results")

            eel.add_delete_button("content")

            if config["debug"]:
                eel.add_HTML_element(results["time_table"], False, "content", "time_parts_table", "Time schema of prediction", "table")
                eel.add_HTML_element(str(results["output"]), True, "content", "printed_output", "Everything printed", "pre-wrapped")

        except Exception:
            eel.add_HTML_element(f"\n Error in making predictions - {traceback.format_exc()} \n", True, "progress_phase", "error-log", "Error log", "pre-wrapped")
            #eel.edit_gui_js(f"\n Error in making predictions - {traceback.format_exc()} \n", "progress_phase")

    eel.start('index.html', port=0)  # mode='chrome-app'

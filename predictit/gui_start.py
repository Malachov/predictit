#%%

from pathlib import Path
import eel
import sys
import warnings
import traceback


# If some information from inside main(), define function here
def edit_gui_py(content, id):
    """Function that will change some element in html GUI and is callable from other py scripts.

    Args:
        content (str): New content.
        id (str): Id of changed element.
    """

    eel.edit_gui_js(content, id)


def run_gui():
    """Start web based GUI.
    """

    web_path = str(Path(__file__).resolve().parents[0] / "files_for_GUI")

    eel.init(web_path)

    this_path = Path(__file__).resolve().parents[1]
    this_path_string = str(this_path)

    # If used not as a library but as standalone framework, add path to be able to import predictit if not opened in folder
    sys.path.insert(0, this_path_string)

    import predictit

    Config = predictit.configuration.Config

    predictit.misc._GUI = 1
    Config.update({
        "show_plot": 0,
        "save_plot": 0,
        "return_type": 'detailed_dictionary',
        "data": None,
        'data_source': 'csv',
        "csv_test_data_relative_path": "",
    })

    # Functions with @eel prefix are usually called from main.js file.

    @eel.expose
    def make_predictions(configured):
        """Function that from web GUI button trigger the predictit main predict function and return results on GUI.

        Args:
            configured (dict): Some configuration values can be configured in GUI.
        """
        for i, j in configured.items():
            if j != "" and i in predictit.configuration.all_variables_set:

                try:
                    val = int(j)
                except ValueError:
                    try:
                        val = float(j)
                    except ValueError:
                        val = j

                setattr(Config, i, val)

            else:
                warnings.warn(f"\n \t Inserted option with command line --{i} not found in Config.py use --help for more information.\n")

        eel.edit_gui_js("Setup finished", "progress_phase")

        try:
            results = predictit.main.predict()

            div = results["plot"]

            #                            content      p_tag    id_parent    id_created    label    classes
            if Config.print_best_model_result:
                eel.add_HTML_element(str(results["best"]), True, "content", "best_result", "Best result")

            eel.add_HTML_element(div, False, "content", "ploted_results", "Interactive plot", ["plot"])

            if Config.print_table:
                eel.add_HTML_element(results["models_table"], False, "content", "models_table", "Models results", "table")


            eel.execute("ploted_results")

            eel.add_delete_button("content")

            if Config.debug:

                eel.add_HTML_element(results["time_table"], False, "content", "time_parts_table", "Time schema of prediction", "table")
                eel.add_HTML_element(results['output'], True, "content", "printed_output", "Everything printed", "pre-wrapped")

        except Exception:

            eel.add_HTML_element(f"\n Error in making predictions - {traceback.format_exc()} \n", True, "progress_phase", "error-log", "Error log", "pre-wrapped")

    eel.start('index.html', port=0)  # mode='chrome-app'


if __name__ == "__main__":
    run_gui()

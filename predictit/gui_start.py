#%%

from pathlib import Path
import sys
import traceback
import mypythontools

import predictit

# Lazy imports
# import eel

# If some information from inside main(), define function here
def edit_gui_py(content, id):
    """Function that will change some element in html GUI and is callable from other py scripts.

    Args:
        content (str): New content.
        id (str): Id of changed element.
    """

    import eel

    eel.edit_gui_js(content, id)


def run_gui():
    """Start web based GUI."""

    import eel

    web_path = str(Path(__file__).resolve().parents[0] / "files_for_GUI")

    eel.init(web_path)

    this_path = Path(__file__).resolve().parents[1]
    this_path_string = str(this_path)

    # If used not as a library but as standalone framework, add path to be able to import predictit if not opened in folder
    sys.path.insert(0, this_path_string)

    config = predictit.config

    predictit.misc.GLOBAL_VARS._GUI = 1
    config.update(
        {
            "show_plot": False,
            "save_plot": False,
            "data": None,
            "table_settigs": {
                "tablefmt": "html",
                "floatfmt": ".3f",
                "numalign": "center",
                "stralign": "center",
            },
        }
    )

    # Functions with @eel prefix are usually called from main.js file.

    @eel.expose
    def make_predictions(configured):
        """Function that from web GUI button trigger the predictit main predict function and return results on GUI.

        Args:
            configured (dict): Some configuration values can be configured in GUI.
        """
        config.update(mypythontools.misc.json_to_py(configured))

        eel.edit_gui_js("Setup finished", "progress_phase")

        try:
            results = predictit.main.predict()

            div = results.plot

            #                            content      p_tag    id_parent    id_created    label    classes
            if config.print_result_details:
                eel.add_HTML_element(
                    str(results.best), True, "content", "best_result", "Best result"
                )

            eel.add_HTML_element(
                div, False, "content", "ploted_results", "Interactive plot", ["plot"]
            )

            if config.print_table:
                eel.add_HTML_element(
                    results.tables.detailed_results,
                    False,
                    "content",
                    "models_table",
                    "Models results",
                    "table",
                )

            eel.execute("ploted_results")

            eel.add_delete_button("content")

            eel.add_HTML_element(
                results.tables.time,
                False,
                "content",
                "time_parts_table",
                "Time schema of prediction",
                "table",
            )
            eel.add_HTML_element(
                results.output,
                True,
                "content",
                "printed_output",
                "Everything printed",
                "pre-wrapped",
            )

        except Exception:

            eel.add_HTML_element(
                f"\n Error in making predictions - {traceback.format_exc()} \n",
                True,
                "progress_phase",
                "error-log",
                "Error log",
                "pre-wrapped",
            )

    eel.start("index.html", port=0)  # mode='chrome-app'


if __name__ == "__main__":
    run_gui()

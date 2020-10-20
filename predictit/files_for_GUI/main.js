// function readd() {
//     var value = document.getElementById("read").value;
//     eel.just_return(value)(function(ret) {console.log(ret)});
// }

eel.expose(edit_gui_js);
function edit_gui_js(content, id) {
  document.getElementById(id).innerHTML = content;
}

eel.expose(add_HTML_element);
function add_HTML_element(
  content,
  into_paragraph,
  id_parent,
  id_created,
  label,
  added_class = "added"
) {
  var new_div = document.createElement("div");
  new_div.id = id_created;
  if (typeof added_class == "string") {
    new_div.classList.add(added_class);
  } else {
    new_div.classList.add(...added_class);
  }

  if (into_paragraph) {
    var new_p = document.createElement("p");
    new_p.innerHTML = content;
    new_div.appendChild(new_p);
  } else {
    new_div.innerHTML = content;
  }

  if (label) {
    var new_label = document.createElement("p");
    new_label.innerHTML = label;
    new_label.classList.add("label");
    document.getElementById(id_parent).appendChild(new_label);
  }

  document.getElementById(id_parent).appendChild(new_div);
}

eel.expose(execute);
function execute(id) {
  var codes = document.getElementById(id).getElementsByTagName("script");
  for (var i = 0; i < codes.length; i++) {
    eval(codes[i].text);
  }
}

// function make_visible() {

//     document.getElementById('plotly_plot').contentWindow.location.reload(true);
//     document.getElementById("plotly_plot").style.visibility = "visible"

// }

function predict() {
  var configurated = {};

  configurated["data"] = document.getElementById("csv_path").value;
  configurated["predicted_column"] = document.getElementById(
    "column_name"
  ).value;
  configurated["debug"] = document.getElementById("debug-config").checked;
  configurated["print_table"] = document.getElementById("table-config").checked;
  configurated["print_best_model_result"] = document.getElementById(
    "results-config"
  ).checked;

  configurated["use_config_preset"] = document.getElementById(
    "config-preset"
  ).value;

  eel.make_predictions(configurated)(function (ret) {
    console.log(ret);
  });
}

function clearcontent(id) {
  document.getElementById(id).innerHTML = "";
}

eel.expose(add_delete_button);
function add_delete_button(id) {
  var new_button = document.createElement("button");
  new_button.innerHTML = "Delete results";
  new_button.classList.add("button", "option-section");
  new_button.onclick = function () {
    clearcontent(id);
  };
  document.getElementById(id).appendChild(new_button);
}

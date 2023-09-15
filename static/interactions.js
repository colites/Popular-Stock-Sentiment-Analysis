
// ================================
// Results Page UI Manipulation Functions
// ================================


/**
 * Replaces the button for visualizations with the visualization options.
 */
function display_visualizations_choices(){
	const newButtons = document.querySelector(".modal-content");
	newButtons.innerHTML = `
			<div class="visualization-choice" id="visualization-choice-container">
                    		<div class="buttons-group">
                        		<button id="piechart_mentions" onclick="load_visualization_options('pie')">piechart</button>
                        		<button id="donutchart_mentions" onclick="load_visualization_options('donut')">donutchart</button>
                        		<button id="bargraph_sentiments" onclick="load_visualization_options('bar')">bargraph</button>
                        		<button id="linegraph_mentions" onclick="load_visualization_options('line')">linegraph</button>
                    		</div>
                	</div>
                	<div class="visualization-container">
                    		<img id="visualization" src="" alt="visualization">
                	</div>
    `;
}


//Need to refactor into smaller functions. Also need to refactor to use switch because it is just better. Do not know why I used else if.
function load_visualization_options(type){
    	const optionsContainer = document.querySelector(".modal-content");
    
    	let specificOptionsHTML = '';

    	if (type === 'pie') {
    		specificOptionsHTML = `
			<div class="dropdown_container">
            			<select id="y_measure_type" name="y_measure_type">
					<option value="" disabled selected>Select Variable to Measure</option>
                			<option value="Positive">Positive Mentions</option>
					<option value="Negative">Negative Mentions</option>
					<option value="Neutral">Neutral Mentions</option>
					<option value="Total">Total Mentions</option>
			        </select>
           		</div>
			<div class="dropdown_container">
            			<select id="x_measure_type" name="x_measure_type">
					<option value="" disabled selected>Choose a Variable to Measure First</option>
						<!-- Populate dynamically based on previous dropdown -->
			        </select>
           		</div>
            		<div class="dropdown_container">
                		<select id="select_symbol" name="select_symbol">
					<option value="" disabled selected>Select a Symbol</option>
					<option value="All">All(not working for now)</option>
 					<!-- Populate dynamically based on what is in the database -->
				</select>
           		</div>
    		`;
	} else if (type === 'donut') {
		specificOptionsHTML = `
			<div class="dropdown_container">
            			<select id="y_measure_type" name="y_measure_type">
					<option value="" disabled selected>Select Variable to Measure</option>
                			<option value="Positive">Positive Mentions</option>
					<option value="Negative">Negative Mentions</option>
					<option value="Neutral">Neutral Mentions</option>
					<option value="Total">Total Mentions</option>
			        </select>
           		</div>
			<div class="dropdown_container">
            			<select id="x_measure_type" name="x_measure_type">
					<option value="" disabled selected>Choose a Variable to Measure First</option>
						<!-- Populate dynamically based on previous dropdown -->
			        </select>
           		</div>
            		<div class="dropdown_container">
                		<select id="select_symbol" name="select_symbol">
					<option value="" disabled selected>Select a Symbol</option>
					<option value="All">All(not working for now)</option>
 					<!-- Populate dynamically based on what is in the database -->
				</select>
           		</div>
    		`;
	} else if (type === 'bar') {
		specificOptionsHTML = `
			<div class="dropdown_container">
            			<select id="y_measure_type" name="y_measure_type">
					<option value="" disabled selected>Select Variable to Measure</option>
                			<option value="Sentiments">All Sentiments</option>
			        </select>
           		</div>
			<div class="dropdown_container">
            			<select id="x_measure_type" name="x_measure_type">
					<option value="" disabled selected>Choose a Variable to Measure First</option>
						<!-- Populate dynamically based on previous dropdown -->
			        </select>
           		</div>
            		<div class="dropdown_container">
                		<select id="select_symbol" name="select_symbol">
					<option value="" disabled selected>Select a Symbol</option>
					<option value="All">All(Not working for now)</option>
 					<!-- Populate dynamically based on what is in the database -->
				</select>
           		</div>
    		`;
	} else {
		specificOptionsHTML = `
			<div class="dropdown_container">
            			<select id="y_measure_type" name="y_measure_type">
					<option value="" disabled selected>Select Variable to Measure</option>
					<option value="Sentiments">All Sentiments</option>
			        </select>
           		</div>
			<div class="dropdown_container">
            			<select id="x_measure_type" name="x_measure_type">
					<option value="" disabled selected>Choose a Variable to Measure First</option>
						<!-- Populate dynamically based on previous dropdown -->
			        </select>
           		</div>
            		<div class="dropdown_container">
                		<select id="select_symbol" name="select_symbol">
					<option value="" disabled selected>Select a Symbol</option>
					<option value="All">All(Not working for now)</option>
 					<!-- Populate dynamically based on what is in the database -->
				</select>
           		</div>
    		`;
	}
    	optionsContainer.innerHTML = `
        		<div class="visualization-options" id="visualization-options-container">
				<div id="date_range_selection">
        				<label for="start_date">Start Date:</label>
        				<input type="date" id="start_date_modal">
        
        				<label for="end_date">End Date:</label>
        				<input type="date" id="end_date_modal">
        
        				<button class="set-date-range-btn" onclick="update_date_range_modal()">Set Date Range</button>
   				</div>
            			${specificOptionsHTML}
				<div class="dropdown_container">
        				<select id="info_source" name="info_source">
						<option value="" disabled selected>Select an Information Source</option>
                				<option value="All">All</option>
                				<option value="Financial News">Financial News</option>
                				<option value="subreddit">Subreddit</option>
					</select>
        			</div>
            			<button onclick="display_visualizations_choices()">Back</button>
            			<button onclick="generate_visualization('${type}')">Submit</button>
       			</div>
        		<div class="visualization-container">
            			<img id="visualization" src="" alt="visualization">
        		</div>
    	`;

	const visualizationImage = document.getElementById('visualization');
    	visualizationImage.onload = function() {
        	this.style.display = 'block';
    	};
	get_all_symbols_dropdown()
	initialize_dropdowns()
}


function show_modal(modal_type) {
	if(modal_type === 'visualizations'){
		display_visualizations_choices();
	}

    	document.getElementById('modal').style.display = "block";
}


function initialize_dropdowns() {
   	 const yMeasureDropdown = document.getElementById('y_measure_type');
    	 yMeasureDropdown.addEventListener('change', handle_dropdown_change_all);
}


function handle_dropdown_change_all() {
    	const xMeasureDropdown = document.getElementById('x_measure_type');
    	const symbolDropdown = document.getElementById('select_symbol');

    	// Clear the existing options
    	xMeasureDropdown.innerHTML = "";
	
	// Add a placeholder option
    	let placeholderOption = document.createElement("option");
    	placeholderOption.value = "";
    	placeholderOption.text = "Select Variable to be Measured";
    	placeholderOption.selected = true;
    	placeholderOption.disabled = true;
    	xMeasureDropdown.appendChild(placeholderOption);

    	// Get the selected value from the y_measure dropdown
    	let selectedValue = this.value;

    	let options = [];

   	 // Choose which options to display based on the selected value
   	 if (selectedValue === "Positive" || selectedValue === "Negative" || selectedValue === "Neutral" || selectedValue === "Total" ) {
		// All the stocks serve as the x-axis for this measure
        	hide_dropdown("x_measure_type")
        	options = [
            		{ value: "None", text: "None" }
        	];

    	} else if (selectedValue === "Sentiments") {
        	// Can only do a singular stock with this option, the stock serves as the x-axis variable
		hide_dropdown("x_measure_type");
		options = [
            		{ value: "None", text: "None" }
        	];
    	}

    	// Populate the x_measure dropdown with the new options
    	options.forEach(function(option) {
        	let newOption = document.createElement("option");
        	newOption.value = option.value;
        	newOption.text = option.text;
        	xMeasureDropdown.appendChild(newOption);
    	});
}


function get_selected_value_from_dropdown(dropdownId) {
    	const dropdownElement = document.getElementById(dropdownId);
    	return dropdownElement.options[dropdownElement.selectedIndex].value;
}


function close_modal() {
   	document.getElementById('modal').style.display = "none";
}


// REMEMBER TO ADD DOCSTRINGS 
function update_date_range() {
    	const container = document.getElementById('buttons_container');
    	container.setAttribute('data-start-date', document.getElementById('start_date').value);
    	container.setAttribute('data-end-date', document.getElementById('end_date').value);
}


function update_date_range_modal() {
    	const container = document.getElementById('visualization-options-container');
    	container.setAttribute('data-start-date-modal', document.getElementById('start_date_modal').value);
    	container.setAttribute('data-end-date-modal', document.getElementById('end_date_modal').value);

}


function show_dropdown(Id) {
    	const dropdown = document.getElementById(Id);
    	dropdown.style.display = 'block';

}


function hide_dropdown(Id) {
    	const dropdown = document.getElementById(Id);
    	dropdown.style.display = 'none'; 
}


/**
 * Creates a dropdown menu with all the stock tickers
 */
async function get_all_symbols_dropdown() {
	const response = await fetch('/symbols');
    	const symbols = await response.json();
	
	const dropdown = document.getElementById('select_symbol');

    	// Populate the dropdown menu with stock symbols
    	symbols.forEach(symbol => {
        	const option = document.createElement('option');
        	option.setAttribute('value', symbol);
        	option.textContent = symbol;
        	dropdown.appendChild(option);
    	});
	
    	// Add an event listener to set an attribute indicating the chosen symbol
    	dropdown.addEventListener('change', (event) => {
        	const dropdown_container = document.getElementById('select_symbol');
        	dropdown_container.setAttribute('selected_symbol', event.target.value);
    	});
}


// ================================
// Data Fetching Functions
// ================================


/**
 * Fetches visualization data from the server and updates the visualization element.
 *
 * @param {string} route - The Flask route to fetch data from.
 * @param {string} selected_symbol - The stock symbol selected from the dropdown menu
 */
function get_visualizations_data(route) {
	const container = document.getElementById('visualization-options-container');
    	const startDate = container.getAttribute('data-start-date-modal');
    	const endDate = container.getAttribute('data-end-date-modal');
	const selectedInfoSource = get_selected_value_from_dropdown('info_source');
    	const selectedMeasureY = get_selected_value_from_dropdown('y_measure_type');
    	const selectedMeasureX = get_selected_value_from_dropdown('x_measure_type');
    	const selectedSymbol = get_selected_value_from_dropdown('select_symbol');

	// Validate inputs
    	if (!startDate || !endDate) {
        	alert('Please select both a start and end date');
        	return;
   	 }
    	if (!selectedSymbol) {
      	 	alert('Please select a stock symbol');
        	return;
    	}
    	if (!selectedInfoSource) {
        	alert('Please select an info source');
        	return;
    	}
    	if (!selectedMeasureY) {
        	alert('Please select an Y variable to measure');
        	return;
    	}
	if (document.getElementById('x_measure_type').style.display !== "none" && !selectedMeasureX) {
        	alert('Please select an X variable for measuring');
        	return;
    	}

    	// Construct query parameters dynamically
    	const params = new URLSearchParams({
        	start_date: startDate,
        	end_date: endDate,
        	symbol: selectedSymbol,
        	source_type: selectedInfoSource,
        	y_measure_type: selectedMeasureY,
        	x_measure_type: selectedMeasureX
    	});	
	
	// Sending the get request to the server and building the image using the response 
	fetch(`${route}?${params.toString()}`)
        	.then(response => {
			if (!response.ok) {
            			throw new Error(`Server error: ${response.status} ${response.statusText}`);
        		}
        		return response.json();
    		})
        	.then(data => {
			if (data === null) {
				throw new Error("No data received")
			}
			if (data && data.image) {
    				let VisualizationElement = document.getElementById('visualization');
    				VisualizationElement.src = 'data:image/png;base64,' + data.image;
			} else {
    				alert("The image could not be generated, invalid image parameters");
    				return;
			}
        	})
		.catch(error => {
			alert(error.message);
			return;
   		 });

}


function generate_visualization(type){
	let route = ''
	switch(type) {
		case 'pie':
			route = '/get_pie_chart';
			break;
		case 'donut':
			route = '/get_donut_chart';
			break;
		case 'bar':
			route = '/get_bar_graph';
			break;
		case 'line':
			route = '/get_line_graph';
			break;
	}
	get_visualizations_data(route)
}


// ================================
// Navigation and Input Page Functions
// ================================

/**
 * Changes the page to the Flask route that has the output HTML template.
 */
function output_screen(){
	window.location.href = 'output';
}


/**
 * Changes the page to the Flask route that has the Input HTML template.
 * Essentially goes back to the input screen.
 */
function input_screen(){
	window.location.href = '/';
}


/**
 * Updates the chosenOptions input value based on the checkboxes that the user has selected.
 *
 * @param {HTMLElement} checkbox - The checkbox that has changed.
 */
function handle_checkbox_change(checkbox) {

    	// Get the current value of the chosenOptions field as an array
    	var chosenOptions = document.getElementById("chosenOptions").value ? document.getElementById("chosenOptions").value.split(",") : [];

    	// Update the array based on the checkbox change
    	if (checkbox.checked) {
        	chosenOptions.push(checkbox.value);
    	} else {
        	chosenOptions = chosenOptions.filter(function(value, index, arr){ return value !== checkbox.value;});
    	}

    	document.getElementById("chosenOptions").value = chosenOptions.join(",");
}


/**
 * Creates the checkboxes if needed and updates the chosenOptions input value based on the option the user has selected.
 *
 * @param {HTMLElement} dropdown - The dropdown menu that has changed.
 */
function handle_dropdown_change(dropdown) {
    	var container = document.getElementById('checkboxContainer');
    	container.innerHTML = ''; 
    	var allOptions = ['bayes', 'SVM', 'distilbert', 'LR', 'RF'];

    	if (dropdown.value === 'Custom') {
        	document.getElementById("chosenOptions").value = "";

		// Create checkboxes for every model option
    		allOptions.forEach(function(option) {
            		var label = document.createElement('label');
            		label.innerHTML = option;
            		var input = document.createElement('input');
            		input.type = 'checkbox';
            		input.name = 'customOptions';
            		input.value = option;
            		input.onchange = function() {handle_checkbox_change(this);}; // event handler for checkboxes
            		container.appendChild(label);
            		container.appendChild(input);
            		container.appendChild(document.createElement('br')); 
        	});

    	} else if (dropdown.value === 'All') {
        	document.getElementById("chosenOptions").value = allOptions.join(",");

    	} else if (dropdown.value === 'Best') {
        	var bestOptions = ['SVM', 'distilbert'];
        	document.getElementById("chosenOptions").value = bestOptions.join(",");
    	}

}


/**
* Initializes the form defaults for the dropdown menu.
*/
function initialize_form() {
    // Set default values
    var allOptions = ['bayes', 'SVM', 'distilbert', 'LR', 'RF'];
    document.getElementById("chosenOptions").value = allOptions.join(",");
}


/**
 * Gets the input values from the form, sends a post request with the values, and then changes the page to the loading screen template on the flask route once a response is received.
 */
async function submit_form() {
	const num_posts = document.getElementById("num_posts").value;
  	const subreddit = document.getElementById("subreddit").value;
    	const models = document.getElementById("chosenOptions").value;
	
 	// If "Custom" is selected and no checkboxes are checked, show an alert and prevent form submission
    	if (document.getElementById('models').value === "Custom" && models === "") {
        alert("Please select at least one model when 'Custom' option is chosen.");
        return;  
    	}

  	const data = {
    		num_posts: num_posts,
    		subreddit: subreddit,
		models: models
  	};

  	const response = await fetch("/run_process_input", {
    		method: "POST",
    		headers: {
      		"Content-Type": "application/json"
    		},
    		body: JSON.stringify(data)
  	});

	if (response.ok) {
            window.location.href = "loading";

        } else {
            console.error("Error in the POST request:", response.statusText);
        }
}


/**
 * Calls the Flask route 'run' and then runs the output_screen function once the flask route is done running.
 */
function run_script(){
	fetch('run')
		.then(response => response.text())
        	.then(text => {
           		if (text === 'done') {
        	 		output_screen();
           		}
         	});
}


// ================================
// Results Tables Functions
// ================================


/**
 * Fetches sentiment statistics from the server and calls the display function.
 */
function sentiment_stats() {
	const container = document.getElementById('buttons_container');
    	const startDate = container.getAttribute('data-start-date');
    	const endDate = container.getAttribute('data-end-date');

	let url = '/sentiment_statistics';
    	if (startDate && endDate) {
        	url += `?start_date=${startDate}&end_date=${endDate}`;
    	}
	else {
		alert('Please select both a start and end date')
		return;
	}

	fetch(url)
        	.then(response => {
			if (!response.ok) {
            			throw new Error(`Server error: ${response.status} ${response.statusText}`);
        		}
        		return response.json();
    		})
        	.then(data => {
			if (data === null) {
				throw new Error("No data received")
			}
            		display_sentiment_statistics(data);
        	})
		.catch(error => {
			alert(error.message);
			return;
   		 });
}


/**
 * Fetches mention information from the server and calls the display function.
 */
function mention_information() {
	const container = document.getElementById('buttons_container');
    	const startDate = container.getAttribute('data-start-date');
    	const endDate = container.getAttribute('data-end-date');

	let url = '/mention_information';
    	if (startDate && endDate) {
        	url += `?start_date=${startDate}&end_date=${endDate}`;
    	}
	else {
		alert('Please select both a start and end date')
		return;
	}

	fetch(url)
        	.then(response => {
			if (!response.ok) {
            			throw new Error(`Server error: ${response.status} ${response.statusText}`);
        		}
        		return response.json();
    		})
        	.then(data => {
			if (data === null) {
				throw new Error("No data received")
			}
            		display_mention_information(data);
        	})
		.catch(error => {
			alert(error.message);
			return;
   		 });
}


/**
 * Creates an HTML table with the provided header and rows.
 *
 * @param {Array} header - An array of header column names.
 * @param {Array} rows - A 2D array of row data.
 * @returns {string} - The HTML string representation of the table.
 */
function create_table(header, rows) {
	let table = `
			<table>
      	       	<thead>
                  		<tr>
                     			${header.map(column => `<th>${column}</th>`).join('')}
                   		</tr>
                 	</thead>
                 	<tbody>
                   		${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}
                 	</tbody>
               	</table>`;

  return table;
}


/**
 * Displays an HTML table inside a specified element with the provided header and rows.
 *
 * @param {string} tableId - The ID of the element where the table should be displayed.
 * @param {Array} header - An array of header column names.
 * @param {Array} rows - A 2D array of row data.
 */
function display_table(tableId, header, rows) {
	const table = create_table(header, rows);
  	document.getElementById(tableId).innerHTML = table;
}


/**
 * Displays mention information data in a table format.
 *
 * @param {Array} data - The data to display in the table.
 */
function display_mention_information(data) {
  	const header = ['Date', 'Ticker', 'Mention Text', 'Sentiment', 'Source Type'];
  	const rows = data.map(item => [item.date, item.symbol, item.mention_text, item.sentiment, item.source_type]);
  	display_table('csv_output', header, rows);
}


/**
 * Displays sentiment statistics data in a table format.
 *
 * @param {Array} data - The data to display in the table.
 */
function display_sentiment_statistics(data) {
  	const header = ['Ticker', 'Total Mentions', 'Positive', 'Negative', 'Neutral'];
  	const rows = data.map(item => [
    		item.symbol,
    		item.total_mentions,
    		item.positive_mentions,
    		item.negative_mentions,
    		item.neutral_mentions,
  	]);

  	display_table('csv_output', header, rows);
}

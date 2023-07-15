/**
 * Replaces the button for visualizations with the visualization options.
 */
function visualizations(){
	const newButtons = document.getElementById("visualizations");
	newButtons.innerHTML = `
					<div class= "visualizations_container">
                				<button id="piechart_mentions" onclick="get_pie_charts()">piechart mentions</button>
            				</div>
            				<div class="visualizations_container">
                				<button id="donutchart_mentions" onclick="get_donut_charts()">donutchart mentions</button>
            				</div>
            				<div class="visualizations_container">
                				<button id="bargraph_sentiments" onclick="get_all_symbols_dropdown()">bargraph sentiments</button>
					</div>							
					<div class="dropdown_container">
 						<div id="dropdownContainer"></div>
					</div>`;

	newButtons.classList.remove("options");
  	newButtons.style.cursor = "default";
}


/**
 * Fetches visualization data from the server and updates the visualization element.
 *
 * @param {string} route - The Flask route to fetch data from.
 */
function get_visualizations_data(route) {
  	fetch(route)
		.then(response => response.json())
        	.then(data => {
            		let VisualizationElement = document.getElementById('visualization');
            		VisualizationElement.src = 'data:image/png;base64,' + data.image;
        	});
}


/**
 * Fetches visualization data from the server and updates the visualization element for the symbol.
 *
 * @param {string} route - The Flask route to fetch data from.
 * @param {string} selected_symbol - The stock symbol selected from the dropdown menu
 */
function get_visualizations_data_symbol(route, selected_symbol) {
  	fetch(`${route}?symbol=${selected_symbol}`)
    		.then(response => response.json())
    		.then(data => {
      			let VisualizationElement = document.getElementById('visualization');
      			VisualizationElement.src = 'data:image/png;base64,' + data.image;
    		});
}


/**
 * Sends an AJAX request to the Flask route that receives the bar graph visualization data.
 * @param {string} selected_symbol - The stock symbol selected from the dropdown menu
 */
function get_bar_graphs(selected_symbol) {
    	get_visualizations_data_symbol('/get_bar_graph', selected_symbol);
}


/**
 * Creates a dropdown menu with all the stock tickers
 */
async function get_all_symbols_dropdown() {
	const response = await fetch('/symbols');
    	const symbols = await response.json();

    	const dropdown = document.createElement('select');
   	dropdown.setAttribute('id', 'stockSymbols');

    	// Add a default option to the dropdown
    	const defaultOption = document.createElement('option');
    	defaultOption.setAttribute('value', '');
    	defaultOption.textContent = 'Select a stock symbol';
    	dropdown.appendChild(defaultOption);

    	// Populate the dropdown menu with stock symbols
    	symbols.forEach(symbol => {
        	const option = document.createElement('option');
        	option.setAttribute('value', symbol);
        	option.textContent = symbol;
        	dropdown.appendChild(option);
    	});

    	// Add the dropdown menu to the container
    	const dropdownContainer = document.getElementById('dropdownContainer');
    	dropdownContainer.innerHTML = ''; 
    	dropdownContainer.appendChild(dropdown);

    	// Add an event listener to handle dropdown selection and call the route for making the bar graph
    	dropdown.addEventListener('change', (event) => {
        	const selected_symbol = event.target.value;
        	if (selected_symbol) {
            		get_bar_graphs(selected_symbol); 
        	}
    	});
}


/**
 * Sends an AJAX request to the Flask route that receives the pie chart visualization data.
 */
function get_pie_charts() {
	get_visualizations_data('/get_pie_chart');
}


/**
 * Sends an AJAX request to the Flask route that receives the donut chart visualization data.
 */
function get_donut_charts() {
	get_visualizations_data('/get_donut_chart');
}


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
function handleCheckboxChange(checkbox) {

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
function handleDropdownChange(dropdown) {
    	console.log(dropdown.value); // Add this line
    	var container = document.getElementById('checkboxContainer');
    	container.innerHTML = ''; 
    	var allOptions = ['bayes', 'SVM', 'distilbert'];

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
            		input.onchange = function() {handleCheckboxChange(this);}; // Add event handler for checkboxes
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
    var allOptions = ['bayes', 'SVM', 'distilbert'];
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


/**
 * Fetches sentiment statistics from the server and calls the display function.
 */
function sentiment_stats() {
    	fetch('/sentiment_statistics')
        	.then(response => response.json())
        	.then(data => {
            		display_sentiment_statistics(data);
        	});
}


/**
 * Fetches mention information from the server and calls the display function.
 */
function mention_information() {
	fetch('/mention_information')
        	.then(response => response.json())
        	.then(data => {
            		display_mention_information(data);
        	});
}


/**
 * Creates an HTML table with the provided header and rows.
 *
 * @param {Array} header - An array of header column names.
 * @param {Array} rows - A 2D array of row data.
 * @returns {string} - The HTML string representation of the table.
 */
function createTable(header, rows) {
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
function displayTable(tableId, header, rows) {
	const table = createTable(header, rows);
  	document.getElementById(tableId).innerHTML = table;
}


/**
 * Displays mention information data in a table format.
 *
 * @param {Array} data - The data to display in the table.
 */
function display_mention_information(data) {
  	const header = ['Ticker', 'Mention Text', 'Sentiment'];
  	const rows = data.map(item => [item.symbol, item.mention_text, item.sentiment]);
  	displayTable('csv_output', header, rows);
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

  	displayTable('csv_output', header, rows);
}

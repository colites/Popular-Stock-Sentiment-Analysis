function mention_information(){
	document.getElementById('csv_output').innerHTML = 'mention'

}

function sentiment_stats(){
	document.getElementById('csv_output').innerHTML = 'sentiment'

} 

function output_screen(){
	window.location.href = 'output'
}

function loading_screen(){
	window.location.href = 'loading'
} 

function run_script(){
	fetch('run').then(output_screen())
}

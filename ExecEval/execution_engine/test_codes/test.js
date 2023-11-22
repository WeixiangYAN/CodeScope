process.stdin.resume();
process.stdin.setEncoding('utf-8');

let inputString = '';
let currentLine = 0;
print = console.log;
process.stdin.on('data', inputStdin => {
	inputString += inputStdin;
});

process.stdin.on('end', _ => {
	inputString = inputString
		.trim()
		.split('\n')
		.map(string => {
			return string.trim();
		});

	main();
});

function readline() {
	return inputString[currentLine++];
}

function main() {
	s = readline();
	// print(s.split(' '));
	let [a, b] = s.split(' ').map(x => parseInt(x));
	print(a + b);
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('user-message');
    const responseDiv = document.getElementById('response');

    form.addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent the form from submitting the traditional way

        const message = input.value; // Get the message from the input field

        // Send the user message to the Flask backend
        fetch('/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ msg: message }), // Send the message as JSON
        })
            .then(response => response.json()) // Parse the JSON response from the server
            .then(data => {
                responseDiv.innerText = data.response; // Display the response in the div
            })
            .catch(error => {
                console.error('Error:', error); // Log any errors to the console
                responseDiv.innerText = 'An error occurred while processing your request.';
            });

        input.value = ''; // Clear the input field
    });
});

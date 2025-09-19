


document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    const micBtn = document.getElementById('mic-btn');

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognition ? new SpeechRecognition() : null;

    if (recognition) {
        recognition.continuous = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            micBtn.classList.add('active');
            userInput.placeholder = "Listening...";
        };
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            chatForm.dispatchEvent(new Event('submit'));
        };
        recognition.onend = () => {
            micBtn.classList.remove('active');
            userInput.placeholder = "Type your legal question...";
        };
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            micBtn.classList.remove('active');
            userInput.placeholder = "Type your legal question...";
            displayMessage("Speech recognition error. Please try again or type your message.", 'assistant');
        };

        micBtn.addEventListener('click', () => {
            if (micBtn.classList.contains('active')) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
    } else {
        micBtn.style.display = 'none';
        console.warn("Web Speech API not supported in this browser.");
    }
    
    function playAudioResponse(base64Audio) {
        if (!base64Audio) {
            console.error("No audio data to play.");
            return;
        }
        const audio = new Audio(`data:audio/mpeg;base64,${base64Audio}`);
        audio.play().catch(e => console.error("Error playing audio:", e));
    }
    
    function displayTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'ai-message', 'typing-indicator');
        typingIndicator.textContent = 'Assistant is typing...';
        chatHistory.appendChild(typingIndicator);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    // UPDATED: Function to display a message on the screen, now handling audio data
    // Function to display a message on the screen, now handling audio data
function displayMessage(text, sender, audioData = null) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    if (sender === 'user') {
        messageElement.classList.add('user-message');
    } else {
        messageElement.classList.add('ai-message');
    }

    const textNode = document.createElement('span');
    textNode.textContent = text;
    messageElement.appendChild(textNode);
    
    if (sender === 'assistant' && audioData) {
        const playButton = document.createElement('button');
        playButton.classList.add('play-audio-btn');
        // Corrected line: Add aria-label for accessibility
        playButton.setAttribute('aria-label', 'Play audio response');
        playButton.innerHTML = '<i class="fas fa-volume-up"></i>';
        playButton.onclick = () => playAudioResponse(audioData);
        messageElement.appendChild(playButton);
    }

    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

    async function loadChatHistory() {
        try {
            const response = await fetch('http://127.0.0.1:5000/get_history');
            const data = await response.json();
            
            if (data.chatHistory && data.chatHistory.length > 0) {
                data.chatHistory.forEach(message => {
                    displayMessage(message.content, message.role);
                });
            } else {
                displayMessage("Hi there! I'm your Legal AI Assistant. How can I help you today?", 'assistant');
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
            displayMessage("An error occurred. Please check the server.", 'assistant');
        }
    }

    // UPDATED: Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();

        if (query) {
            displayMessage(query, 'user');
            userInput.value = '';
            displayTypingIndicator();

            try {
                const response = await fetch('http://127.0.0.1:5000/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });

                const data = await response.json();
                console.log("Data received from server:", data);
                removeTypingIndicator();

                if (data.chatHistory) {
                    const latestAiMessage = data.chatHistory[data.chatHistory.length - 1];
                    
                    // Clear and re-render the chat window for continuity
                    chatHistory.innerHTML = '';
                    loadChatHistory();
                    
                    // Display the new AI message with the audio button
                    displayMessage(latestAiMessage.content, latestAiMessage.role, data.audioResponse);
                } else if (data.error) {
                    displayMessage(`Server Error: ${data.error}`, 'assistant');
                    console.error('Server returned an error:', data.error);
                }
            } catch (error) {
                removeTypingIndicator();
                console.error('Error during chat:', error);
                displayMessage("An error occurred. Please check the server.", 'assistant');
            }
        }
    });

    loadChatHistory();
});
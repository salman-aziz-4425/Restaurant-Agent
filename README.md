# Restaurant Multi Agent System

A sophisticated AI-powered restaurant management system that provides voice-interactive agents for handling reservations, takeaway orders, and payments.

## Features

### 🤖 Specialized AI Agents
- **Greeter Agent**: Welcomes customers and directs them to appropriate services
- **Reservation Agent**: Handles table bookings and reservation management
- **Takeaway Agent**: Processes food orders for pickup
- **Checkout Agent**: Manages payment processing

### 🎯 Key Capabilities
- Natural voice interactions using ElevenLabs TTS and Deepgram STT
- Intelligent context preservation between agent transfers
- LRU caching for optimized performance
- Concurrent request handling with WorkerPool
- Secure payment processing
- Real-time agent status monitoring

### 🔊 Voice System
- Unique voice identities for each agent type
- High-quality text-to-speech using ElevenLabs
- Accurate speech recognition with Deepgram

## Technology Stack

- **Backend**: Python with FastAPI
- **AI/ML**: OpenAI GPT-4
- **Voice Processing**: 
  - Text-to-Speech: ElevenLabs
  - Speech-to-Text: Deepgram
  - Voice Activity Detection: Silero VAD
- **Real-time Communication**: LiveKit
- **Concurrency**: AsyncIO

## Prerequisites

- Python 3.8+
- LiveKit account and server setup
- OpenAI API access
- ElevenLabs API access
- Deepgram API access

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/restaurant-agent.git
   cd restaurant-agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   - Copy `.env.example` to `.env`
   - Fill in your API keys and configuration:
     ```
     LIVEKIT_API_KEY=your_livekit_api_key
     LIVEKIT_API_SECRET=your_livekit_api_secret
     LIVEKIT_URL=your_livekit_url
     OPENAI_API_KEY=your_openai_api_key
     ELEVENLABS_API_KEY=your_elevenlabs_api_key
     ```

## Usage

1. **Start the server**
   ```bash
   uvicorn src.api.router:app --reload
   ```

2. **Access the API**
   - The API will be available at `http://localhost:8000`
   - API documentation is available at `http://localhost:8000/docs`

3. **Monitor Workers**
   - Check worker status at `/workers/status`
   - Default maximum concurrent workers: 5

## API Endpoints

- `POST /token`: Generate access token for room connection
- `GET /workers/status`: Get status of all workers in the pool

## Architecture

The system uses a multi-agent architecture where each agent specializes in specific tasks:

1. **Greeter Agent**: Entry point for customer interactions
2. **Reservation Agent**: Handles booking flow
3. **Takeaway Agent**: Manages order processing
4. **Checkout Agent**: Handles payment flow

Each agent maintains its context and can seamlessly transfer customers to other agents while preserving conversation history.

## Worker Pool Management

- Implements a WorkerPool class for handling concurrent requests
- Maximum 5 concurrent workers by default
- Queue system for additional requests
- Metrics tracking for worker performance

## Security

- Environment variables for sensitive data
- Secure credit card information handling
- API key management
- Push protection for sensitive data

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LiveKit for real-time communication
- OpenAI for language processing
- ElevenLabs for voice synthesis
- Deepgram for speech recognition 

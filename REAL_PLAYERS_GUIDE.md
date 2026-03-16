# 🌟 Using Real Player Names - Complete Guide

## ✅ What's Working Now

Your API now supports **searching and predicting by real player names** instead of just IDs!

### 🎯 Available Features:

1. **Search Players by Name** - Find players using partial or full names
2. **Predict by Player Name** - Get predictions using player names directly
3. **Real Player Database** - 38 real football stars with accurate data

---

## 🔍 Feature 1: Search Players

### Endpoint: `POST /search`

Search for players by name (case-insensitive, partial matching).

### Example Requests:

#### Search for "Messi":
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Messi", "limit": 5}'
```

**Response:**
```json
{
  "success": true,
  "players": [
    {
      "player_id": "real_000",
      "player_name": "Lionel Messi",
      "age": 36,
      "position": "Forward",
      "club": "Inter Miami",
      "market_value": 53900000.0,
      "nationality": "Argentina"
    }
  ],
  "total_found": 1,
  "message": "Found 1 player(s) matching 'Messi'"
}
```

#### Search for "De" (finds all players with "De" in name):
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"player_name": "De", "limit": 10}'
```

**Finds:** Kevin De Bruyne, Frenkie de Jong, Eder Militão, etc.

---

## 🎯 Feature 2: Predict by Player Name

### Endpoint: `POST /predict/by-name`

Get transfer value predictions using player names.

### Example Request:

```bash
curl -X POST "http://localhost:8000/predict/by-name" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Erling Haaland",
    "model_type": "ensemble",
    "include_confidence": true,
    "fetch_realtime_data": false
  }'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "player_id": "real_003",
    "predicted_value": 135200000.0,
    "confidence_interval": {
      "lower": 121680000.0,
      "upper": 148720000.0,
      "confidence_level": 0.95
    },
    "model_used": "ensemble",
    "timestamp": "2026-03-11T10:30:00"
  },
  "message": "Prediction successful (Player: Erling Haaland)"
}
```

---

## 👥 Available Real Players

### Forwards (10 players):
- Lionel Messi (Inter Miami)
- Cristiano Ronaldo (Al Nassr)
- Kylian Mbappé (Real Madrid)
- Erling Haaland (Manchester City)
- Harry Kane (Bayern Munich)
- Robert Lewandowski (Barcelona)
- Mohamed Salah (Liverpool)
- Vinícius Júnior (Real Madrid)
- Neymar Jr (Al Hilal)
- Karim Benzema (Al Ittihad)

### Midfielders (10 players):
- Kevin De Bruyne (Manchester City)
- Luka Modrić (Real Madrid)
- Jude Bellingham (Real Madrid)
- Bruno Fernandes (Manchester United)
- Pedri (Barcelona)
- Frenkie de Jong (Barcelona)
- Casemiro (Manchester United)
- Rodri (Manchester City)
- Toni Kroos (Real Madrid)
- İlkay Gündoğan (Barcelona)

### Defenders (10 players):
- Virgil van Dijk (Liverpool)
- Rúben Dias (Manchester City)
- Antonio Rüdiger (Real Madrid)
- Marquinhos (Paris Saint-Germain)
- Joško Gvardiol (Manchester City)
- William Saliba (Arsenal)
- Eder Militão (Real Madrid)
- Kim Min-jae (Bayern Munich)
- Alessandro Bastoni (Inter Milan)
- Theo Hernández (AC Milan)

### Goalkeepers (8 players):
- Thibaut Courtois (Real Madrid)
- Alisson Becker (Liverpool)
- Ederson (Manchester City)
- Marc-André ter Stegen (Barcelona)
- Gianluigi Donnarumma (Paris Saint-Germain)
- Mike Maignan (AC Milan)
- Jan Oblak (Atlético Madrid)
- Emiliano Martínez (Aston Villa)

---

## 🎨 Using Swagger UI (Easiest Way!)

1. **Open:** http://localhost:8000/docs

2. **Try Search:**
   - Click on `POST /search`
   - Click "Try it out"
   - Enter: `{"player_name": "Messi", "limit": 5}`
   - Click "Execute"
   - See results!

3. **Try Prediction:**
   - Click on `POST /predict/by-name`
   - Click "Try it out"
   - Enter player name: `Kylian Mbappé`
   - Click "Execute"
   - See predicted transfer value!

---

## 🔄 Real-Time Data (Future Enhancement)

The `fetch_realtime_data` parameter is ready for integration:

```json
{
  "player_name": "Erling Haaland",
  "fetch_realtime_data": true
}
```

**When implemented, this will:**
1. Fetch latest stats from StatsBomb API
2. Get current market value from Transfermarkt
3. Collect recent social media sentiment from Twitter
4. Update injury status
5. Generate fresh prediction with latest data

**Current Status:** Uses existing database (faster, but not real-time)

---

## 📊 Search Tips

### Exact Match:
```json
{"player_name": "Lionel Messi"}
```

### Partial Match:
```json
{"player_name": "Messi"}  // Finds "Lionel Messi"
```

### Last Name Only:
```json
{"player_name": "Ronaldo"}  // Finds "Cristiano Ronaldo"
```

### Find All Players from a Club:
```json
{"player_name": "Real Madrid", "limit": 20}
```

### Find Players by Position:
```json
{"player_name": "Forward", "limit": 10}
```

---

## 🚀 Quick Test Commands

### Test 1: Search for Messi
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Messi"}'
```

### Test 2: Predict Haaland's Value
```bash
curl -X POST "http://localhost:8000/predict/by-name" \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Erling Haaland"}'
```

### Test 3: Find All Defenders
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Defender", "limit": 20}'
```

---

## 🎯 Next Steps

### 1. Add More Players
Edit `create_real_players_data.py` and add more players to the `REAL_PLAYERS` list.

### 2. Integrate Real-Time Data
Implement the data fetching in `src/api/app.py`:
- Connect to Transfermarkt API
- Fetch StatsBomb latest stats
- Get Twitter sentiment
- Update injury database

### 3. Train Real Models
Replace mock predictions with trained models:
```bash
python scripts/train_lstm.py
python scripts/run_pipeline.py --train-ensemble
```

### 4. Build a Frontend
Create a web interface where users can:
- Search for players by name
- See player profiles
- View predicted transfer values
- Compare multiple players

---

## 📝 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/search` | POST | Search players by name |
| `/predict` | POST | Predict by player ID |
| `/predict/by-name` | POST | Predict by player name |
| `/predict/batch` | POST | Batch predictions |

---

## ✅ What's Working

✅ Search by player name (partial matching)
✅ 38 real players with accurate data
✅ Position, club, nationality information
✅ Market value estimates
✅ Interactive Swagger UI documentation

## ⚠️ What's Pending

⏳ Real-time data fetching (framework ready)
⏳ Trained ML models (using mock predictions currently)
⏳ Historical data tracking
⏳ Player comparison features

---

**Start testing now:** http://localhost:8000/docs

Search for your favorite players by name! 🌟
